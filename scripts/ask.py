from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from vn_labor_law_ai_assistant.answering import build_messages, parse_answer_payload
from vn_labor_law_ai_assistant.llm import (
    DEFAULT_PROVIDER,
    SUPPORTED_PROVIDERS,
    chat_completion,
    provider_model_label,
)
from vn_labor_law_ai_assistant.retriever import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_RERANKER_TOP_N,
    HybridRetriever,
    format_intent_summary,
    format_context_for_prompt,
    select_contexts_for_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a Vietnamese labor-law question against the local hybrid index."
    )
    parser.add_argument("question", nargs="?", help="Natural-language question in Vietnamese.")
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("artifacts/index"),
        help="Path to artifacts/index or directly to current.json.",
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=SUPPORTED_PROVIDERS,
        help=f"LLM provider for answer generation (default: {DEFAULT_PROVIDER}).",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model name to use for answer generation. Leave empty to use the provider default.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of hybrid search hits before context assembly.",
    )
    parser.add_argument(
        "--prefetch-limit",
        type=int,
        default=24,
        help="Candidate count for each dense/sparse branch before fusion.",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=6,
        help="Maximum number of assembled context blocks sent to the LLM.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help=f"Hard character budget for assembled prompt context (default: {DEFAULT_MAX_CONTEXT_CHARS}).",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=DEFAULT_MAX_CONTEXT_TOKENS,
        help=(
            "Soft token budget for assembled prompt context. Lower-ranked context blocks are "
            f"dropped before truncation (default: {DEFAULT_MAX_CONTEXT_TOKENS})."
        ),
    )
    parser.add_argument(
        "--reranker-model",
        default=os.getenv("RERANKER_MODEL", "").strip(),
        help=(
            "Optional cross-encoder reranker model applied after Hybrid Search, e.g. "
            "`BAAI/bge-reranker-v2-m3`. Leave empty to disable."
        ),
    )
    parser.add_argument(
        "--reranker-top-n",
        type=int,
        default=max(1, int(os.getenv("RERANKER_TOP_N", str(DEFAULT_RERANKER_TOP_N)))),
        help=f"Number of top candidates sent to the reranker (default: {DEFAULT_RERANKER_TOP_N}).",
    )
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Only print retrieved context and skip the LLM generation step.",
    )
    parser.add_argument(
        "--show-hits",
        action="store_true",
        help="Print raw hybrid hits before the final answer.",
    )
    return parser.parse_args()


def print_answer_block(label: str, answer_payload) -> None:
    print(f"\n===== {label} =====")
    print("\nTra loi:")
    print(answer_payload.answer or "Khong co noi dung tra loi.")

    print("\nCo so phap ly:")
    if answer_payload.legal_basis:
        for citation in answer_payload.legal_basis:
            print(f"- {citation}")
    else:
        print("- Khong co co so phap ly hop le duoc xac nhan tu output cua mo hinh.")

    print(f"\nInsufficient context: {'co' if answer_payload.insufficient_context else 'khong'}")
    if answer_payload.notes:
        print(f"Ghi chu: {answer_payload.notes}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")

    args = parse_args()
    question = args.question or input("Nhap cau hoi: ").strip()
    if not question:
        raise SystemExit("Cau hoi khong duoc de trong.")

    retriever = HybridRetriever(
        index_path=args.index_path,
        reranker_model=args.reranker_model,
        reranker_top_n=args.reranker_top_n,
    )
    try:
        result = retriever.retrieve(
            question,
            top_k=args.top_k,
            prefetch_limit=args.prefetch_limit,
        )

        print(f"Query routing: {format_intent_summary(result.intent)}")
        if retriever.reranker_enabled:
            print(f"Semantic reranker: {retriever.reranker_model_name} (top {args.reranker_top_n})")

        if args.show_hits:
            print("\nTop hits:")
            for hit in result.hits:
                print(f"- {hit.score:.4f} | {hit.citation_text}")

        contexts = select_contexts_for_prompt(
            result.contexts,
            max_contexts=args.max_contexts,
            max_chars=args.max_context_chars,
            max_tokens=args.max_context_tokens,
        )
        if not contexts:
            print("\nKhong tim thay ngu canh phu hop trong index.")
            return

        print("\nNgu canh truy hoi:")
        for context in contexts:
            print(f"- {context.citation_text}")

        if args.retrieve_only:
            print("\n----- CONTEXT -----")
            print(
                format_context_for_prompt(
                    contexts,
                    max_chars=args.max_context_chars,
                    max_tokens=args.max_context_tokens,
                )
            )
            return

        response = chat_completion(
            provider=args.provider,
            model=args.model,
            messages=build_messages(
                question,
                contexts,
                max_context_chars=args.max_context_chars,
                max_context_tokens=args.max_context_tokens,
            ),
            temperature=0,
        )
        parsed = parse_answer_payload(response.content, contexts)
        print_answer_block(provider_model_label(response.provider, response.model), parsed)
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
