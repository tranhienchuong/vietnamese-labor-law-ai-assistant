from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import ollama

from vn_labor_law_ai_assistant.answering import build_messages, parse_answer_payload
from vn_labor_law_ai_assistant.retriever import (
    DEFAULT_MAX_CONTEXT_CHARS,
    HybridRetriever,
    format_intent_summary,
    format_context_for_prompt,
    select_contexts_for_prompt,
)


DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")


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
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name to use for answer generation (default: {DEFAULT_MODEL}).",
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
        "--retrieve-only",
        action="store_true",
        help="Only print retrieved context and skip the Ollama generation step.",
    )
    parser.add_argument(
        "--show-hits",
        action="store_true",
        help="Print raw hybrid hits before the final answer.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")

    args = parse_args()
    question = args.question or input("Nhap cau hoi: ").strip()
    if not question:
        raise SystemExit("Cau hoi khong duoc de trong.")

    retriever = HybridRetriever(index_path=args.index_path)
    try:
        result = retriever.retrieve(
            question,
            top_k=args.top_k,
            prefetch_limit=args.prefetch_limit,
        )

        print(f"Query routing: {format_intent_summary(result.intent)}")

        if args.show_hits:
            print("\nTop hits:")
            for hit in result.hits:
                print(f"- {hit.score:.4f} | {hit.citation_text}")

        contexts = select_contexts_for_prompt(
            result.contexts,
            max_contexts=args.max_contexts,
            max_chars=args.max_context_chars,
        )
        if not contexts:
            print("\nKhong tim thay ngu canh phu hop trong index.")
            return

        print("\nNgu canh truy hoi:")
        for context in contexts:
            print(f"- {context.citation_text}")

        if args.retrieve_only:
            print("\n----- CONTEXT -----")
            print(format_context_for_prompt(contexts, max_chars=args.max_context_chars))
            return

        response = ollama.chat(
            model=args.model,
            format="json",
            options={"temperature": 0},
            messages=build_messages(question, contexts, max_context_chars=args.max_context_chars),
        )
        parsed = parse_answer_payload(response["message"]["content"], contexts)

        print("\nTra loi:")
        print(parsed.answer or "Khong co noi dung tra loi.")

        print("\nCo so phap ly:")
        if parsed.legal_basis:
            for citation in parsed.legal_basis:
                print(f"- {citation}")
        else:
            print("- Khong co co so phap ly hop le duoc xac nhan tu output cua mo hinh.")

        print(f"\nInsufficient context: {'co' if parsed.insufficient_context else 'khong'}")
        if parsed.notes:
            print(f"Ghi chu: {parsed.notes}")
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
