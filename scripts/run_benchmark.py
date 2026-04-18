from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import sys

from vn_labor_law_ai_assistant.answering import build_messages, parse_answer_payload
from vn_labor_law_ai_assistant.evaluation import (
    BENCHMARK_JSONL_NAME,
    BenchmarkCase,
    document_families_from_chunk_paths,
    expected_citations,
    expected_citation_scope,
    expected_citations_in_scope,
    expected_citations_out_of_scope,
    load_benchmark_jsonl,
    retrieval_hit_at_k,
    score_citation_correctness_for_scope,
    write_results_csv,
    write_results_jsonl,
)
from vn_labor_law_ai_assistant.llm import (
    DEFAULT_PROVIDER,
    SUPPORTED_PROVIDERS,
    chat_completion,
    provider_model_label,
)
from vn_labor_law_ai_assistant.retriever import (
    DEFAULT_MAX_CONTEXT_CHARS,
    HybridRetriever,
    select_contexts_for_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the current RAG pipeline against the golden benchmark."
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("eval/data") / BENCHMARK_JSONL_NAME,
        help=f"Path to imported benchmark JSONL (default: eval/data/{BENCHMARK_JSONL_NAME}).",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("artifacts/index"),
        help="Path to artifacts/index or current.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results"),
        help="Directory for benchmark outputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Retrieval cut-off for hit@k and citation review.",
    )
    parser.add_argument(
        "--prefetch-limit",
        type=int,
        default=24,
        help="Candidate count per dense/sparse branch before fusion.",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=6,
        help="Maximum number of contexts considered for generation.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help=f"Hard character budget for prompt context (default: {DEFAULT_MAX_CONTEXT_CHARS}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional number of benchmark cases to run. 0 means all.",
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=SUPPORTED_PROVIDERS,
        help=f"LLM provider for answer generation when --model is set (default: {DEFAULT_PROVIDER}).",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional model name. Leave empty for retrieval-only benchmarking.",
    )
    parser.add_argument(
        "--evaluator",
        default="auto-benchmark",
        help="Evaluator label stored in output rows.",
    )
    return parser.parse_args()


def build_result_row(
    *,
    case: BenchmarkCase,
    model_version: str,
    top_hit_citations: list[str],
    retrieval_hit: bool | None,
    evaluator: str,
    expected_citations_all: tuple[str, ...],
    expected_citations_scoped: tuple[str, ...],
    expected_citations_excluded: tuple[str, ...],
    case_scope: str,
    generated_answer: str = "",
    generated_legal_basis: tuple[str, ...] = (),
    insufficient_context: str = "",
) -> dict[str, object]:
    if retrieval_hit is None:
        retrieval_hit_value = "N/A"
    else:
        retrieval_hit_value = "Yes" if retrieval_hit else "No"

    return {
        "id": case.id,
        "model_version": model_version,
        "retrieval_hit_at_5": retrieval_hit_value,
        "citation_correct": "",
        "answer_correct": "",
        "hallucination_flag": "",
        "abstention_correct": "",
        "clarity_score_1_5": "",
        "format_score_1_5": "",
        "final_score_10": "",
        "evaluator": evaluator,
        "comments": "",
        "question": case.question,
        "expected_citations": " | ".join(expected_citations_all),
        "expected_citations_in_scope": " | ".join(expected_citations_scoped),
        "expected_citations_out_of_scope": " | ".join(expected_citations_excluded),
        "case_scope": case_scope,
        "retrieved_citations": " | ".join(top_hit_citations),
        "generated_answer": generated_answer,
        "generated_legal_basis": " | ".join(generated_legal_basis),
        "insufficient_context": insufficient_context,
    }


def slugify_model_version(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())
    return cleaned.strip("-") or "run"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    cases = load_benchmark_jsonl(args.benchmark_path)
    if args.limit > 0:
        cases = cases[: args.limit]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    retriever = HybridRetriever(index_path=args.index_path)
    rows: list[dict[str, object]] = []
    model_version = provider_model_label(args.provider, args.model) if args.model else "retrieval-only"
    output_stem = f"benchmark_{slugify_model_version(model_version)}_{timestamp}"
    output_jsonl = args.output_dir / f"{output_stem}.jsonl"
    output_csv = args.output_dir / f"{output_stem}.csv"
    allowed_document_families = document_families_from_chunk_paths(
        retriever.manifest.get("chunk_paths", [])
    )

    try:
        for index, case in enumerate(cases, start=1):
            retrieval_result = retriever.retrieve(
                case.question,
                top_k=max(args.top_k, args.max_contexts),
                prefetch_limit=args.prefetch_limit,
            )
            top_hit_citations = [hit.citation_text for hit in retrieval_result.hits[: args.top_k]]
            expected_all = expected_citations(case)
            expected_scoped = expected_citations_in_scope(
                case,
                allowed_document_families=allowed_document_families,
            )
            expected_excluded = expected_citations_out_of_scope(
                case,
                allowed_document_families=allowed_document_families,
            )
            case_scope = expected_citation_scope(
                case,
                allowed_document_families=allowed_document_families,
            )
            retrieval_hit = retrieval_hit_at_k(
                case,
                top_hit_citations,
                k=args.top_k,
                allowed_document_families=allowed_document_families,
            )

            generated_answer = ""
            generated_legal_basis: tuple[str, ...] = ()
            insufficient_context = ""
            citation_correct = ""
            abstention_correct = ""
            hallucination_flag = ""

            if args.model:
                contexts = select_contexts_for_prompt(
                    retrieval_result.contexts,
                    max_contexts=args.max_contexts,
                    max_chars=args.max_context_chars,
                )
                response = chat_completion(
                    provider=args.provider,
                    model=args.model,
                    messages=build_messages(
                        case.question,
                        contexts,
                        max_context_chars=args.max_context_chars,
                    ),
                    temperature=0,
                )
                parsed = parse_answer_payload(response.content, contexts)
                generated_answer = parsed.answer
                generated_legal_basis = parsed.legal_basis
                insufficient_context = "Yes" if parsed.insufficient_context else "No"
                citation_correct = score_citation_correctness_for_scope(
                    case,
                    parsed.legal_basis,
                    allowed_document_families=allowed_document_families,
                )
                abstention_correct = (
                    "yes" if parsed.insufficient_context == case.abstain_required else "no"
                )
                if citation_correct == "na":
                    hallucination_flag = "na"
                else:
                    hallucination_flag = (
                        "yes"
                        if (not parsed.insufficient_context and citation_correct == "no")
                        else "no"
                    )

            row = build_result_row(
                case=case,
                model_version=model_version,
                top_hit_citations=top_hit_citations,
                retrieval_hit=retrieval_hit,
                evaluator=args.evaluator,
                expected_citations_all=expected_all,
                expected_citations_scoped=expected_scoped,
                expected_citations_excluded=expected_excluded,
                case_scope=case_scope,
                generated_answer=generated_answer,
                generated_legal_basis=generated_legal_basis,
                insufficient_context=insufficient_context,
            )
            if args.model:
                row["citation_correct"] = citation_correct
                row["abstention_correct"] = abstention_correct
                row["hallucination_flag"] = hallucination_flag
                row["comments"] = f"Retrieved: {' | '.join(top_hit_citations)}"

            rows.append(row)
            print(f"[{index}/{len(cases)}] {case.id}: retrieval_hit_at_{args.top_k}={row['retrieval_hit_at_5']}")

    finally:
        retriever.close()

    write_results_jsonl(rows, output_jsonl)
    write_results_csv(rows, output_csv)
    print(f"Saved benchmark JSONL: {output_jsonl.resolve()}")
    print(f"Saved benchmark CSV: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
