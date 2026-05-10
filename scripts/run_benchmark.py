from __future__ import annotations

import argparse
import json
from datetime import datetime
import os
from pathlib import Path
import re
import sys

from vn_labor_law_ai_assistant.answering import (
    EvidenceQuote,
    build_messages,
    parse_answer_payload,
)
from vn_labor_law_ai_assistant.evaluation import (
    BENCHMARK_JSONL_NAME,
    BenchmarkCase,
    JUDGE_JSON_SCHEMA,
    build_judge_messages,
    compute_final_score_10,
    document_families_from_chunk_paths,
    expected_citations,
    expected_citation_scope,
    expected_citations_in_scope,
    expected_citations_out_of_scope,
    first_relevant_rank,
    load_benchmark_jsonl,
    mean_reciprocal_rank,
    parse_judge_payload,
    reciprocal_rank,
    result_columns,
    retrieval_hit_at_k,
    score_citation_article_correctness_for_scope,
    score_citation_document_correctness_for_scope,
    write_results_csv,
    write_results_jsonl,
)
from vn_labor_law_ai_assistant.llm import (
    DEFAULT_PROVIDER,
    SUPPORTED_PROVIDERS,
    chat_completion,
    default_benchmark_judge_model,
    default_benchmark_judge_provider,
    default_model_for_provider,
    normalize_provider,
    provider_model_label,
    resolve_model_name,
)
from vn_labor_law_ai_assistant.retriever import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_RERANKER_TOP_N,
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
        "--max-context-tokens",
        type=int,
        default=DEFAULT_MAX_CONTEXT_TOKENS,
        help=(
            "Soft token budget for prompt context. Lower-ranked context blocks are dropped "
            f"before truncating text (default: {DEFAULT_MAX_CONTEXT_TOKENS})."
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
        "--judge-provider",
        default="",
        help=(
            "Optional judge provider for answer scoring. Leave empty to use the benchmark "
            f"default ({default_benchmark_judge_provider()})."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help=(
            "Optional judge model name for LLM-as-a-judge scoring. Leave empty to use the "
            "default model for the chosen judge provider."
        ),
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable LLM-as-a-judge scoring even when --model is provided.",
    )
    parser.add_argument(
        "--evaluator",
        default="auto-benchmark",
        help="Evaluator label stored in output rows.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "Path to a previous benchmark .jsonl or .csv. Completed case IDs are skipped, "
            "and progress is appended by rewriting the same output stem."
        ),
    )
    return parser.parse_args()


def build_result_row(
    *,
    case: BenchmarkCase,
    model_version: str,
    retrieval_hit_column: str,
    top_hit_citations: list[str],
    retrieval_hit: bool | None,
    retrieval_first_relevant_rank: int | None,
    retrieval_reciprocal_rank: float | None,
    evaluator: str,
    expected_citations_all: tuple[str, ...],
    expected_citations_scoped: tuple[str, ...],
    expected_citations_excluded: tuple[str, ...],
    case_scope: str,
    generated_answer: str = "",
    generated_legal_basis: tuple[str, ...] = (),
    generated_evidence_quotes: tuple[EvidenceQuote, ...] = (),
    insufficient_context: str = "",
) -> dict[str, object]:
    if retrieval_hit is None:
        retrieval_hit_value = "N/A"
    else:
        retrieval_hit_value = "Yes" if retrieval_hit else "No"

    return {
        "id": case.id,
        "model_version": model_version,
        retrieval_hit_column: retrieval_hit_value,
        "retrieval_first_relevant_rank": (
            "" if retrieval_first_relevant_rank is None else retrieval_first_relevant_rank
        ),
        "retrieval_reciprocal_rank": (
            "" if retrieval_reciprocal_rank is None else round(retrieval_reciprocal_rank, 6)
        ),
        "citation_correct": "",
        "citation_document_correct": "",
        "citation_provision_correct": "",
        "citation_article_correct": "",
        "citation_supports_answer": "",
        "answer_correct": "",
        "legal_issue_classification_correct": "",
        "legal_reasoning_score_1_5": "",
        "missing_information_score_0_2": "",
        "hallucination_flag": "",
        "hallucination_types": "",
        "abstention_correct": "",
        "groundedness_score_1_5": "",
        "clarity_score_1_5": "",
        "format_score_1_5": "",
        "final_score_10": "",
        "evaluator": evaluator,
        "comments": "",
        "skill_tag": case.skill_tag or "",
        "question": case.question,
        "expected_citations": " | ".join(expected_citations_all),
        "expected_citations_in_scope": " | ".join(expected_citations_scoped),
        "expected_citations_out_of_scope": " | ".join(expected_citations_excluded),
        "case_scope": case_scope,
        "retrieved_citations": " | ".join(top_hit_citations),
        "generated_answer": generated_answer,
        "generated_legal_basis": " | ".join(generated_legal_basis),
        "generated_evidence_quotes": format_evidence_quotes(generated_evidence_quotes),
        "insufficient_context": insufficient_context,
    }


def format_evidence_quotes(evidence_quotes: tuple[EvidenceQuote, ...]) -> str:
    return " | ".join(
        f"{evidence_quote.citation}: {evidence_quote.quote}"
        for evidence_quote in evidence_quotes
    )


def slugify_model_version(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())
    return cleaned.strip("-") or "run"


def load_existing_result_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Resume file does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".csv":
        import csv

        with path.open(encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError("--resume-from must point to a .jsonl or .csv file.")


def resolve_judge_configuration(args: argparse.Namespace) -> tuple[str, str, bool]:
    if args.no_judge or not str(args.model or "").strip():
        return "", "", False

    provider_input = str(args.judge_provider or "").strip()
    if provider_input:
        provider_name = normalize_provider(provider_input)
        if str(args.judge_model or "").strip():
            model_name = resolve_model_name(provider_name, args.judge_model)
        else:
            model_name = default_model_for_provider(provider_name)
        return provider_name, model_name, True

    provider_name = default_benchmark_judge_provider()
    model_name = default_benchmark_judge_model(provider_name)
    if str(args.judge_model or "").strip():
        model_name = str(args.judge_model).strip()
    return provider_name, model_name, True


def resolve_evaluator_label(
    requested_label: str,
    *,
    judge_enabled: bool,
    judge_provider: str,
    judge_model: str,
) -> str:
    if requested_label != "auto-benchmark" or not judge_enabled:
        return requested_label
    return f"llm-judge:{provider_model_label(judge_provider, judge_model)}"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    if args.judge_provider:
        normalize_provider(args.judge_provider)
    if args.no_judge and (args.judge_provider or args.judge_model):
        raise ValueError("--no-judge cannot be combined with --judge-provider or --judge-model.")
    if (args.judge_provider or args.judge_model) and not args.model:
        raise ValueError("Judge options require --model because there is no generated answer otherwise.")

    cases = load_benchmark_jsonl(args.benchmark_path)
    if args.limit > 0:
        cases = cases[: args.limit]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    retriever = HybridRetriever(
        index_path=args.index_path,
        reranker_model=args.reranker_model,
        reranker_top_n=args.reranker_top_n,
    )
    model_version = provider_model_label(args.provider, args.model) if args.model else "retrieval-only"
    if args.resume_from:
        resume_path = args.resume_from
        output_stem = resume_path.stem
        output_jsonl = resume_path.with_suffix(".jsonl")
        output_csv = resume_path.with_suffix(".csv")
        rows = load_existing_result_rows(resume_path)
        completed_ids = {str(row.get("id") or "") for row in rows if row.get("id")}
        cases = [case for case in cases if case.id not in completed_ids]
        print(f"Resuming from {resume_path}: {len(completed_ids)} completed, {len(cases)} remaining")
    else:
        rows: list[dict[str, object]] = []
        output_stem = f"benchmark_{slugify_model_version(model_version)}_{timestamp}"
        output_jsonl = args.output_dir / f"{output_stem}.jsonl"
        output_csv = args.output_dir / f"{output_stem}.csv"
    del output_stem
    reciprocal_ranks: list[float | None] = []
    retrieval_hit_column = f"retrieval_hit_at_{args.top_k}"
    fieldnames = result_columns(retrieval_hit_column)
    allowed_document_families = document_families_from_chunk_paths(
        retriever.manifest.get("chunk_paths", [])
    )
    judge_provider, judge_model, judge_enabled = resolve_judge_configuration(args)
    evaluator_label = resolve_evaluator_label(
        args.evaluator,
        judge_enabled=judge_enabled,
        judge_provider=judge_provider,
        judge_model=judge_model,
    )

    def save_progress() -> None:
        if not rows:
            return
        write_results_jsonl(rows, output_jsonl)
        write_results_csv(rows, output_csv, fieldnames=fieldnames)

    completed = False
    try:
        if retriever.reranker_enabled:
            print(f"Semantic reranker: {retriever.reranker_model_name} (top {args.reranker_top_n})")
        else:
            print("Semantic reranker: disabled")
        if args.model:
            print(f"Answer model: {provider_model_label(args.provider, args.model)}")
            if judge_enabled:
                print(f"Judge model: {provider_model_label(judge_provider, judge_model)}")
            else:
                print("Judge model: disabled")
        remaining_count = len(cases)
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
            relevant_rank = first_relevant_rank(
                case,
                top_hit_citations,
                k=args.top_k,
                allowed_document_families=allowed_document_families,
            )
            reciprocal = reciprocal_rank(
                case,
                top_hit_citations,
                k=args.top_k,
                allowed_document_families=allowed_document_families,
            )
            reciprocal_ranks.append(reciprocal)

            generated_answer = ""
            generated_legal_basis: tuple[str, ...] = ()
            generated_evidence_quotes: tuple[EvidenceQuote, ...] = ()
            insufficient_context = ""
            citation_correct = ""
            citation_document_correct = ""
            citation_provision_correct = ""
            citation_article_correct = ""
            citation_supports_answer = ""
            abstention_correct = ""
            hallucination_flag = ""
            hallucination_types = ""
            answer_correct = ""
            legal_issue_classification_correct = ""
            legal_reasoning_score = ""
            missing_information_score = ""
            groundedness_score = ""
            clarity_score = ""
            format_score = ""
            final_score = ""
            judge_comment = ""

            if args.model:
                contexts = select_contexts_for_prompt(
                    retrieval_result.contexts,
                    max_contexts=args.max_contexts,
                    max_chars=args.max_context_chars,
                    max_tokens=args.max_context_tokens,
                )
                response = chat_completion(
                    provider=args.provider,
                    model=args.model,
                    messages=build_messages(
                        case.question,
                        contexts,
                        max_context_chars=args.max_context_chars,
                        max_context_tokens=args.max_context_tokens,
                    ),
                    temperature=0,
                    json_schema_name="legal_answer",
                )
                parsed = parse_answer_payload(
                    response.content,
                    contexts,
                    question=case.question,
                )
                generated_answer = parsed.answer
                generated_legal_basis = parsed.legal_basis
                generated_evidence_quotes = parsed.evidence_quotes
                insufficient_context = "Yes" if parsed.insufficient_context else "No"
                citation_article_correct = score_citation_article_correctness_for_scope(
                    case,
                    parsed.legal_basis,
                    allowed_document_families=allowed_document_families,
                )
                citation_provision_correct = citation_article_correct
                citation_document_correct = score_citation_document_correctness_for_scope(
                    case,
                    parsed.legal_basis,
                    allowed_document_families=allowed_document_families,
                )
                citation_correct = citation_article_correct
                if case.abstain_required:
                    abstention_correct = "yes" if parsed.insufficient_context else "no"
                else:
                    abstention_correct = "no" if parsed.insufficient_context else "n/a"

                if judge_enabled:
                    judge_response = chat_completion(
                        provider=judge_provider,
                        model=judge_model,
                        messages=build_judge_messages(
                            case,
                            generated_answer=generated_answer,
                            generated_legal_basis=generated_legal_basis,
                            insufficient_context=insufficient_context,
                            expected_citations_scoped=expected_scoped,
                            retrieved_citations=top_hit_citations,
                            case_scope=case_scope,
                        ),
                        temperature=0,
                        json_schema=JUDGE_JSON_SCHEMA,
                        json_schema_name="benchmark_judge",
                    )
                    judged = parse_judge_payload(judge_response.content)
                    if judged is not None:
                        answer_correct = judged.answer_correct
                        legal_issue_classification_correct = (
                            judged.legal_issue_classification_correct
                        )
                        legal_reasoning_score = judged.legal_reasoning_score_1_5
                        missing_information_score = judged.missing_information_score_0_2
                        citation_supports_answer = judged.citation_supports_answer
                        groundedness_score = judged.groundedness_score_1_5
                        clarity_score = judged.clarity_score_1_5
                        format_score = judged.format_score_1_5
                        hallucination_types = " | ".join(judged.hallucination_types) or "none"
                        final_score = compute_final_score_10(
                            case=case,
                            answer_correct=answer_correct,
                            legal_issue_classification_correct=legal_issue_classification_correct,
                            legal_reasoning_score_1_5=legal_reasoning_score,
                            missing_information_score_0_2=missing_information_score,
                            citation_article_correct=citation_provision_correct,
                            citation_supports_answer=citation_supports_answer,
                            groundedness_score_1_5=groundedness_score,
                            clarity_score_1_5=clarity_score,
                            format_score_1_5=format_score,
                        )
                        judge_comment = judged.comments
                    else:
                        judge_comment = "Judge model did not return valid JSON."

                if citation_correct == "na":
                    hallucination_flag = "na"
                else:
                    likely_hallucinated = not parsed.insufficient_context and citation_correct == "no"
                    if (
                        judge_enabled
                        and groundedness_score != ""
                        and not parsed.insufficient_context
                        and int(groundedness_score) <= 2
                    ):
                        likely_hallucinated = True
                    if (
                        citation_provision_correct == "no"
                        and citation_supports_answer == "no"
                    ):
                        likely_hallucinated = True
                    if (
                        answer_correct == "no"
                        and legal_reasoning_score != ""
                        and int(legal_reasoning_score) <= 2
                    ):
                        likely_hallucinated = True
                    if hallucination_types not in {"", "none"}:
                        likely_hallucinated = True
                    hallucination_flag = "yes" if likely_hallucinated else "no"

            row = build_result_row(
                case=case,
                model_version=model_version,
                retrieval_hit_column=retrieval_hit_column,
                top_hit_citations=top_hit_citations,
                retrieval_hit=retrieval_hit,
                retrieval_first_relevant_rank=relevant_rank,
                retrieval_reciprocal_rank=reciprocal,
                evaluator=evaluator_label,
                expected_citations_all=expected_all,
                expected_citations_scoped=expected_scoped,
                expected_citations_excluded=expected_excluded,
                case_scope=case_scope,
                generated_answer=generated_answer,
                generated_legal_basis=generated_legal_basis,
                generated_evidence_quotes=generated_evidence_quotes,
                insufficient_context=insufficient_context,
            )
            if args.model:
                row["citation_correct"] = citation_correct
                row["citation_document_correct"] = citation_document_correct
                row["citation_provision_correct"] = citation_provision_correct
                row["citation_article_correct"] = citation_article_correct
                row["citation_supports_answer"] = citation_supports_answer
                row["answer_correct"] = answer_correct
                row["legal_issue_classification_correct"] = legal_issue_classification_correct
                row["legal_reasoning_score_1_5"] = legal_reasoning_score
                row["missing_information_score_0_2"] = missing_information_score
                row["abstention_correct"] = abstention_correct
                row["hallucination_flag"] = hallucination_flag
                row["hallucination_types"] = hallucination_types
                row["groundedness_score_1_5"] = groundedness_score
                row["clarity_score_1_5"] = clarity_score
                row["format_score_1_5"] = format_score
                row["final_score_10"] = final_score
                comments = [f"Retrieved: {' | '.join(top_hit_citations)}"]
                if judge_comment:
                    comments.append(f"Judge: {judge_comment}")
                row["comments"] = " | ".join(comment for comment in comments if comment)

            rows.append(row)
            save_progress()
            print(
                f"[{index}/{remaining_count}] {case.id}: "
                f"{retrieval_hit_column}={row[retrieval_hit_column]}"
            )
        completed = True

    finally:
        retriever.close()
        save_progress()
        if rows and not completed:
            print(f"Saved partial benchmark JSONL: {output_jsonl.resolve()}")
            print(f"Saved partial benchmark CSV: {output_csv.resolve()}")

    scored_hit_cases = [
        row for row in rows if row[retrieval_hit_column] in {"Yes", "No"}
    ]
    hit_rate = (
        sum(1 for row in scored_hit_cases if row[retrieval_hit_column] == "Yes")
        / len(scored_hit_cases)
        if scored_hit_cases
        else 0.0
    )
    mrr = mean_reciprocal_rank(reciprocal_ranks)
    print(f"Retrieval hit@{args.top_k}: {hit_rate:.2%} over {len(scored_hit_cases)} scored cases")
    if mrr is not None:
        print(f"Retrieval MRR@{args.top_k}: {mrr:.4f}")
    print(f"Saved benchmark JSONL: {output_jsonl.resolve()}")
    print(f"Saved benchmark CSV: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
