from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scripts.evaluate_retrieval_modes import (
    DEFAULT_BENCHMARK_PATH,
    BenchmarkItem,
    CitationSpec,
    EnvOverride,
    RetrievedItem,
    citation_order_violations,
    forbidden_violations,
    load_or_create_benchmark,
    markdown_table,
    rank_of_spec,
    retrieved_from_context,
)
from vn_labor_law_ai_assistant.rag.answering import (
    generate_grounded_answer,
    validate_answer_quality,
)
from vn_labor_law_ai_assistant.rag.retrieval import HybridRetriever, RetrievalContext


DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "evaluation"
QUALITY_CHECK_FIELDS = (
    "direct_answer_present",
    "required_legal_rule_present",
    "numeric_answer_present",
    "yes_no_answer_present",
    "conditions_listed",
    "exception_answer_present",
    "no_article_title_only_answer",
    "all_legal_claims_have_citations",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate graph-augmented retrieval plus grounded answer generation end to end.")
    parser.add_argument("--index-path", type=Path, default=REPO_ROOT / "artifacts" / "index")
    parser.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", type=str, default="end_to_end_rag")
    parser.add_argument("--comparison-summary-path", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--prefetch-limit", type=int, default=24)
    parser.add_argument("--max-answer-contexts", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--provider", type=str, default=os.getenv("ANSWER_GENERATION_PROVIDER", "extractive"))
    parser.add_argument("--model", type=str, default=os.getenv("ANSWER_GENERATION_MODEL", os.getenv("GROQ_MODEL", "")))
    parser.add_argument("--reranker-model", type=str, default=os.getenv("RERANKER_MODEL", ""))
    parser.add_argument("--embedding-provider", type=str, default=os.getenv("GRAPH_RETRIEVAL_TEST_EMBEDDING_PROVIDER", "sentence_transformers"))
    parser.add_argument("--device", type=str, default=os.getenv("GRAPH_RETRIEVAL_TEST_DEVICE", "cpu"))
    return parser.parse_args()


def status_passed(value: object) -> bool:
    return value is True or value == "not_applicable"


def status_failed(value: object) -> bool:
    return value is False


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def bool_rate(values: Iterable[bool]) -> float:
    return mean(1.0 if value else 0.0 for value in values)


def spec_rank(
    contexts: Sequence[RetrievedItem],
    citation: CitationSpec,
    *,
    top_k: int,
) -> int | None:
    citation_limit = min(top_k, citation.top_n) if citation.top_n else top_k
    return rank_of_spec(contexts, citation, limit=citation_limit)


def found_required_citations(
    contexts: Sequence[RetrievedItem],
    required_citations: Sequence[CitationSpec],
    *,
    top_k: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    found: list[str] = []
    missing: list[str] = []
    ranks: dict[str, int] = {}
    for citation in required_citations:
        rank = spec_rank(contexts, citation, top_k=top_k)
        if rank is None:
            missing.append(citation.label)
        else:
            found.append(citation.label)
            ranks[citation.label] = rank
    return found, missing, ranks


def retrieval_source_distribution(contexts: Sequence[RetrievalContext], *, top_k: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for context in contexts[:top_k]:
        source = str(context.payload.get("retrieval_source") or "unknown")
        counter[source] += 1
    return dict(counter)


def context_graph_depth(context: RetrievalContext) -> int | None:
    value = context.payload.get("graph_depth")
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def context_graph_edge_types(context: RetrievalContext) -> tuple[str, ...]:
    values = context.payload.get("graph_edge_types") or ()
    if isinstance(values, str):
        return (values,)
    return tuple(str(value) for value in values)


def is_graph_expanded_context(context: RetrievalContext) -> bool:
    source = str(context.payload.get("retrieval_source") or "")
    method = str(context.payload.get("retrieval_method") or "")
    graph_score = context.payload.get("graph_score")
    try:
        graph_score_value = float(graph_score or 0.0)
    except (TypeError, ValueError):
        graph_score_value = 0.0
    return (
        source in {"graph", "hybrid"}
        or method.startswith("neo4j")
        or method.startswith("graph")
        or context_graph_depth(context) is not None
        or graph_score_value > 0
    )


def top_context_to_json(context: RetrievalContext) -> dict[str, object]:
    payload = dict(context.payload)
    return {
        "chunk_id": context.chunk_id,
        "citation_text": context.citation_text,
        "score": context.score,
        "document_id": payload.get("document_id"),
        "document_type": payload.get("document_type"),
        "normative_rank": payload.get("normative_rank"),
        "article_number": payload.get("article_number"),
        "clause_ref": payload.get("clause_ref"),
        "point_refs": payload.get("point_refs") or [],
        "retrieval_source": payload.get("retrieval_source"),
        "retrieval_method": payload.get("retrieval_method"),
        "graph_depth": payload.get("graph_depth"),
        "graph_edge_types": payload.get("graph_edge_types") or [],
        "vector_score": payload.get("vector_score"),
        "graph_score": payload.get("graph_score"),
        "final_score": payload.get("final_score"),
    }


def retrieval_metrics_for_query(
    item: BenchmarkItem,
    contexts: Sequence[RetrievalContext],
    *,
    top_k: int,
) -> dict[str, object]:
    retrieved = tuple(retrieved_from_context(context) for context in contexts[:top_k])
    found, missing, ranks = found_required_citations(retrieved, item.required_citations, top_k=top_k)
    forbidden = forbidden_violations(retrieved, item.forbidden_citations, default_limit=5)
    order_violations = citation_order_violations(retrieved, item.citation_order_rules)
    graph_contexts = [context for context in contexts[:top_k] if is_graph_expanded_context(context)]
    graph_depths = [depth for depth in (context_graph_depth(context) for context in graph_contexts) if depth is not None]
    edge_counter: Counter[str] = Counter()
    for context in graph_contexts:
        edge_counter.update(context_graph_edge_types(context))

    retrieval_passed = not missing and not forbidden and not order_violations
    return {
        "required_citations_found": found,
        "missing_required_citations": missing,
        "required_citation_ranks": ranks,
        "required_citation_coverage": (len(found) / len(item.required_citations)) if item.required_citations else 1.0,
        "forbidden_citation_violations": [*forbidden, *order_violations],
        "citation_order_violations": order_violations,
        "retrieval_source_distribution": retrieval_source_distribution(contexts, top_k=top_k),
        "graph_expansion_used": bool(graph_contexts),
        "graph_expanded_chunks": len(graph_contexts),
        "graph_depth": max(graph_depths) if graph_depths else 0,
        "average_graph_depth": mean(float(value) for value in graph_depths),
        "graph_edge_types": dict(edge_counter),
        "top_k_contexts": [top_context_to_json(context) for context in contexts[:top_k]],
        "retrieval_passed": retrieval_passed,
    }


def item_metadata(item: BenchmarkItem) -> dict[str, object]:
    return {
        "topic": item.topic,
        "difficulty": item.difficulty,
        "requires_graph": item.requires_graph,
        "requires_normative_hierarchy": item.requires_normative_hierarchy,
        "expected_answer_points": list(item.expected_answer_points),
    }


def answer_completeness_score(quality_validation: dict[str, object]) -> float:
    applicable_checks = [
        quality_validation[field]
        for field in QUALITY_CHECK_FIELDS
        if quality_validation.get(field) != "not_applicable"
    ]
    if not applicable_checks:
        return 0.0
    return mean(1.0 if status_passed(value) else 0.0 for value in applicable_checks)


def answer_metrics_for_query(answer_result: Any, quality_validation: Any) -> dict[str, object]:
    validation = asdict(answer_result.validation)
    quality = asdict(quality_validation)
    citation_grounding_passed = (
        bool(validation["passed"])
        and bool(validation["citations_allowed"])
        and not validation["unsupported_article_numbers"]
        and not validation["unretrieved_citations"]
    )
    insufficient_context_handled = (
        bool(validation["has_uncertainty_when_insufficient"])
        if answer_result.parsed.insufficient_context
        else True
    )
    legal_claim_citation_coverage = quality["all_legal_claims_have_citations"]
    answer_faithfulness_passed = (
        citation_grounding_passed
        and status_passed(legal_claim_citation_coverage)
        and not validation["ignores_higher_rank_context"]
        and int(quality["low_information_quotes_count"]) == 0
    )
    completeness_score = answer_completeness_score(quality)
    answer_passed = bool(validation["passed"]) and bool(quality["passed"])
    return {
        "direct_answer_present": quality["direct_answer_present"],
        "required_legal_rule_present": quality["required_legal_rule_present"],
        "numeric_answer_present": quality["numeric_answer_present"],
        "yes_no_answer_present": quality["yes_no_answer_present"],
        "conditions_listed": quality["conditions_listed"],
        "exception_answer_present": quality["exception_answer_present"],
        "citation_validation_passed": bool(validation["passed"]),
        "quality_validation_passed": bool(quality["passed"]),
        "unsupported_article_numbers": list(validation["unsupported_article_numbers"]),
        "unretrieved_citations": list(validation["unretrieved_citations"]),
        "low_information_quotes_count": int(quality["low_information_quotes_count"]),
        "legal_claim_citation_coverage": legal_claim_citation_coverage,
        "all_legal_claims_have_citations": quality["all_legal_claims_have_citations"],
        "no_article_title_only_answer": quality["no_article_title_only_answer"],
        "insufficient_context_handled": insufficient_context_handled,
        "citation_grounding_passed": citation_grounding_passed,
        "answer_faithfulness_passed": answer_faithfulness_passed,
        "answer_completeness_score": completeness_score,
        "answer_passed": answer_passed,
        "insufficient_context": bool(answer_result.parsed.insufficient_context),
        "ignores_higher_rank_context": bool(validation["ignores_higher_rank_context"]),
        "quality_warnings": list(quality["warnings"]),
        "citation_warnings": list(validation["warnings"]),
        "applied_answer_intent": quality["applied_answer_intent"],
    }


def final_quality_score(
    *,
    retrieval_metrics: dict[str, object],
    answer_metrics: dict[str, object],
) -> float:
    retrieval_score = (
        0.75 * float(retrieval_metrics["required_citation_coverage"])
        + 0.25 * (1.0 if not retrieval_metrics["forbidden_citation_violations"] else 0.0)
    )
    answer_score = mean(
        (
            1.0 if answer_metrics["citation_grounding_passed"] else 0.0,
            1.0 if answer_metrics["answer_faithfulness_passed"] else 0.0,
            float(answer_metrics["answer_completeness_score"]),
            1.0 if answer_metrics["insufficient_context_handled"] else 0.0,
        )
    )
    return round(100.0 * (0.45 * retrieval_score + 0.55 * answer_score), 2)


def classify_failure_reasons(
    *,
    retrieval_metrics: dict[str, object],
    answer_metrics: dict[str, object],
) -> list[str]:
    reasons: list[str] = []
    if retrieval_metrics["missing_required_citations"]:
        reasons.append("retrieval_missing_required_context")
    if retrieval_metrics["forbidden_citation_violations"]:
        reasons.append("retrieval_over_expansion")
    if status_failed(answer_metrics["direct_answer_present"]):
        reasons.append("answer_not_direct")
    if status_failed(answer_metrics["required_legal_rule_present"]):
        reasons.append("answer_missing_required_rule")
    if any(
        status_failed(answer_metrics.get(field))
        for field in (
            "numeric_answer_present",
            "yes_no_answer_present",
            "conditions_listed",
            "exception_answer_present",
        )
    ) and "answer_missing_required_rule" not in reasons:
        reasons.append("answer_missing_required_rule")
    if not answer_metrics["citation_grounding_passed"]:
        reasons.append("hallucinated_citation")
    if answer_metrics["unsupported_article_numbers"]:
        reasons.append("unsupported_article_number")
    if int(answer_metrics["low_information_quotes_count"]) > 0:
        reasons.append("low_information_answer")
    if status_failed(answer_metrics.get("no_article_title_only_answer")) and "low_information_answer" not in reasons:
        reasons.append("low_information_answer")
    if status_failed(answer_metrics.get("all_legal_claims_have_citations")) and "answer_missing_required_rule" not in reasons:
        reasons.append("answer_missing_required_rule")
    if answer_metrics["ignores_higher_rank_context"]:
        reasons.append("wrong_normative_priority")
    if not answer_metrics["insufficient_context_handled"]:
        reasons.append("insufficient_context_not_reported")
    return reasons


def end_to_end_metrics_for_query(
    *,
    retrieval_metrics: dict[str, object],
    answer_metrics: dict[str, object],
) -> dict[str, object]:
    retrieval_passed = bool(retrieval_metrics["retrieval_passed"])
    answer_passed = bool(answer_metrics["answer_passed"])
    citation_grounding_passed = bool(answer_metrics["citation_grounding_passed"])
    answer_faithfulness_passed = bool(answer_metrics["answer_faithfulness_passed"])
    score = final_quality_score(
        retrieval_metrics=retrieval_metrics,
        answer_metrics=answer_metrics,
    )
    failure_reasons = classify_failure_reasons(
        retrieval_metrics=retrieval_metrics,
        answer_metrics=answer_metrics,
    )
    end_to_end_passed = (
        retrieval_passed
        and answer_passed
        and citation_grounding_passed
        and answer_faithfulness_passed
        and not failure_reasons
    )
    return {
        "end_to_end_passed": end_to_end_passed,
        "retrieval_passed": retrieval_passed,
        "answer_passed": answer_passed,
        "citation_grounding_passed": citation_grounding_passed,
        "answer_faithfulness_passed": answer_faithfulness_passed,
        "answer_completeness_score": answer_metrics["answer_completeness_score"],
        "final_quality_score": score,
        "failure_reasons": failure_reasons,
    }


def run_end_to_end(args: argparse.Namespace) -> dict[str, object]:
    if args.embedding_provider:
        os.environ["EMBEDDING_PROVIDER"] = args.embedding_provider

    items = load_or_create_benchmark(args.benchmark_path, limit=args.limit)
    env = {
        "LEGAL_GRAPH_ENABLED": "true",
        "LEGAL_GRAPH_BACKEND": "neo4j",
        "LEGAL_GRAPH_COMPLEX_QUERY_ONLY": "true",
        "LEGAL_GRAPH_MAX_EXPANDED_CHUNKS": "16",
        "LEGAL_GRAPH_EXPANSION_DEPTH": "2",
        "EMBEDDING_PROVIDER": args.embedding_provider,
    }
    results: list[dict[str, object]] = []

    with EnvOverride(env):
        retriever = HybridRetriever(
            index_path=args.index_path,
            device=args.device,
            reranker_model=args.reranker_model,
            query_router_enabled=False,
        )
        try:
            for item in items:
                retrieval_result = retriever.retrieve(
                    item.query,
                    top_k=args.top_k,
                    prefetch_limit=args.prefetch_limit,
                )
                contexts = retrieval_result.contexts[: args.top_k]
                retrieval_metrics = retrieval_metrics_for_query(
                    item,
                    contexts,
                    top_k=args.top_k,
                )
                answer_result = generate_grounded_answer(
                    item.query,
                    contexts,
                    provider=args.provider,
                    model=args.model,
                    max_answer_contexts=args.max_answer_contexts,
                )
                quality_validation = validate_answer_quality(
                    item.query,
                    answer_result.parsed,
                    answer_result.contexts,
                    final_answer=answer_result.answer,
                )
                answer_metrics = answer_metrics_for_query(answer_result, quality_validation)
                end_to_end_metrics = end_to_end_metrics_for_query(
                    retrieval_metrics=retrieval_metrics,
                    answer_metrics=answer_metrics,
                )
                results.append(
                    {
                        "id": item.id,
                        "query": item.query,
                        "category": item.category,
                        **item_metadata(item),
                        "expected_documents": list(item.expected_documents),
                        "required_citations": [asdict(citation) for citation in item.required_citations],
                        "forbidden_citations": [asdict(citation) for citation in item.forbidden_citations],
                        "retrieval_metrics": retrieval_metrics,
                        "answer_metrics": answer_metrics,
                        "end_to_end_metrics": end_to_end_metrics,
                        "answer": answer_result.answer,
                        "legal_basis": list(answer_result.parsed.legal_basis),
                        "evidence_quotes": [asdict(quote) for quote in answer_result.parsed.evidence_quotes],
                        "provider": answer_result.provider,
                        "model": answer_result.model,
                        "generation_method": answer_result.generation_method,
                    }
                )
        finally:
            retriever.close()

    return build_output(
        args=args,
        results=results,
        benchmark_count=len(items),
    )


def aggregate_by_category(results: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for result in results:
        grouped[str(result["category"])].append(result)

    output: dict[str, dict[str, object]] = {}
    for category, values in sorted(grouped.items()):
        output[category] = {
            "query_count": len(values),
            "retrieval_pass_rate": bool_rate(bool(value["end_to_end_metrics"]["retrieval_passed"]) for value in values),
            "answer_pass_rate": bool_rate(bool(value["end_to_end_metrics"]["answer_passed"]) for value in values),
            "citation_pass_rate": bool_rate(bool(value["end_to_end_metrics"]["citation_grounding_passed"]) for value in values),
            "end_to_end_pass_rate": bool_rate(bool(value["end_to_end_metrics"]["end_to_end_passed"]) for value in values),
            "average_quality_score": mean(float(value["end_to_end_metrics"]["final_quality_score"]) for value in values),
        }
    return output


def aggregate_by_field(results: Sequence[dict[str, object]], field: str) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for result in results:
        value = result.get(field)
        if value in (None, ""):
            value = "unspecified"
        grouped[str(value)].append(result)

    output: dict[str, dict[str, object]] = {}
    for value, records in sorted(grouped.items()):
        output[value] = {
            "query_count": len(records),
            "retrieval_pass_rate": bool_rate(bool(record["end_to_end_metrics"]["retrieval_passed"]) for record in records),
            "answer_pass_rate": bool_rate(bool(record["end_to_end_metrics"]["answer_passed"]) for record in records),
            "citation_pass_rate": bool_rate(bool(record["end_to_end_metrics"]["citation_grounding_passed"]) for record in records),
            "quality_pass_rate": bool_rate(bool(record["answer_metrics"]["quality_validation_passed"]) for record in records),
            "end_to_end_pass_rate": bool_rate(bool(record["end_to_end_metrics"]["end_to_end_passed"]) for record in records),
            "average_quality_score": mean(float(record["end_to_end_metrics"]["final_quality_score"]) for record in records),
        }
    return output


def failure_analysis(results: Sequence[dict[str, object]]) -> dict[str, object]:
    counter: Counter[str] = Counter()
    failed_queries: list[dict[str, object]] = []
    for result in results:
        reasons = list(result["end_to_end_metrics"]["failure_reasons"])
        if not result["end_to_end_metrics"]["end_to_end_passed"] and not reasons:
            reasons = ["unclassified_failure"]
        counter.update(reasons)
        if reasons:
            failed_queries.append(
                {
                    "id": result["id"],
                    "query": result["query"],
                    "category": result["category"],
                    "failure_reasons": reasons,
                    "missing_required_citations": result["retrieval_metrics"]["missing_required_citations"],
                    "forbidden_citation_violations": result["retrieval_metrics"]["forbidden_citation_violations"],
                    "unsupported_article_numbers": result["answer_metrics"]["unsupported_article_numbers"],
                    "unretrieved_citations": result["answer_metrics"]["unretrieved_citations"],
                }
            )
    return {
        "failure_reason_counts": dict(counter),
        "failed_queries": failed_queries,
    }


def build_output(
    *,
    args: argparse.Namespace,
    results: Sequence[dict[str, object]],
    benchmark_count: int,
) -> dict[str, object]:
    end_to_end_pass_rate = bool_rate(bool(result["end_to_end_metrics"]["end_to_end_passed"]) for result in results)
    retrieval_pass_rate = bool_rate(bool(result["end_to_end_metrics"]["retrieval_passed"]) for result in results)
    answer_pass_rate = bool_rate(bool(result["end_to_end_metrics"]["answer_passed"]) for result in results)
    citation_pass_rate = bool_rate(bool(result["end_to_end_metrics"]["citation_grounding_passed"]) for result in results)
    quality_pass_rate = bool_rate(bool(result["answer_metrics"]["quality_validation_passed"]) for result in results)
    low_information_quotes = sum(int(result["answer_metrics"]["low_information_quotes_count"]) for result in results)
    unsupported_article_numbers = [
        item
        for result in results
        for item in result["answer_metrics"]["unsupported_article_numbers"]
    ]
    unretrieved_citations = [
        item
        for result in results
        for item in result["answer_metrics"]["unretrieved_citations"]
    ]
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_path": str(args.benchmark_path),
        "benchmark_count": benchmark_count,
        "top_k": args.top_k,
        "provider": args.provider,
        "model": args.model or "deterministic",
        "passed": all(bool(result["end_to_end_metrics"]["end_to_end_passed"]) for result in results),
        "overall": {
            "query_count": len(results),
            "end_to_end_pass_rate": end_to_end_pass_rate,
            "retrieval_pass_rate": retrieval_pass_rate,
            "answer_pass_rate": answer_pass_rate,
            "citation_validation_pass_rate": citation_pass_rate,
            "quality_validation_pass_rate": quality_pass_rate,
            "average_final_quality_score": mean(float(result["end_to_end_metrics"]["final_quality_score"]) for result in results),
            "low_information_quotes_count": low_information_quotes,
            "unsupported_article_numbers": unsupported_article_numbers,
            "unretrieved_citations": unretrieved_citations,
            "graph_expansion_used_count": sum(1 for result in results if result["retrieval_metrics"]["graph_expansion_used"]),
            "average_graph_depth": mean(float(result["retrieval_metrics"]["average_graph_depth"]) for result in results),
        },
        "category_metrics": aggregate_by_category(results),
        "topic_metrics": aggregate_by_field(results, "topic"),
        "difficulty_metrics": aggregate_by_field(results, "difficulty"),
        "graph_required_metrics": aggregate_by_field(results, "requires_graph"),
        "normative_hierarchy_metrics": aggregate_by_field(results, "requires_normative_hierarchy"),
        "failure_analysis": failure_analysis(results),
        "results": list(results),
    }
    return output


def csv_rows(output: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in output["results"]:
        retrieval = result["retrieval_metrics"]
        answer = result["answer_metrics"]
        e2e = result["end_to_end_metrics"]
        rows.append(
            {
                "id": result["id"],
                "category": result["category"],
                "topic": result.get("topic", ""),
                "difficulty": result.get("difficulty", ""),
                "requires_graph": result.get("requires_graph", False),
                "requires_normative_hierarchy": result.get("requires_normative_hierarchy", False),
                "query": result["query"],
                "end_to_end_passed": e2e["end_to_end_passed"],
                "retrieval_passed": e2e["retrieval_passed"],
                "answer_passed": e2e["answer_passed"],
                "citation_grounding_passed": e2e["citation_grounding_passed"],
                "answer_faithfulness_passed": e2e["answer_faithfulness_passed"],
                "final_quality_score": e2e["final_quality_score"],
                "answer_completeness_score": e2e["answer_completeness_score"],
                "required_citation_coverage": retrieval["required_citation_coverage"],
                "required_citations_found": " || ".join(retrieval["required_citations_found"]),
                "missing_required_citations": " || ".join(retrieval["missing_required_citations"]),
                "forbidden_citation_violations": " || ".join(retrieval["forbidden_citation_violations"]),
                "retrieval_source_distribution": json.dumps(retrieval["retrieval_source_distribution"], ensure_ascii=False),
                "graph_expansion_used": retrieval["graph_expansion_used"],
                "graph_depth": retrieval["graph_depth"],
                "graph_edge_types": json.dumps(retrieval["graph_edge_types"], ensure_ascii=False),
                "citation_validation_passed": answer["citation_validation_passed"],
                "quality_validation_passed": answer["quality_validation_passed"],
                "unsupported_article_numbers": " || ".join(answer["unsupported_article_numbers"]),
                "unretrieved_citations": " || ".join(answer["unretrieved_citations"]),
                "low_information_quotes_count": answer["low_information_quotes_count"],
                "legal_claim_citation_coverage": answer["legal_claim_citation_coverage"],
                "no_article_title_only_answer": answer["no_article_title_only_answer"],
                "insufficient_context_handled": answer["insufficient_context_handled"],
                "failure_reasons": " || ".join(e2e["failure_reasons"]),
                "top3_citations": " || ".join(
                    str(context["citation_text"])
                    for context in retrieval["top_k_contexts"][:3]
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_rate(value: object) -> str:
    return f"{float(value):.3f}"


def first_sentence(text: str, *, max_chars: int = 360) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rsplit(" ", 1)[0] + "..."


def successful_examples(results: Sequence[dict[str, object]], *, limit: int = 3) -> list[dict[str, object]]:
    values = [
        result
        for result in results
        if result["end_to_end_metrics"]["end_to_end_passed"]
        and result["retrieval_metrics"]["graph_expansion_used"]
    ]
    values.sort(key=lambda result: (-float(result["end_to_end_metrics"]["final_quality_score"]), str(result["id"])))
    return values[:limit]


def load_mode_comparison(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def render_group_metrics_table(metrics_by_group: dict[str, dict[str, object]], group_label: str) -> list[str]:
    rows = [
        (
            group,
            metrics["query_count"],
            format_rate(metrics["retrieval_pass_rate"]),
            format_rate(metrics["answer_pass_rate"]),
            format_rate(metrics["citation_pass_rate"]),
            format_rate(metrics["quality_pass_rate"]),
            format_rate(metrics["end_to_end_pass_rate"]),
            f"{float(metrics['average_quality_score']):.2f}",
        )
        for group, metrics in metrics_by_group.items()
    ]
    return markdown_table(
        (
            group_label,
            "Queries",
            "Retrieval pass",
            "Answer pass",
            "Citation pass",
            "Quality pass",
            "E2E pass",
            "Avg quality",
        ),
        rows,
    )


def render_mode_comparison(output: dict[str, object]) -> list[str]:
    comparison = output.get("mode_comparison")
    if not isinstance(comparison, dict):
        return ["- Retrieval mode comparison was not provided for this run."]

    overall = output["overall"]
    retrieval_metrics = comparison.get("overall_metrics") or {}
    graph_metrics = retrieval_metrics.get("graph_augmented") or {}
    rows = []
    for mode in ("vector_only", "hybrid", "graph_augmented"):
        metrics = retrieval_metrics.get(mode) or {}
        rows.append(
            (
                mode,
                f"{float(metrics.get('recall_at_10', 0.0)):.3f}",
                f"{float(metrics.get('required_citation_coverage', 0.0)):.3f}",
                "N/A",
                "N/A",
                "N/A",
            )
        )
    rows.append(
        (
            "end_to_end_graph_rag",
            f"{float(graph_metrics.get('recall_at_10', 0.0)):.3f}",
            format_rate(overall["retrieval_pass_rate"]),
            format_rate(overall["end_to_end_pass_rate"]),
            format_rate(overall["citation_validation_pass_rate"]),
            format_rate(overall["quality_validation_pass_rate"]),
        )
    )
    return markdown_table(
        (
            "System",
            "Recall@10",
            "Required citation coverage",
            "End-to-end pass rate",
            "Citation validation pass rate",
            "Quality validation pass rate",
        ),
        rows,
    )


def render_report(output: dict[str, object]) -> str:
    overall = output["overall"]
    results = output["results"]
    failure = output["failure_analysis"]

    lines: list[str] = [
        "# End-to-End Legal RAG Evaluation",
        "",
        f"- Generated at: {output['generated_at']}",
        f"- Benchmark queries: {output['benchmark_count']}",
        f"- Top K: {output['top_k']}",
        f"- Provider: {output['provider']}",
        f"- End-to-end passed: {output['passed']}",
        "",
        "## Overall Summary",
        "",
    ]
    lines.extend(
        markdown_table(
            (
                "Queries",
                "E2E pass rate",
                "Retrieval pass rate",
                "Answer pass rate",
                "Citation pass rate",
                "Quality pass rate",
                "Avg quality score",
                "Low-info quotes",
            ),
            [
                (
                    overall["query_count"],
                    format_rate(overall["end_to_end_pass_rate"]),
                    format_rate(overall["retrieval_pass_rate"]),
                    format_rate(overall["answer_pass_rate"]),
                    format_rate(overall["citation_validation_pass_rate"]),
                    format_rate(overall["quality_validation_pass_rate"]),
                    f"{float(overall['average_final_quality_score']):.2f}",
                    overall["low_information_quotes_count"],
                )
            ],
        )
    )
    lines.extend(
        [
            "",
            f"- Unsupported article numbers: {', '.join(overall['unsupported_article_numbers']) or 'None'}",
            f"- Unretrieved citations: {', '.join(overall['unretrieved_citations']) or 'None'}",
            f"- Graph expansion used: {overall['graph_expansion_used_count']} queries",
            f"- Average graph depth: {float(overall['average_graph_depth']):.3f}",
            "",
            "## Per-Category Results",
            "",
        ]
    )
    category_rows = []
    for category, metrics in output["category_metrics"].items():
        category_rows.append(
            (
                category,
                metrics["query_count"],
                format_rate(metrics["retrieval_pass_rate"]),
                format_rate(metrics["answer_pass_rate"]),
                format_rate(metrics["citation_pass_rate"]),
                format_rate(metrics["end_to_end_pass_rate"]),
                f"{float(metrics['average_quality_score']):.2f}",
            )
        )
    lines.extend(
        markdown_table(
            (
                "Category",
                "Queries",
                "Retrieval pass",
                "Answer pass",
                "Citation pass",
                "E2E pass",
                "Avg quality",
            ),
            category_rows,
        )
    )

    lines.extend(["", "## Per-Topic Results", ""])
    lines.extend(render_group_metrics_table(output["topic_metrics"], "Topic"))

    lines.extend(["", "## Per-Difficulty Results", ""])
    lines.extend(render_group_metrics_table(output["difficulty_metrics"], "Difficulty"))

    lines.extend(["", "## Graph-Required Results", ""])
    lines.extend(render_group_metrics_table(output["graph_required_metrics"], "Requires graph"))

    lines.extend(["", "## Normative-Hierarchy Results", ""])
    lines.extend(render_group_metrics_table(output["normative_hierarchy_metrics"], "Requires hierarchy"))

    lines.extend(["", "## Retrieval Mode And End-To-End Comparison", ""])
    lines.extend(render_mode_comparison(output))

    lines.extend(["", "## Per-Query Results", ""])
    query_rows = []
    for result in results:
        retrieval = result["retrieval_metrics"]
        answer = result["answer_metrics"]
        e2e = result["end_to_end_metrics"]
        query_rows.append(
            (
                result["id"],
                result["category"],
                result.get("topic", ""),
                result.get("difficulty", ""),
                "Pass" if e2e["end_to_end_passed"] else "Fail",
                f"{float(retrieval['required_citation_coverage']):.2f}",
                "Pass" if answer["citation_validation_passed"] else "Fail",
                "Pass" if answer["quality_validation_passed"] else "Fail",
                f"{float(e2e['final_quality_score']):.2f}",
                ", ".join(e2e["failure_reasons"]) or "None",
            )
        )
    lines.extend(
        markdown_table(
            (
                "ID",
                "Category",
                "Topic",
                "Difficulty",
                "E2E",
                "Required coverage",
                "Citation",
                "Quality",
                "Score",
                "Failure reasons",
            ),
            query_rows,
        )
    )

    lines.extend(["", "## Successful Graph-Augmented Answers", ""])
    examples = successful_examples(results)
    if examples:
        for result in examples:
            top_citations = [
                str(context["citation_text"])
                for context in result["retrieval_metrics"]["top_k_contexts"][:3]
            ]
            lines.extend(
                [
                    f"### {result['id']}",
                    "",
                    f"- Query: {result['query']}",
                    f"- Top citations: {'; '.join(top_citations)}",
                    f"- Answer excerpt: {first_sentence(str(result['answer']))}",
                    "",
                ]
            )
    else:
        lines.append("- No fully passing graph-expanded examples were found.")

    lines.extend(["", "## Retrieval Failures", ""])
    failed_queries = failure["failed_queries"]
    retrieval_failures = [
        item for item in failed_queries if {"retrieval_missing_required_context", "retrieval_over_expansion"}.intersection(item["failure_reasons"])
    ]
    if retrieval_failures:
        for item in retrieval_failures:
            lines.extend(
                [
                    f"- `{item['id']}`: {', '.join(item['failure_reasons'])}",
                    f"  Query: {item['query']}",
                    f"  Missing required citations: {', '.join(item['missing_required_citations']) or 'None'}",
                    f"  Citation issues: {', '.join(item['unretrieved_citations']) or 'None'}",
                ]
            )
    else:
        lines.append("- No retrieval failures were observed in this benchmark run.")

    lines.extend(["", "## Answer Generation Failures", ""])
    answer_failures = [
        item for item in failed_queries if not {"retrieval_missing_required_context", "retrieval_over_expansion"}.intersection(item["failure_reasons"])
    ]
    if answer_failures:
        for item in answer_failures:
            lines.extend(
                [
                    f"- `{item['id']}`: {', '.join(item['failure_reasons'])}",
                    f"  Query: {item['query']}",
                    f"  Citation issues: {', '.join(item['unretrieved_citations']) or 'None'}",
                ]
            )
    else:
        lines.append("- No answer-generation failures were observed in this benchmark run.")

    lines.extend(["", "## Remaining Limitations", ""])
    if failed_queries:
        lines.append(
            "- Some benchmark items did not satisfy all required-citation checks; these failures should be interpreted "
            "as retrieval coverage gaps for the constructed benchmark, not as a full assessment of legal correctness."
        )
    else:
        lines.append("- No end-to-end failures were observed in this benchmark run.")
    lines.append(
        "- The benchmark is manually constructed from selected Vietnamese labor-law topics and does not prove universal legal correctness."
    )
    lines.append(
        "- Deterministic answer synthesis favors citation safety and may be less fluent than a carefully constrained LLM provider."
    )

    lines.extend(
        [
            "",
            "## Failure Reason Counts",
            "",
        ]
    )
    reason_counts = failure["failure_reason_counts"]
    if reason_counts:
        lines.extend(markdown_table(("Reason", "Count"), sorted(reason_counts.items())))
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Thesis-Ready Conclusion",
            "",
            (
                "Based on the constructed benchmark, the end-to-end evaluation indicates that graph-augmented retrieval "
                "is most helpful for Vietnamese labor-law questions that require connecting multiple provisions, such as "
                "Labor Code rules with implementing decrees, circular guidance, exceptions, or court-jurisdiction rules. "
                "The results should not be read as a claim of universal legal correctness. Instead, they show that the "
                "system can maintain citation grounding on this benchmark by only citing retrieved legal contexts, which "
                "reduces hallucinated legal references and makes remaining retrieval gaps easier to inspect."
            ),
        ]
    )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    output = run_end_to_end(args)
    comparison_path = args.comparison_summary_path
    if comparison_path is not None:
        output["mode_comparison"] = load_mode_comparison(comparison_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(args.output_prefix or "end_to_end_rag").strip()
    json_path = args.output_dir / f"{output_prefix}_results.json"
    csv_path = args.output_dir / f"{output_prefix}_results.csv"
    report_path = args.output_dir / f"{output_prefix}_report.md"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, csv_rows(output))
    report_path.write_text(render_report(output), encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": output["passed"],
                "benchmark_count": output["benchmark_count"],
                "end_to_end_pass_rate": output["overall"]["end_to_end_pass_rate"],
                "citation_validation_pass_rate": output["overall"]["citation_validation_pass_rate"],
                "low_information_quotes_count": output["overall"]["low_information_quotes_count"],
                "json_path": str(json_path),
                "csv_path": str(csv_path),
                "report_path": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
