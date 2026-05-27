from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vn_labor_law_ai_assistant.rag.answering import (
    generate_grounded_answer,
    validate_answer_quality,
)
from vn_labor_law_ai_assistant.rag.retrieval import HybridRetriever, RetrievalContext


TEST_QUERIES = (
    "Người 14 tuổi có được làm việc không?",
    "Người chưa đủ 15 tuổi làm việc cần điều kiện gì?",
    "Nữ nghỉ hưu năm 2026 thì bao nhiêu tuổi?",
    "Hợp đồng lao động cần có những nội dung gì?",
    "Tranh chấp sa thải có cần hòa giải trước khi kiện không?",
    "Người lao động đơn phương chấm dứt hợp đồng trái luật thì phải bồi thường gì?",
    "Công ty thay đổi cơ cấu thì phải trả trợ cấp gì?",
    "Khi nào người lao động được nghỉ việc không cần báo trước?",
    "Trường hợp nào được làm thêm giờ và giới hạn làm thêm theo tháng là bao nhiêu?",
)


class EnvOverride:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values
        self.previous: dict[str, str | None] = {}

    def __enter__(self) -> None:
        self.previous = {key: os.environ.get(key) for key in self.values}
        os.environ.update(self.values)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grounded answer generation over graph-augmented retrieval.")
    parser.add_argument("--index-path", type=Path, default=REPO_ROOT / "artifacts" / "index")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "answers")
    parser.add_argument(
        "--fallback-retrieval-results-path",
        type=Path,
        default=REPO_ROOT / "artifacts" / "retrieval" / "graph_augmented_retrieval_test_results.json",
    )
    parser.add_argument(
        "--fallback-answer-results-path",
        type=Path,
        default=REPO_ROOT / "artifacts" / "answers" / "grounded_answer_test_results.json",
    )
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--prefetch-limit", type=int, default=24)
    parser.add_argument("--max-answer-contexts", type=int, default=8)
    parser.add_argument(
        "--provider",
        type=str,
        default=os.getenv("ANSWER_GENERATION_PROVIDER", "extractive"),
        help="Use 'extractive' for deterministic offline generation, or 'groq'/'auto' to call the LLM with fallback.",
    )
    parser.add_argument("--model", type=str, default=os.getenv("ANSWER_GENERATION_MODEL", os.getenv("GROQ_MODEL", "")))
    parser.add_argument("--reranker-model", type=str, default=os.getenv("RERANKER_MODEL", ""))
    parser.add_argument("--embedding-provider", type=str, default=os.getenv("GRAPH_RETRIEVAL_TEST_EMBEDDING_PROVIDER", "sentence_transformers"))
    parser.add_argument("--device", type=str, default=os.getenv("GRAPH_RETRIEVAL_TEST_DEVICE", "cpu"))
    return parser.parse_args()


def load_index_record_lookup(index_path: Path) -> dict[str, dict[str, Any]]:
    manifest_path = index_path / "current.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records_path = REPO_ROOT / str(manifest.get("records_jsonl_path") or "")
    if not records_path.exists():
        records_path = Path(str(manifest.get("records_jsonl_path") or ""))
    if not records_path.exists():
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        chunk_id = str(row.get("chunk_id") or row.get("payload", {}).get("chunk_id") or "")
        if chunk_id:
            lookup[chunk_id] = row
    return lookup


def retrieval_context_from_mapping(
    item: dict[str, Any],
    *,
    record_lookup: dict[str, dict[str, Any]],
) -> RetrievalContext:
    chunk_id = str(item.get("chunk_id") or "")
    record = record_lookup.get(chunk_id, {})
    record_payload = dict(record.get("payload") or {})
    payload = {**record_payload, **item}
    citation_text = str(item.get("citation_text") or record.get("citation_text") or payload.get("citation_text") or "")
    text = str(
        record.get("text")
        or item.get("text")
        or item.get("retrieval_text")
        or payload.get("retrieval_text")
        or ""
    )
    matched_citations = item.get("matched_citations")
    if isinstance(matched_citations, list):
        matched_citations_tuple = tuple(str(value) for value in matched_citations)
    else:
        matched_citations_tuple = (citation_text,) if citation_text else ()
    matched_chunk_ids = item.get("matched_chunk_ids")
    if isinstance(matched_chunk_ids, list):
        matched_chunk_ids_tuple = tuple(str(value) for value in matched_chunk_ids)
    else:
        matched_chunk_ids_tuple = (chunk_id,) if chunk_id else ()
    score = item.get("final_score", item.get("score", record.get("score", 0.0)))
    try:
        score_value = float(score or 0.0)
    except (TypeError, ValueError):
        score_value = 0.0
    return RetrievalContext(
        chunk_id=chunk_id,
        citation_text=citation_text,
        text=text,
        payload=payload,
        score=score_value,
        matched_chunk_ids=matched_chunk_ids_tuple,
        matched_citations=matched_citations_tuple,
    )


def load_fallback_query_contexts(args: argparse.Namespace) -> dict[str, tuple[RetrievalContext, ...]]:
    record_lookup = load_index_record_lookup(args.index_path)
    query_contexts: dict[str, tuple[RetrievalContext, ...]] = {}

    if args.fallback_retrieval_results_path.exists():
        retrieval_report = json.loads(args.fallback_retrieval_results_path.read_text(encoding="utf-8"))
        for item in retrieval_report.get("queries", []):
            if not isinstance(item, dict):
                continue
            query = str(item.get("query") or "")
            contexts = item.get("contexts") or []
            if query and isinstance(contexts, list):
                query_contexts[query] = tuple(
                    retrieval_context_from_mapping(context, record_lookup=record_lookup)
                    for context in contexts
                    if isinstance(context, dict)
                )

    if args.fallback_answer_results_path.exists():
        answer_report = json.loads(args.fallback_answer_results_path.read_text(encoding="utf-8"))
        for item in answer_report.get("results", []):
            if not isinstance(item, dict):
                continue
            query = str(item.get("query") or "")
            if not query or query in query_contexts:
                continue
            contexts = item.get("ordered_answer_contexts") or []
            if isinstance(contexts, list):
                query_contexts[query] = tuple(
                    retrieval_context_from_mapping(context, record_lookup=record_lookup)
                    for context in contexts
                    if isinstance(context, dict)
                )
    return query_contexts


def context_to_json(context: RetrievalContext) -> dict[str, object]:
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
        "graph_depth": payload.get("graph_depth"),
        "graph_edge_types": payload.get("graph_edge_types") or [],
    }


def run_generation(args: argparse.Namespace) -> dict[str, object]:
    env = {
        "LEGAL_GRAPH_ENABLED": "true",
        "LEGAL_GRAPH_BACKEND": "neo4j",
        "LEGAL_GRAPH_COMPLEX_QUERY_ONLY": "true",
        "LEGAL_GRAPH_MAX_EXPANDED_CHUNKS": "16",
        "LEGAL_GRAPH_EXPANSION_DEPTH": "2",
        "EMBEDDING_PROVIDER": args.embedding_provider,
    }
    query_results: list[dict[str, object]] = []
    retrieval_fallback_used = False
    retrieval_error = ""

    def append_generated_result(query: str, contexts: Sequence[RetrievalContext]) -> None:
        answer_result = generate_grounded_answer(
            query,
            contexts,
            provider=args.provider,
            model=args.model,
            max_answer_contexts=args.max_answer_contexts,
        )
        quality_validation = validate_answer_quality(
            query,
            answer_result.parsed,
            answer_result.contexts,
            final_answer=answer_result.answer,
        )
        passed = answer_result.validation.passed and quality_validation.passed
        query_results.append(
            {
                "query": query,
                "retrieved_top_citations": [
                    context.citation_text for context in contexts[: args.top_k]
                ],
                "ordered_answer_contexts": [
                    context_to_json(context)
                    for context in answer_result.contexts
                ],
                "answer": answer_result.answer,
                "legal_basis": list(answer_result.parsed.legal_basis),
                "evidence_quotes": [
                    asdict(quote) for quote in answer_result.parsed.evidence_quotes
                ],
                "insufficient_context": answer_result.parsed.insufficient_context,
                "validation": asdict(answer_result.validation),
                "quality_validation": asdict(quality_validation),
                "passed": passed,
                "provider": answer_result.provider,
                "model": answer_result.model,
                "generation_method": answer_result.generation_method,
            }
        )

    try:
        with EnvOverride(env):
            retriever = HybridRetriever(
                index_path=args.index_path,
                device=args.device,
                reranker_model=args.reranker_model,
                query_router_enabled=False,
            )
            try:
                for query in TEST_QUERIES:
                    retrieval_result = retriever.retrieve(
                        query,
                        top_k=args.top_k,
                        prefetch_limit=args.prefetch_limit,
                    )
                    append_generated_result(query, retrieval_result.contexts)
            finally:
                retriever.close()
    except Exception as exc:
        retrieval_fallback_used = True
        retrieval_error = str(exc)
        query_contexts = load_fallback_query_contexts(args)
        for query in TEST_QUERIES:
            contexts = query_contexts.get(query, ())
            if not contexts:
                raise RuntimeError(
                    f"No fallback graph-augmented retrieval contexts available for query: {query}"
                ) from exc
            append_generated_result(query, contexts)

    passed = all(bool(item["passed"]) for item in query_results)
    citation_validation_passed = all(bool(item["validation"]["passed"]) for item in query_results)
    quality_validation_passed = all(bool(item["quality_validation"]["passed"]) for item in query_results)
    used_providers = sorted({str(item["provider"]) for item in query_results})
    used_models = sorted({str(item["model"]) for item in query_results})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "citation_validation_passed": citation_validation_passed,
        "quality_validation_passed": quality_validation_passed,
        "query_count": len(query_results),
        "provider": used_providers[0] if len(used_providers) == 1 else ", ".join(used_providers),
        "model": used_models[0] if len(used_models) == 1 else ", ".join(used_models),
        "retrieval_fallback_used": retrieval_fallback_used,
        "retrieval_error": retrieval_error,
        "results": query_results,
    }


def markdown_list(values: Sequence[str], *, limit: int = 8) -> list[str]:
    if not values:
        return ["- None"]
    return [f"- {value}" for value in values[:limit]]


QUALITY_CHECK_LABELS = (
    ("direct_answer_present", "Direct answer present"),
    ("required_legal_rule_present", "Required legal rule present"),
    ("numeric_answer_present", "Numeric answer present"),
    ("yes_no_answer_present", "Yes/no answer present"),
    ("conditions_listed", "Conditions listed"),
    ("exception_answer_present", "Exception answer present"),
    ("no_article_title_only_answer", "No article-title-only answer"),
    ("all_legal_claims_have_citations", "Legal claim citation coverage"),
)


def display_validation_value(value: object) -> str:
    if value == "not_applicable":
        return "N/A"
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def render_quality_check_lines(quality: dict[str, object]) -> list[str]:
    return [
        f"- {label}: {display_validation_value(quality.get(key))}"
        for key, label in QUALITY_CHECK_LABELS
    ]


def render_markdown(output: dict[str, object]) -> str:
    lines: list[str] = [
        "# Grounded Answer Generation Test Results",
        "",
        f"- Generated at: {output['generated_at']}",
        f"- Queries: {output['query_count']}",
        f"- Provider: {output['provider']}",
        f"- Model: {output['model'] or 'default'}",
        f"- Passed: {output['passed']}",
        f"- Citation validation passed: {output['citation_validation_passed']}",
        f"- Quality validation passed: {output['quality_validation_passed']}",
        "",
    ]
    for index, item in enumerate(output["results"], start=1):
        validation = item["validation"]
        quality = item["quality_validation"]
        lines.extend(
            [
                f"## {index}. {item['query']}",
                "",
                f"- Passed: {item['passed']}",
                f"- Generation method: {item['generation_method']}",
                f"- Citation validation: {validation['passed']}",
                f"- Quality validation: {quality['passed']}",
                f"- Applied answer intent: {quality['applied_answer_intent']}",
                f"- Low-information quotes: {quality['low_information_quotes_count']}",
                f"- Unsupported article numbers: {', '.join(validation['unsupported_article_numbers']) or 'None'}",
                f"- Unretrieved citations: {', '.join(validation['unretrieved_citations']) or 'None'}",
                f"- Citation warnings: {', '.join(validation['warnings']) or 'None'}",
                f"- Quality warnings: {', '.join(quality['warnings']) or 'None'}",
                "",
                "### Retrieved Top Citations",
                "",
                *markdown_list(item["retrieved_top_citations"], limit=10),
                "",
                "### Generated Answer",
                "",
                str(item["answer"]).strip(),
                "",
                "### Quality Checks",
                "",
                *render_quality_check_lines(quality),
                "",
                "### Legal Basis",
                "",
                *markdown_list(item["legal_basis"], limit=10),
                "",
                "### Evidence Quotes",
                "",
                *markdown_list(
                    [
                        f"{quote['citation']}: {quote['quote']}"
                        for quote in item["evidence_quotes"]
                    ],
                    limit=10,
                ),
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def build_quality_output(output: dict[str, object]) -> dict[str, object]:
    return {
        "generated_at": output["generated_at"],
        "passed": output["quality_validation_passed"],
        "query_count": output["query_count"],
        "provider": output["provider"],
        "model": output["model"],
        "results": [
            {
                "query": item["query"],
                "passed": item["quality_validation"]["passed"],
                "quality_validation": item["quality_validation"],
                "citation_validation_passed": item["validation"]["passed"],
                "legal_basis": item["legal_basis"],
                "evidence_quotes": item["evidence_quotes"],
                "answer": item["answer"],
            }
            for item in output["results"]
        ],
    }


def render_quality_markdown(output: dict[str, object]) -> str:
    quality_output = build_quality_output(output)
    lines: list[str] = [
        "# Grounded Answer Quality Report",
        "",
        f"- Generated at: {quality_output['generated_at']}",
        f"- Queries: {quality_output['query_count']}",
        f"- Passed: {quality_output['passed']}",
        "",
    ]
    for index, item in enumerate(quality_output["results"], start=1):
        quality = item["quality_validation"]
        lines.extend(
            [
                f"## {index}. {item['query']}",
                "",
                f"- Passed: {item['passed']}",
                f"- Intent: {quality['applied_answer_intent']}",
                f"- Citation validation: {display_validation_value(item['citation_validation_passed'])}",
                f"- Quality validation: {display_validation_value(item['passed'])}",
                *render_quality_check_lines(quality),
                f"- Low-information quotes: {quality['low_information_quotes_count']}",
                f"- Warnings: {', '.join(quality['warnings']) or 'None'}",
                "",
                "### Generated Answer",
                "",
                str(item["answer"]).strip(),
                "",
                "### Legal Basis",
                "",
                *markdown_list(item["legal_basis"], limit=10),
                "",
                "### Evidence Quotes",
                "",
                *markdown_list(
                    [
                        f"{quote['citation']}: {quote['quote']}"
                        for quote in item["evidence_quotes"]
                    ],
                    limit=10,
                ),
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    output = run_generation(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "grounded_answer_test_results.json"
    md_path = args.output_dir / "grounded_answer_test_results.md"
    quality_json_path = args.output_dir / "grounded_answer_quality_report.json"
    quality_md_path = args.output_dir / "grounded_answer_quality_report.md"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(output), encoding="utf-8")
    quality_output = build_quality_output(output)
    quality_json_path.write_text(json.dumps(quality_output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    quality_md_path.write_text(render_quality_markdown(output), encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": output["passed"],
                "citation_validation_passed": output["citation_validation_passed"],
                "quality_validation_passed": output["quality_validation_passed"],
                "query_count": output["query_count"],
                "json_path": str(json_path),
                "md_path": str(md_path),
                "quality_json_path": str(quality_json_path),
                "quality_md_path": str(quality_md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
