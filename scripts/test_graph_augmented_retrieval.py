from __future__ import annotations

import argparse
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

from vn_labor_law_ai_assistant.rag.retrieval import HybridRetriever


TEST_QUERIES = (
    "Người 14 tuổi có được làm việc không?",
    "Người chưa đủ 15 tuổi làm việc cần điều kiện gì?",
    "Nữ nghỉ hưu năm 2026 thì bao nhiêu tuổi?",
    "Hợp đồng lao động cần có những nội dung gì?",
    "Tranh chấp sa thải có cần hòa giải trước khi kiện không?",
    "Người lao động đơn phương chấm dứt hợp đồng trái luật thì phải bồi thường gì?",
    "Công ty thay đổi cơ cấu thì phải trả trợ cấp gì?",
    "Khi nào người lao động được nghỉ việc không cần báo trước?",
)

EXPECTED_BEHAVIOR: dict[str, dict[str, object]] = {
    TEST_QUERIES[0]: {
        "required_bll_articles_any": ("143", "145", "146", "147"),
        "required_documents": ("thong-tu-09-2020-tt-bldtbxh",),
    },
    TEST_QUERIES[1]: {
        "required_bll_articles_any": ("143", "145", "146", "147"),
        "required_documents": ("thong-tu-09-2020-tt-bldtbxh",),
    },
    TEST_QUERIES[2]: {
        "required_bll_articles_any": ("169",),
        "required_documents": ("nghi-dinh-135-2020-nd-cp",),
    },
    TEST_QUERIES[3]: {
        "required_bll_articles_any": ("21",),
        "required_documents": ("thong-tu-10-2020-tt-bldtbxh",),
    },
    TEST_QUERIES[4]: {
        "required_bll_articles_any": ("179", "188", "190"),
        "required_documents": ("92-2015-qh13-labor-only",),
    },
    TEST_QUERIES[5]: {
        "required_bll_articles_any": ("39", "40", "41"),
        "required_documents_any": ("nghi-dinh-145-2020-nd-cp",),
    },
    TEST_QUERIES[6]: {
        "required_bll_articles_any": ("42", "44", "47"),
        "required_documents_any": ("nghi-dinh-145-2020-nd-cp",),
    },
    TEST_QUERIES[7]: {
        "required_bll_articles_any": ("35",),
        "required_documents_any": ("nghi-dinh-145-2020-nd-cp",),
    },
}

STRICT_EXPECTATIONS: dict[str, dict[str, object]] = {
    TEST_QUERIES[0]: {
        "required": (
            {"label": "BLLĐ Điều 143", "document_id": "45-2019-qh14", "article_number": "143", "top_n": 12},
            {"label": "BLLĐ Điều 145", "document_id": "45-2019-qh14", "article_number": "145", "top_n": 12},
            {"label": "BLLĐ Điều 146", "document_id": "45-2019-qh14", "article_number": "146", "top_n": 12},
            {"label": "TT09 hướng dẫn lao động chưa thành niên", "document_id": "thong-tu-09-2020-tt-bldtbxh", "top_n": 12},
        ),
    },
    TEST_QUERIES[1]: {
        "required": (
            {"label": "BLLĐ Điều 145", "document_id": "45-2019-qh14", "article_number": "145", "top_n": 12},
            {"label": "BLLĐ Điều 146", "document_id": "45-2019-qh14", "article_number": "146", "top_n": 12},
            {"label": "TT09 Điều 3", "document_id": "thong-tu-09-2020-tt-bldtbxh", "article_number": "3", "top_n": 12},
        ),
    },
    TEST_QUERIES[2]: {
        "required": (
            {"label": "BLLĐ Điều 169", "document_id": "45-2019-qh14", "article_number": "169", "top_n": 12},
            {"label": "NĐ135 Điều 4", "document_id": "nghi-dinh-135-2020-nd-cp", "article_number": "4", "top_n": 12},
            {"label": "NĐ135 bảng/phụ lục tuổi nghỉ hưu", "document_id": "nghi-dinh-135-2020-nd-cp", "chunk_id_contains": "Phu_Luc", "top_n": 12},
        ),
    },
    TEST_QUERIES[3]: {
        "required": (
            {"label": "BLLĐ Điều 21 trong top 5", "document_id": "45-2019-qh14", "article_number": "21", "top_n": 5},
            {"label": "TT10 Điều 3 trong top 5", "document_id": "thong-tu-10-2020-tt-bldtbxh", "article_number": "3", "top_n": 5},
        ),
        "forbidden_top5": (
            {"label": "NĐ145 Điều 17 ký quỹ", "document_id": "nghi-dinh-145-2020-nd-cp", "article_number": "17"},
        ),
    },
    TEST_QUERIES[4]: {
        "required": (
            {"label": "BLLĐ Điều 188", "document_id": "45-2019-qh14", "article_number": "188", "top_n": 12},
            {"label": "BLTTDS Điều 32", "document_id": "92-2015-qh13-labor-only", "article_number": "32", "top_n": 12},
            {"label": "BLLĐ Điều 190", "document_id": "45-2019-qh14", "article_number": "190", "top_n": 12},
        ),
    },
    TEST_QUERIES[5]: {
        "required": (
            {"label": "BLLĐ Điều 40 trong top 3", "document_id": "45-2019-qh14", "article_number": "40", "top_n": 3},
            {"label": "BLLĐ Điều 39 trong top 5", "document_id": "45-2019-qh14", "article_number": "39", "top_n": 5},
        ),
    },
    TEST_QUERIES[6]: {
        "required": (
            {"label": "BLLĐ Điều 42", "document_id": "45-2019-qh14", "article_number": "42", "top_n": 12},
            {"label": "BLLĐ Điều 47", "document_id": "45-2019-qh14", "article_number": "47", "top_n": 12},
            {"label": "NĐ145 Điều 8", "document_id": "nghi-dinh-145-2020-nd-cp", "article_number": "8", "top_n": 12},
        ),
        "forbidden_top5": (
            {"label": "BLLĐ Điều 40", "document_id": "45-2019-qh14", "article_number": "40"},
        ),
    },
    TEST_QUERIES[7]: {
        "required": (
            {"label": "BLLĐ Điều 35 khoản 2 trong top 3", "document_id": "45-2019-qh14", "article_number": "35", "clause_ref": "2", "top_n": 3},
        ),
        "not_above_required": (
            {"label": "BLLĐ Điều 40", "document_id": "45-2019-qh14", "article_number": "40"},
            {"label": "BLLĐ Điều 41", "document_id": "45-2019-qh14", "article_number": "41"},
            {"label": "BLLĐ Điều 48", "document_id": "45-2019-qh14", "article_number": "48"},
        ),
    },
}

CONTEXT_METADATA_FIELDS = (
    "chunk_id",
    "document_id",
    "document_title",
    "document_type",
    "normative_rank",
    "rank_label",
    "article_number",
    "article_title",
    "clause_ref",
    "point_ref",
    "point_refs",
    "citation_text",
    "retrieval_text",
    "graph_path",
    "graph_depth",
    "graph_edge_types",
    "retrieval_source",
)
DEBUG_FIELDS = (
    "vector_score",
    "graph_score",
    "final_score",
    "seed_chunk_ids",
    "expanded_node_ids",
    "graph_paths",
    "applied_query_intent",
    "expansion_depth",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run graph-augmented retrieval smoke queries.")
    parser.add_argument("--index-path", type=Path, default=REPO_ROOT / "artifacts" / "index")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "retrieval",
    )
    parser.add_argument("--reranker-model", type=str, default=os.getenv("RERANKER_MODEL", ""))
    parser.add_argument(
        "--embedding-provider",
        type=str,
        default=os.getenv("GRAPH_RETRIEVAL_TEST_EMBEDDING_PROVIDER", "sentence_transformers"),
        help="Embedding provider for query vectors. Defaults to local sentence-transformers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv("GRAPH_RETRIEVAL_TEST_DEVICE", "cpu"),
        help="Device for the local sentence-transformer query encoder.",
    )
    return parser.parse_args()


def context_to_dict(context: Any) -> dict[str, object]:
    payload = dict(context.payload)
    item = {field: payload.get(field) for field in CONTEXT_METADATA_FIELDS}
    item.update({field: payload.get(field) for field in DEBUG_FIELDS})
    item["chunk_id"] = context.chunk_id
    item["citation_text"] = context.citation_text
    item["score"] = context.score
    item["matched_chunk_ids"] = list(context.matched_chunk_ids)
    item["matched_citations"] = list(context.matched_citations)
    return item


def hit_to_debug(hit: Any) -> dict[str, object]:
    payload = dict(hit.payload)
    return {
        "chunk_id": hit.chunk_id,
        "citation_text": hit.citation_text,
        "score": hit.score,
        "retrieval_source": payload.get("retrieval_source"),
        "retrieval_method": payload.get("retrieval_method"),
        "document_id": payload.get("document_id"),
        "article_number": payload.get("article_number"),
        **{field: payload.get(field) for field in DEBUG_FIELDS},
        "graph_edge_types": payload.get("graph_edge_types"),
    }


def _has_document(contexts: Sequence[dict[str, object]], document_id: str) -> bool:
    return any(context.get("document_id") == document_id for context in contexts)


def _has_bll_article(contexts: Sequence[dict[str, object]], article_number: str) -> bool:
    return any(
        context.get("document_id") == "45-2019-qh14"
        and str(context.get("article_number") or "") == article_number
        for context in contexts
    )


def _article_label(context: dict[str, object]) -> str:
    document_id = str(context.get("document_id") or "")
    article_number = str(context.get("article_number") or "")
    clause_ref = str(context.get("clause_ref") or "")
    if not article_number:
        return f"{document_id}:appendix"
    if clause_ref:
        return f"{document_id}:Điều {article_number}:khoản {clause_ref}"
    return f"{document_id}:Điều {article_number}"


def _context_matches(context: dict[str, object], spec: dict[str, object]) -> bool:
    if "document_id" in spec and context.get("document_id") != spec["document_id"]:
        return False
    if "article_number" in spec and str(context.get("article_number") or "") != str(spec["article_number"]):
        return False
    if "clause_ref" in spec and str(context.get("clause_ref") or "") != str(spec["clause_ref"]):
        return False
    if "chunk_id_contains" in spec and str(spec["chunk_id_contains"]) not in str(context.get("chunk_id") or ""):
        return False
    return True


def _find_first_position(contexts: Sequence[dict[str, object]], spec: dict[str, object]) -> int | None:
    top_n = int(spec.get("top_n") or len(contexts))
    for index, context in enumerate(contexts[:top_n], start=1):
        if _context_matches(context, spec):
            return index
    return None


def evaluate_strict_query(query: str, contexts: Sequence[dict[str, object]]) -> dict[str, object]:
    expected = STRICT_EXPECTATIONS.get(query, {})
    required = tuple(spec for spec in expected.get("required") or () if isinstance(spec, dict))
    found: list[str] = []
    missing: list[str] = []
    required_positions: dict[str, int] = {}
    for spec in required:
        label = str(spec.get("label") or spec)
        position = _find_first_position(contexts, spec)
        if position is None:
            missing.append(label)
        else:
            found.append(label)
            required_positions[label] = position

    forbidden_found = [
        str(spec.get("label") or spec)
        for spec in tuple(expected.get("forbidden_top5") or ())
        if isinstance(spec, dict)
        and any(_context_matches(context, spec) for context in contexts[:5])
    ]

    ordering_violations: list[str] = []
    first_required_position = min(required_positions.values(), default=None)
    if first_required_position is not None:
        for spec in tuple(expected.get("not_above_required") or ()):
            if not isinstance(spec, dict):
                continue
            forbidden_position = _find_first_position(contexts, {**spec, "top_n": first_required_position - 1})
            if forbidden_position is not None:
                ordering_violations.append(str(spec.get("label") or spec))

    top3 = [_article_label(context) for context in contexts[:3]]
    top5 = [_article_label(context) for context in contexts[:5]]
    return {
        "required_citations_found": found,
        "missing_required_citations": missing,
        "forbidden_or_low_relevance_citations_in_top5": forbidden_found + ordering_violations,
        "top3_article_numbers": top3,
        "top5_article_numbers": top5,
        "passed_strict_validation": not missing and not forbidden_found and not ordering_violations,
    }


def evaluate_query(query: str, contexts: Sequence[dict[str, object]]) -> dict[str, object]:
    expected = EXPECTED_BEHAVIOR.get(query, {})
    required_articles = tuple(str(value) for value in expected.get("required_bll_articles_any") or ())
    required_documents = tuple(str(value) for value in expected.get("required_documents") or ())
    required_documents_any = tuple(str(value) for value in expected.get("required_documents_any") or ())
    duplicate_chunk_ids = len({context["chunk_id"] for context in contexts}) != len(contexts)
    has_relevant_bll = bool(required_articles) and any(
        _has_bll_article(contexts, article) for article in required_articles
    )
    required_documents_present = all(_has_document(contexts, document_id) for document_id in required_documents)
    required_any_document_present = (
        not required_documents_any
        or any(_has_document(contexts, document_id) for document_id in required_documents_any)
    )
    citation_text_preserved = all(bool(context.get("citation_text")) for context in contexts)
    graph_debug_present = any(
        context.get("retrieval_source") in {"graph", "hybrid"}
        and context.get("seed_chunk_ids") is not None
        for context in contexts
    )
    broad_validation = {
        "has_relevant_bll_chunk": has_relevant_bll,
        "required_documents_present": required_documents_present,
        "required_any_document_present": required_any_document_present,
        "citation_text_preserved": citation_text_preserved,
        "no_duplicate_chunk_id": not duplicate_chunk_ids,
        "debug_trace_present": graph_debug_present,
        "passed": (
            has_relevant_bll
            and required_documents_present
            and required_any_document_present
            and citation_text_preserved
            and not duplicate_chunk_ids
            and graph_debug_present
        ),
    }
    return {
        **broad_validation,
        **evaluate_strict_query(query, contexts),
    }


def render_markdown(results: dict[str, object]) -> str:
    lines = [
        "# Graph-Augmented Retrieval Test Results",
        "",
        f"- Generated at: {results['generated_at']}",
        f"- Passed: {results['passed']}",
        f"- Broad validation passed: {results['broad_passed']}",
        f"- Strict validation passed: {results['strict_passed']}",
        f"- Queries: {len(results['queries'])}",
        "",
        "| Query | Broad | Strict | Top citations |",
        "| --- | ---: | ---: | --- |",
    ]
    for item in results["queries"]:
        citations = "; ".join(
            str(context.get("citation_text") or "")
            for context in item["contexts"][:5]
            if context.get("citation_text")
        )
        lines.append(
            f"| {item['query']} | {item['validation']['passed']} | "
            f"{item['validation']['passed_strict_validation']} | {citations} |"
        )
    return "\n".join(lines).strip() + "\n"


def render_strict_markdown(results: dict[str, object]) -> str:
    lines = [
        "# Graph-Augmented Retrieval Strict Validation",
        "",
        f"- Generated at: {results['generated_at']}",
        f"- Strict validation passed: {results['strict_passed']}",
        "",
    ]
    for item in results["queries"]:
        validation = item["validation"]
        lines.extend(
            [
                f"## {item['query']}",
                "",
                f"- Passed strict validation: {validation['passed_strict_validation']}",
                f"- Required citations found: {', '.join(validation['required_citations_found']) or 'None'}",
                f"- Missing required citations: {', '.join(validation['missing_required_citations']) or 'None'}",
                f"- Forbidden/low relevance in top 5: {', '.join(validation['forbidden_or_low_relevance_citations_in_top5']) or 'None'}",
                f"- Top 3 articles: {', '.join(validation['top3_article_numbers'])}",
                f"- Top 5 articles: {', '.join(validation['top5_article_numbers'])}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    os.environ["LEGAL_GRAPH_ENABLED"] = "true"
    os.environ.setdefault("LEGAL_GRAPH_BACKEND", "neo4j")
    os.environ.setdefault("LEGAL_GRAPH_COMPLEX_QUERY_ONLY", "true")
    os.environ.setdefault("LEGAL_GRAPH_MAX_EXPANDED_CHUNKS", "16")
    os.environ.setdefault("LEGAL_GRAPH_EXPANSION_DEPTH", "2")
    if args.embedding_provider:
        os.environ["EMBEDDING_PROVIDER"] = args.embedding_provider

    retriever = HybridRetriever(
        index_path=args.index_path,
        device=args.device or None,
        reranker_model=args.reranker_model,
        query_router_enabled=False,
    )
    query_results: list[dict[str, object]] = []
    try:
        for query in TEST_QUERIES:
            result = retriever.retrieve(query, top_k=args.top_k)
            contexts = [context_to_dict(context) for context in result.contexts[: args.top_k]]
            query_results.append(
                {
                    "query": query,
                    "intent": {
                        "actor_filters": list(result.intent.actor_filters),
                        "topic_filters": list(result.intent.topic_filters),
                        "issue_filters": list(result.intent.issue_filters),
                        "document_filters": list(result.intent.document_filters),
                        "article_numbers": list(result.intent.all_article_numbers),
                        "query_types": list(result.intent.query_types),
                    },
                    "hits_debug": [hit_to_debug(hit) for hit in result.hits[: args.top_k]],
                    "contexts": contexts,
                    "validation": evaluate_query(query, contexts),
                }
            )
    finally:
        retriever.close()

    broad_passed = all(item["validation"]["passed"] for item in query_results)
    strict_passed = all(item["validation"]["passed_strict_validation"] for item in query_results)
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "index_path": str(args.index_path),
        "top_k": args.top_k,
        "broad_passed": broad_passed,
        "strict_passed": strict_passed,
        "passed": broad_passed and strict_passed,
        "queries": query_results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "graph_augmented_retrieval_test_results.json"
    md_path = args.output_dir / "graph_augmented_retrieval_test_results.md"
    strict_md_path = args.output_dir / "graph_augmented_retrieval_strict_validation.md"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(output), encoding="utf-8")
    strict_md_path.write_text(render_strict_markdown(output), encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": output["passed"],
                "broad_passed": output["broad_passed"],
                "strict_passed": output["strict_passed"],
                "json_path": str(json_path),
                "md_path": str(md_path),
                "strict_md_path": str(strict_md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
