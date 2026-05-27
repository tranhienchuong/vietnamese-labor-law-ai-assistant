from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, Sequence

from .legal_reference_edges import read_jsonl
from .rag.graph.structural_parser import legal_unit_node_ids_for_payload


CRITICAL_REFERENCES: set[tuple[str, str]] = {
    ("45-2019-qh14", "21"),
    ("45-2019-qh14", "73"),
    ("45-2019-qh14", "142"),
    ("45-2019-qh14", "143"),
    ("45-2019-qh14", "145"),
    ("45-2019-qh14", "146"),
    ("45-2019-qh14", "147"),
    ("45-2019-qh14", "169"),
    ("92-2015-qh13-labor-only", "32"),
    ("92-2015-qh13-labor-only", "33"),
    ("92-2015-qh13-labor-only", "118"),
    ("92-2015-qh13-labor-only", "119"),
}


def _node_legal_unit_parts(
    node_id: str,
) -> tuple[str | None, str | None, str | None, str | None]:
    parts = node_id.split(":")
    if len(parts) >= 3 and parts[0] in {"article", "clause", "point"}:
        document_id = parts[1]
        article = parts[2]
        clause = parts[3] if len(parts) >= 4 else None
        point = parts[4] if len(parts) >= 5 else None
        return document_id, article, clause, point
    return None, None, None, None


def _is_unresolved_inverse_edge(edge: Mapping[str, object]) -> bool:
    return (
        edge.get("resolved") is False
        and str(edge.get("extraction_method") or "").startswith("inverse:")
    )


def _attempted_target_document_id(edge: Mapping[str, object]) -> str:
    if _is_unresolved_inverse_edge(edge):
        document_id, _, _, _ = _node_legal_unit_parts(str(edge.get("source_id") or ""))
        if document_id:
            return document_id
    return str(edge.get("target_document_id") or "")


def _target_parts(edge: Mapping[str, object]) -> tuple[str | None, str | None, str | None]:
    if _is_unresolved_inverse_edge(edge):
        _, article, clause, point = _node_legal_unit_parts(str(edge.get("source_id") or ""))
        return article, clause, point

    article = edge.get("target_article")
    clause = edge.get("target_clause")
    point = edge.get("target_point")
    if article:
        return str(article), str(clause) if clause else None, str(point) if point else None

    target_id = str(edge.get("target_id") or "")
    parts = target_id.split(":")
    if len(parts) >= 3 and parts[0] in {"article", "clause", "point"}:
        article = parts[2]
        clause = parts[3] if len(parts) >= 4 else None
        point = parts[4] if len(parts) >= 5 else None
        return article, clause, point
    return None, None, None


def build_legal_unit_indexes(
    chunks: Sequence[Mapping[str, object]],
) -> tuple[set[str], set[tuple[str, str]], set[tuple[str, str, str]], set[tuple[str, str, str, str]]]:
    documents: set[str] = set()
    articles: set[tuple[str, str]] = set()
    clauses: set[tuple[str, str, str]] = set()
    points: set[tuple[str, str, str, str]] = set()

    for chunk in chunks:
        document_id = str(chunk.get("document_id") or "")
        if document_id:
            documents.add(document_id)
        for node_id in legal_unit_node_ids_for_payload(dict(chunk)):
            parts = node_id.split(":")
            if parts[0] == "article" and len(parts) >= 3:
                articles.add((parts[1], parts[2]))
            elif parts[0] == "clause" and len(parts) >= 4:
                articles.add((parts[1], parts[2]))
                clauses.add((parts[1], parts[2], parts[3]))
            elif parts[0] == "point" and len(parts) >= 5:
                articles.add((parts[1], parts[2]))
                clauses.add((parts[1], parts[2], parts[3]))
                points.add((parts[1], parts[2], parts[3], parts[4]))
    return documents, articles, clauses, points


def classify_unresolved_reason(
    edge: Mapping[str, object],
    *,
    documents: set[str],
    articles: set[tuple[str, str]],
    clauses: set[tuple[str, str, str]],
    points: set[tuple[str, str, str, str]],
) -> str:
    target_document_id = _attempted_target_document_id(edge)
    article, clause, point = _target_parts(edge)
    matched_text = " ".join(
        str(edge.get(field) or "") for field in ("original_matched_text", "normalized_matched_text")
    ).lower()

    if target_document_id.startswith("external-") or target_document_id not in documents:
        return "target document outside current corpus"
    if any(marker in matched_text for marker in ("phu luc", "mau so", "bang ")):
        return "appendix/form/table reference"
    if not article and str(edge.get("reference_level") or "") != "document":
        return "parser failed to resolve article number"
    if article and (target_document_id, article) not in articles:
        return "target article does not exist"
    if clause and (target_document_id, article or "", clause) not in clauses:
        return "target clause does not exist"
    if point and (target_document_id, article or "", clause or "", point) not in points:
        return "target point does not exist"
    if "luat nay" in matched_text:
        return "ambiguous document alias"
    return "parser failed to resolve document"


def recommended_action(reason: str) -> str:
    actions = {
        "target document outside current corpus": "acceptable_external_reference",
        "target article does not exist": "Verify whether the target article is outside the curated subset; add source text only if needed for retrieval scope.",
        "target clause does not exist": "Fall back to article-level retrieval or add clause-level source if legally important.",
        "target point does not exist": "Fall back to clause/article-level retrieval or split point-level source if legally important.",
        "ambiguous document alias": "Inspect source context and add a document alias/context rule if it should map to a corpus document.",
        "appendix/form/table reference": "Resolve against appendix/table nodes after appendix graph nodes are modeled.",
        "parser failed to resolve document": "Add a parser pattern or document alias for this reference form.",
        "parser failed to resolve article number": "Add a parser pattern for this article-number format.",
    }
    return actions.get(reason, "Review manually.")


def _is_scope_article_1_source(edge: Mapping[str, object], document_id: str) -> bool:
    if str(edge.get("source_document_id") or "") != document_id:
        return False
    source_id = str(edge.get("source_id") or "")
    return (
        source_id.startswith(f"article:{document_id}:1")
        or source_id.startswith(f"clause:{document_id}:1:")
        or source_id.startswith(f"article:{document_id}:dieu-1")
        or source_id.startswith(f"clause:{document_id}:dieu-1:")
    )


def _is_critical_wrong_target(edge: Mapping[str, object]) -> bool:
    if str(edge.get("extraction_method") or "").startswith("inverse:"):
        return False
    target_document_id = str(edge.get("target_document_id") or "")
    source_document_id = str(edge.get("source_document_id") or "")
    article, _, _ = _target_parts(edge)
    if not article:
        return False

    if (
        source_document_id == "thong-tu-09-2020-tt-bldtbxh"
        and target_document_id == source_document_id
        and article in {"143", "145", "146", "147"}
    ):
        return True
    if (
        source_document_id == "thong-tu-10-2020-tt-bldtbxh"
        and target_document_id == source_document_id
        and article in {"21", "73", "142"}
    ):
        return True
    if (
        target_document_id == "nghi-dinh-145-2020-nd-cp"
        and _is_scope_article_1_source(edge, "nghi-dinh-145-2020-nd-cp")
    ):
        return True
    if (
        source_document_id == "nghi-dinh-135-2020-nd-cp"
        and target_document_id == source_document_id
        and article == "169"
    ):
        return True
    return False


def _wrong_target_record(edge: Mapping[str, object]) -> dict[str, object]:
    article, clause, point = _target_parts(edge)
    return {
        "edge_id": edge.get("edge_id"),
        "source_chunk_id": edge.get("source_chunk_id"),
        "source_document_id": edge.get("source_document_id"),
        "citation_text": edge.get("citation_text"),
        "original_matched_text": edge.get("original_matched_text"),
        "normalized_matched_text": edge.get("normalized_matched_text"),
        "attempted_target_document_id": _attempted_target_document_id(edge),
        "attempted_target_article": article,
        "attempted_target_clause": clause,
        "attempted_target_point": point,
        "unresolved_reason": "wrong target document for critical guiding reference",
        "recommended_action": "Retarget this guiding-document reference to 45-2019-qh14.",
    }


def build_unresolved_reference_report(
    edges: Sequence[Mapping[str, object]],
    chunks: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    documents, articles, clauses, points = build_legal_unit_indexes(chunks)
    unresolved_records: list[dict[str, object]] = []
    reason_counter: Counter[str] = Counter()
    critical_records: list[dict[str, object]] = []
    wrong_target_records: list[dict[str, object]] = []

    for edge in edges:
        if _is_critical_wrong_target(edge):
            wrong_target_records.append(_wrong_target_record(edge))
        if edge.get("resolved") is not False:
            continue
        article, clause, point = _target_parts(edge)
        reason = classify_unresolved_reason(
            edge,
            documents=documents,
            articles=articles,
            clauses=clauses,
            points=points,
        )
        reason_counter[reason] += 1
        record = {
            "edge_id": edge.get("edge_id"),
            "source_chunk_id": edge.get("source_chunk_id"),
            "source_document_id": edge.get("source_document_id"),
            "citation_text": edge.get("citation_text"),
            "original_matched_text": edge.get("original_matched_text"),
            "normalized_matched_text": edge.get("normalized_matched_text"),
            "attempted_target_document_id": edge.get("target_document_id"),
            "attempted_target_article": article,
            "attempted_target_clause": clause,
            "attempted_target_point": point,
            "unresolved_reason": reason,
            "recommended_action": recommended_action(reason),
        }
        unresolved_records.append(record)
        if article and (_attempted_target_document_id(edge), article) in CRITICAL_REFERENCES:
            critical_records.append(record)

    critical_reference_issues = [*critical_records, *wrong_target_records]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_unresolved_edges": len(unresolved_records),
        "critical_unresolved_references": len(critical_reference_issues),
        "critical_unresolved_edge_count": len(critical_records),
        "critical_wrong_target_references": len(wrong_target_records),
        "unresolved_by_reason": dict(sorted(reason_counter.items())),
        "critical_references": critical_reference_issues,
        "critical_wrong_target_edges": wrong_target_records,
        "unresolved_edges": unresolved_records,
    }


def render_unresolved_reference_report_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Unresolved Reference Edges Report",
        "",
        f"- Total unresolved edges: {report['total_unresolved_edges']}",
        f"- Critical unresolved references: {report['critical_unresolved_references']}",
        "",
        "## Unresolved By Reason",
        "",
        "| Reason | Edges |",
        "| --- | ---: |",
    ]
    reasons = report.get("unresolved_by_reason")
    if isinstance(reasons, Mapping):
        for reason, count in reasons.items():
            lines.append(f"| {reason} | {count} |")

    critical = report.get("critical_references")
    if isinstance(critical, Sequence) and critical:
        lines.extend(["", "## Critical Unresolved References", ""])
        for item in critical:
            if isinstance(item, Mapping):
                lines.append(
                    f"- `{item.get('source_chunk_id')}` -> `{item.get('attempted_target_document_id')}` "
                    f"Điều {item.get('attempted_target_article')}: {item.get('unresolved_reason')}"
                )

    lines.extend(
        [
            "",
            "## Unresolved Edges",
            "",
            "| Source chunk | Target | Reason | Recommended action |",
            "| --- | --- | --- | --- |",
        ]
    )
    records = report.get("unresolved_edges")
    if isinstance(records, Sequence):
        for item in records:
            if not isinstance(item, Mapping):
                continue
            target = (
                f"{item.get('attempted_target_document_id')} "
                f"Điều {item.get('attempted_target_article') or '-'}"
            )
            if item.get("attempted_target_clause"):
                target += f" khoản {item.get('attempted_target_clause')}"
            if item.get("attempted_target_point"):
                target += f" điểm {item.get('attempted_target_point')}"
            lines.append(
                f"| `{item.get('source_chunk_id')}` | {target} | "
                f"{item.get('unresolved_reason')} | {item.get('recommended_action')} |"
            )
    return "\n".join(lines).strip() + "\n"


def write_unresolved_reference_report(
    edges_path: Path,
    chunks_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    edges = read_jsonl(edges_path)
    chunks = read_jsonl(chunks_path)
    report = build_unresolved_reference_report(edges, chunks)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "unresolved_reference_edges_report.json"
    markdown_path = output_dir / "unresolved_reference_edges_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_unresolved_reference_report_markdown(report), encoding="utf-8")
    return json_path, markdown_path


__all__ = [
    "CRITICAL_REFERENCES",
    "build_unresolved_reference_report",
    "classify_unresolved_reason",
    "render_unresolved_reference_report_markdown",
    "write_unresolved_reference_report",
]
