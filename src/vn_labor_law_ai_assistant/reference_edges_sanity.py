from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, Sequence

from .legal_reference_edges import _retrieval_body, _text_for_matching, read_jsonl


LABOR_CODE_DOCUMENT_ID = "45-2019-qh14"
GUIDING_DOCUMENT_IDS = {
    "nghi-dinh-145-2020-nd-cp",
    "nghi-dinh-135-2020-nd-cp",
    "thong-tu-09-2020-tt-bldtbxh",
    "thong-tu-10-2020-tt-bldtbxh",
}
IMPOSSIBLE_SELF_ARTICLES = {
    "thong-tu-09-2020-tt-bldtbxh": {"143", "145", "146", "147"},
    "thong-tu-10-2020-tt-bldtbxh": {"21", "73", "142"},
    "nghi-dinh-145-2020-nd-cp": {"122", "130", "161", "209", "210"},
    "nghi-dinh-135-2020-nd-cp": {"169", "219"},
}
FIXED_DOCUMENT_ALIASES = {
    "bo luat lao dong nam 2019": LABOR_CODE_DOCUMENT_ID,
    "bo luat lao dong": LABOR_CODE_DOCUMENT_ID,
    "luat lao dong 2019": LABOR_CODE_DOCUMENT_ID,
    "luat bao hiem xa hoi": "external-luat-bao-hiem-xa-hoi",
    "bo luat to tung hinh su": "external-bo-luat-to-tung-hinh-su",
    "bo luat dan su": "external-bo-luat-dan-su",
    "bo luat to tung dan su": "92-2015-qh13-labor-only",
    "blttds": "92-2015-qh13-labor-only",
}


def _is_inverse_edge(edge: Mapping[str, object]) -> bool:
    return str(edge.get("extraction_method") or "").startswith("inverse:")


def _base_match_text(edge: Mapping[str, object]) -> str:
    text = str(edge.get("normalized_matched_text") or edge.get("original_matched_text") or "")
    return text.split("#", 1)[0].strip()


def _candidate_match_texts(edge: Mapping[str, object]) -> list[str]:
    candidates = [_base_match_text(edge)]
    article = str(edge.get("target_article") or "")
    clause = str(edge.get("target_clause") or "")
    point = str(edge.get("target_point") or "")
    if article and clause and point:
        candidates.append(f"diem {point} khoan {clause} dieu {article}")
    if article and clause:
        candidates.append(f"khoan {clause} dieu {article}")
    if article:
        candidates.append(f"dieu {article}")
    deduped: list[str] = []
    for candidate in candidates:
        normalized = _text_for_matching(candidate)
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _match_context(edge: Mapping[str, object], chunk: Mapping[str, object]) -> dict[str, str]:
    source_text_parts = [
        str(chunk.get("text") or ""),
        _retrieval_body(str(chunk.get("retrieval_text") or "")),
    ]
    for raw_text in source_text_parts:
        normalized_text = _text_for_matching(raw_text)
        for match_text in _candidate_match_texts(edge):
            position = normalized_text.find(match_text)
            if position < 0:
                continue
            suffix = normalized_text[position + len(match_text) : position + len(match_text) + 700]
            period_index = suffix.find(".")
            if period_index >= 0:
                suffix = suffix[:period_index]
            return {
                "matched_text": match_text,
                "suffix": suffix,
                "context": normalized_text[
                    max(0, position - 160) : position + len(match_text) + 320
                ],
            }
    return {"matched_text": _text_for_matching(_base_match_text(edge)), "suffix": "", "context": ""}


def _alias_target(alias: str, source_document_id: str) -> str | None:
    if alias == "nghi dinh nay":
        return source_document_id if source_document_id.startswith("nghi-dinh") else None
    if alias == "thong tu nay":
        return source_document_id if source_document_id.startswith("thong-tu") else None
    return FIXED_DOCUMENT_ALIASES.get(alias)


def _first_alias_target(segment: str, source_document_id: str) -> str | None:
    candidates: list[tuple[int, int, str]] = []
    aliases = [*FIXED_DOCUMENT_ALIASES, "nghi dinh nay", "thong tu nay"]
    for alias in aliases:
        position = segment.find(alias)
        if position < 0:
            continue
        target = _alias_target(alias, source_document_id)
        if target:
            candidates.append((position, -len(alias), target))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def _expected_document_id(
    edge: Mapping[str, object],
    chunk: Mapping[str, object],
) -> tuple[str | None, str, str]:
    source_document_id = str(edge.get("source_document_id") or "")
    if source_document_id not in GUIDING_DOCUMENT_IDS:
        return None, "", ""

    context = _match_context(edge, chunk)
    matched_text = context["matched_text"]
    expected = _first_alias_target(matched_text, source_document_id)
    if expected:
        return expected, "explicit_document_in_match", context["context"]

    expected = _first_alias_target(context["suffix"], source_document_id)
    if expected:
        return expected, "explicit_document_in_reference_list", context["context"]

    if str(chunk.get("article_number") or "") == "1":
        return LABOR_CODE_DOCUMENT_ID, "guiding_scope_article_defaults_to_labor_code", context[
            "context"
        ]
    return None, "", context["context"]


def _target_ref_key(edge: Mapping[str, object]) -> tuple[str, str, str, str]:
    return (
        str(edge.get("source_chunk_id") or ""),
        str(edge.get("target_article") or ""),
        str(edge.get("target_clause") or ""),
        str(edge.get("target_point") or ""),
    )


def _edge_example(
    edge: Mapping[str, object],
    *,
    expected_document_id: str | None = None,
    reason: str = "",
    context: str = "",
) -> dict[str, object]:
    return {
        "edge_id": edge.get("edge_id"),
        "source_chunk_id": edge.get("source_chunk_id"),
        "source_document_id": edge.get("source_document_id"),
        "target_document_id": edge.get("target_document_id"),
        "expected_document_id": expected_document_id,
        "target_article": edge.get("target_article"),
        "target_clause": edge.get("target_clause"),
        "target_point": edge.get("target_point"),
        "original_matched_text": edge.get("original_matched_text"),
        "resolved": edge.get("resolved"),
        "reason": reason,
        "context": context[:500],
    }


def _dual_target_duplicate_examples(
    edges: Sequence[Mapping[str, object]],
) -> tuple[int, list[dict[str, object]]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for edge in edges:
        if _is_inverse_edge(edge):
            continue
        key = _target_ref_key(edge)
        if key[1]:
            grouped[key].append(edge)

    examples: list[dict[str, object]] = []
    duplicate_count = 0
    for key, group in grouped.items():
        source_document_id = str(group[0].get("source_document_id") or "")
        target_documents = {str(edge.get("target_document_id") or "") for edge in group}
        if (
            source_document_id
            and source_document_id != LABOR_CODE_DOCUMENT_ID
            and source_document_id in target_documents
            and LABOR_CODE_DOCUMENT_ID in target_documents
        ):
            duplicate_count += 1
            if len(examples) < 20:
                examples.append(
                    {
                        "source_chunk_id": key[0],
                        "target_article": key[1],
                        "target_clause": key[2] or None,
                        "target_point": key[3] or None,
                        "target_documents": sorted(target_documents),
                        "edge_ids": [edge.get("edge_id") for edge in group],
                    }
                )
    return duplicate_count, examples


def _load_unresolved_report(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_reference_edges_sanity_report(
    edges: Sequence[Mapping[str, object]],
    chunks: Sequence[Mapping[str, object]],
    unresolved_report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    chunk_by_id = {str(chunk.get("chunk_id") or ""): chunk for chunk in chunks}
    suspicious_resolved: list[dict[str, object]] = []
    suspicious_unresolved: list[dict[str, object]] = []

    for edge in edges:
        if _is_inverse_edge(edge):
            continue
        chunk = chunk_by_id.get(str(edge.get("source_chunk_id") or ""), {})
        source_document_id = str(edge.get("source_document_id") or "")
        target_document_id = str(edge.get("target_document_id") or "")
        target_article = str(edge.get("target_article") or "")

        expected, reason, context = _expected_document_id(edge, chunk)
        is_suspicious = bool(expected and target_document_id != expected)
        if (
            not is_suspicious
            and target_document_id == source_document_id
            and target_article in IMPOSSIBLE_SELF_ARTICLES.get(source_document_id, set())
        ):
            expected = LABOR_CODE_DOCUMENT_ID
            reason = "impossible_guiding_document_self_article"
            context = _match_context(edge, chunk).get("context", "")
            is_suspicious = True

        if not is_suspicious:
            continue
        record = _edge_example(
            edge,
            expected_document_id=expected,
            reason=reason,
            context=context,
        )
        if edge.get("resolved") is True:
            suspicious_resolved.append(record)
        else:
            suspicious_unresolved.append(record)

    dual_count, dual_examples = _dual_target_duplicate_examples(edges)
    unresolved_report = unresolved_report or {}
    unresolved_edges = unresolved_report.get("unresolved_edges")
    external_classification_issues = []
    if isinstance(unresolved_edges, Sequence):
        for item in unresolved_edges:
            if not isinstance(item, Mapping):
                continue
            target_document_id = str(item.get("attempted_target_document_id") or "")
            if target_document_id.startswith("external-") and (
                item.get("recommended_action") != "acceptable_external_reference"
            ):
                external_classification_issues.append(dict(item))

    passed = (
        len(suspicious_resolved) == 0
        and len(suspicious_unresolved) == 0
        and dual_count == 0
        and int(unresolved_report.get("critical_unresolved_references") or 0) == 0
        and len(external_classification_issues) == 0
    )
    recommended_action = (
        "No parser changes needed. Reference-edge sanity checks passed; keep vector index and "
        "Neo4j graph builds as separate next steps."
        if passed
        else "Fix parser target resolution, regenerate reference-edge artifacts, and rerun this sanity check before building the vector index or Neo4j graph."
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_reference_edges": len(edges),
        "resolved_edges": sum(1 for edge in edges if edge.get("resolved") is True),
        "unresolved_edges": sum(1 for edge in edges if edge.get("resolved") is False),
        "suspicious_resolved_edges_count": len(suspicious_resolved),
        "suspicious_unresolved_edges_count": len(suspicious_unresolved),
        "dual_target_duplicate_count": dual_count,
        "critical_unresolved_references": int(
            unresolved_report.get("critical_unresolved_references") or 0
        ),
        "external_reference_classification_issues_count": len(external_classification_issues),
        "external_acceptable_reference_count": sum(
            1
            for item in unresolved_edges or []
            if isinstance(item, Mapping)
            and str(item.get("attempted_target_document_id") or "").startswith("external-")
            and item.get("recommended_action") == "acceptable_external_reference"
        ),
        "passed": passed,
        "examples": {
            "suspicious_resolved_edges": suspicious_resolved[:20],
            "suspicious_unresolved_edges": suspicious_unresolved[:20],
            "dual_target_duplicates": dual_examples,
            "external_reference_classification_issues": external_classification_issues[:20],
        },
        "recommended_action": recommended_action,
    }


def render_reference_edges_sanity_markdown(report: Mapping[str, object]) -> str:
    examples = report.get("examples")
    lines = [
        "# Reference Edges Sanity Report",
        "",
        f"- Total reference edges: {report['total_reference_edges']}",
        f"- Resolved edges: {report['resolved_edges']}",
        f"- Unresolved edges: {report['unresolved_edges']}",
        f"- Suspicious resolved edges: {report['suspicious_resolved_edges_count']}",
        f"- Suspicious unresolved edges: {report['suspicious_unresolved_edges_count']}",
        f"- Dual-target duplicates: {report['dual_target_duplicate_count']}",
        f"- Critical unresolved references: {report['critical_unresolved_references']}",
        "- External classification issues: "
        f"{report['external_reference_classification_issues_count']}",
        "- Acceptable external references: "
        f"{report['external_acceptable_reference_count']}",
        f"- Passed: {report['passed']}",
        "",
        "## Recommended Action",
        "",
        str(report["recommended_action"]),
        "",
        "## Examples",
    ]
    if isinstance(examples, Mapping):
        for key in (
            "suspicious_resolved_edges",
            "suspicious_unresolved_edges",
            "dual_target_duplicates",
            "external_reference_classification_issues",
        ):
            values = examples.get(key)
            lines.extend(["", f"### {key}", ""])
            if isinstance(values, Sequence) and values:
                for item in values:
                    lines.append(f"- `{json.dumps(item, ensure_ascii=False)}`")
            else:
                lines.append("- None")
    return "\n".join(lines).strip() + "\n"


def write_reference_edges_sanity_report(
    edges_path: Path,
    chunks_path: Path,
    unresolved_report_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    report = build_reference_edges_sanity_report(
        read_jsonl(edges_path),
        read_jsonl(chunks_path),
        _load_unresolved_report(unresolved_report_path),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "reference_edges_sanity_report.json"
    markdown_path = output_dir / "reference_edges_sanity_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_reference_edges_sanity_markdown(report), encoding="utf-8")
    return json_path, markdown_path


__all__ = [
    "build_reference_edges_sanity_report",
    "render_reference_edges_sanity_markdown",
    "write_reference_edges_sanity_report",
]
