from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from .corpus_pipeline import extract_chunk_body, normalize_for_matching
from .legal_chunk_enrichment import DOCUMENT_ALIASES, DOCUMENT_METADATA_BY_TYPE
from .rag.graph.structural_parser import (
    article_node_id,
    clause_node_id,
    document_node_id,
    evidence_chunk_node_id,
    legal_unit_node_ids_for_payload,
    most_specific_legal_unit_node_id,
    point_node_id,
)


ARTICLE_REFERENCE_RE = re.compile(
    r"(?:(?:theo|tai|can cu|quy dinh tai|duoc quy dinh tai|theo quy dinh tai)\s+)*"
    r"(?:(?:diem)\s+(?P<point>[a-z](?:\.\d+)?)\s+)?"
    r"(?:(?:khoan)\s+(?P<clause>\d+)\s+)?"
    r"(?:dieu)\s+(?P<article>\d+[a-z]?)"
    r"(?:\s+(?:cua|tai|thuoc|boi)?\s*(?P<document>bo luat lao dong(?: nam 2019)?|luat lao dong 2019|bo luat to tung dan su|blttds|bo luat to tung hinh su|bo luat dan su|luat bao hiem xa hoi|nghi dinh nay|thong tu nay|luat nay))?",
    re.IGNORECASE,
)
MULTI_CLAUSE_REFERENCE_RE = re.compile(
    r"(?:(?:theo|tai|can cu|quy dinh tai|duoc quy dinh tai|theo quy dinh tai)\s+)*"
    r"(?:khoan)\s+(?P<clauses>\d+(?:\s*,\s*\d+)*(?:\s+va\s+\d+)?)\s+"
    r"(?:dieu)\s+(?P<article>\d+[a-z]?)"
    r"(?:\s+(?:cua|tai|thuoc|boi)?\s*(?P<document>bo luat lao dong(?: nam 2019)?|luat lao dong 2019|bo luat to tung dan su|blttds|bo luat to tung hinh su|bo luat dan su|luat bao hiem xa hoi|nghi dinh nay|thong tu nay|luat nay))?",
    re.IGNORECASE,
)
DOCUMENT_REFERENCE_RE = re.compile(
    r"(?P<kind>nghi dinh|thong tu)\s+"
    r"(?P<number>0?\d{1,3})\s*/\s*(?P<year>\d{4})"
    r"(?:\s*/\s*(?P<suffix>nd\s*-?\s*cp|tt\s*-\s*bldtbxh))?",
    re.IGNORECASE,
)
EXTERNAL_DOCUMENT_ALIASES: dict[str, str] = {
    "bo luat to tung hinh su": "external-bo-luat-to-tung-hinh-su",
    "bo luat dan su": "external-bo-luat-dan-su",
    "luat bao hiem xa hoi": "external-luat-bao-hiem-xa-hoi",
}
GUIDING_SCOPE_DOCUMENT_IDS = {
    "nghi-dinh-145-2020-nd-cp",
    "nghi-dinh-135-2020-nd-cp",
    "thong-tu-09-2020-tt-bldtbxh",
    "thong-tu-10-2020-tt-bldtbxh",
}
LABOR_CODE_DOCUMENT_ID = "45-2019-qh14"
CURRENT_DECREE_ALIAS = "nghi dinh nay"
CURRENT_CIRCULAR_ALIAS = "thong tu nay"
CURRENT_LAW_ALIAS = "luat nay"
LOCAL_DOCUMENT_ALIAS_ORDER = tuple(
    sorted(
        [
            *DOCUMENT_ALIASES.keys(),
            *EXTERNAL_DOCUMENT_ALIASES.keys(),
            CURRENT_DECREE_ALIAS,
            CURRENT_CIRCULAR_ALIAS,
            CURRENT_LAW_ALIAS,
        ],
        key=lambda value: len(normalize_for_matching(value)),
        reverse=True,
    )
)


@dataclass(frozen=True)
class ParsedLegalReference:
    target_document_id: str
    target_id: str
    reference_level: str
    article_number: str | None
    clause_ref: str | None
    point_ref: str | None
    original_matched_text: str
    normalized_matched_text: str
    extraction_method: str
    confidence: float


def _repair_common_mojibake(text: str) -> str:
    replacements = {
        "Äiá»u": "Điều",
        "Ä‘iá»u": "điều",
        "khoáº£n": "khoản",
        "Khoáº£n": "Khoản",
        "Ä‘iá»ƒm": "điểm",
        "Äiá»ƒm": "Điểm",
        "Bá»™ luáº­t Lao Ä‘á»™ng": "Bộ luật Lao động",
        "Bá»™ Luáº­t Lao Äá»™ng": "Bộ luật Lao động",
        "Bá»™ luáº­t Tá»‘ tá»¥ng dÃ¢n sá»±": "Bộ luật Tố tụng dân sự",
        "Nghá»‹ Ä‘á»‹nh": "Nghị định",
        "ThÃ´ng tÆ°": "Thông tư",
        "NÄ-CP": "NĐ-CP",
        "BLÄTBXH": "BLĐTBXH",
    }
    repaired = text
    for bad, good in replacements.items():
        repaired = repaired.replace(bad, good)
    return repaired


def _text_for_matching(text: str) -> str:
    return normalize_for_matching(_repair_common_mojibake(text))


def _retrieval_body(text: str) -> str:
    normalized = _text_for_matching(text)
    marker = "quy dinh:"
    marker_index = normalized.find(marker)
    if marker_index < 0:
        return text

    # Work on normalized text when the source may be mojibake; preserving exact raw offsets
    # is less important here than avoiding citation-prefix false positives.
    return normalized[marker_index + len(marker) :].strip()


def _resolve_document_alias(normalized_text: str) -> str | None:
    for alias, document_id in DOCUMENT_ALIASES.items():
        if normalize_for_matching(alias) in normalized_text:
            return document_id
    for alias, document_id in EXTERNAL_DOCUMENT_ALIASES.items():
        if normalize_for_matching(alias) in normalized_text:
            return document_id
    return None


def _resolve_relative_document_alias(
    normalized_text: str,
    source_chunk: Mapping[str, object],
) -> str | None:
    source_document_id = str(source_chunk.get("document_id") or "")
    source_article_number = str(source_chunk.get("article_number") or "")
    if normalized_text == CURRENT_DECREE_ALIAS and source_document_id.startswith("nghi-dinh"):
        return source_document_id
    if normalized_text == CURRENT_CIRCULAR_ALIAS and source_document_id.startswith("thong-tu"):
        return source_document_id
    if normalized_text == CURRENT_LAW_ALIAS:
        if source_article_number == "219":
            return "external-quoted-amended-law"
        return source_document_id or None
    return None


def _resolve_document_text(
    document_text: str,
    source_chunk: Mapping[str, object],
) -> str | None:
    normalized_document_text = normalize_for_matching(document_text)
    if not normalized_document_text:
        return None
    return _resolve_relative_document_alias(
        normalized_document_text,
        source_chunk,
    ) or _resolve_document_alias(normalized_document_text)


def _reference_suffix_segment(normalized_text: str, end: int) -> str:
    suffix = normalized_text[end : end + 700]
    period_index = suffix.find(".")
    if period_index >= 0:
        suffix = suffix[:period_index]
    return suffix


def _resolve_first_local_document_alias(
    normalized_text: str,
    source_chunk: Mapping[str, object],
) -> str | None:
    candidates: list[tuple[int, int, str]] = []
    for alias in LOCAL_DOCUMENT_ALIAS_ORDER:
        normalized_alias = normalize_for_matching(alias)
        position = normalized_text.find(normalized_alias)
        if position < 0:
            continue
        target_document_id = _resolve_relative_document_alias(
            normalized_alias,
            source_chunk,
        ) or _resolve_document_alias(normalized_alias)
        if target_document_id:
            candidates.append((position, -len(normalized_alias), target_document_id))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def _resolve_article_reference_document(
    document_text: str,
    normalized_text: str,
    start: int,
    end: int,
    source_chunk: Mapping[str, object],
) -> str | None:
    resolved = _resolve_document_text(document_text, source_chunk)
    if resolved:
        return resolved
    suffix_resolved = _resolve_first_local_document_alias(
        _reference_suffix_segment(normalized_text, end),
        source_chunk,
    )
    if suffix_resolved:
        return suffix_resolved
    if _is_guiding_scope_chunk(source_chunk):
        return LABOR_CODE_DOCUMENT_ID
    return None


def _is_guiding_scope_chunk(source_chunk: Mapping[str, object]) -> bool:
    return (
        str(source_chunk.get("document_id") or "") in GUIDING_SCOPE_DOCUMENT_IDS
        and str(source_chunk.get("article_number") or "") == "1"
    )


def _reference_context(normalized_text: str, start: int, end: int) -> str:
    return normalized_text[max(0, start - 300) : end + 300]


def _inside_span(span: tuple[int, int], outer_spans: Sequence[tuple[int, int]]) -> bool:
    return any(start <= span[0] and span[1] <= end for start, end in outer_spans)


def _parse_clause_numbers(value: str) -> list[str]:
    return re.findall(r"\d+", value)


def _is_preceded_by_point_marker(normalized_text: str, start: int) -> bool:
    prefix = normalized_text[max(0, start - 24) : start]
    return re.search(r"diem\s+[a-z](?:\.\d+)?\s+$", prefix) is not None


def _resolve_document_reference(match: re.Match[str]) -> str | None:
    kind = match.group("kind")
    number = str(match.group("number") or "").lstrip("0") or "0"
    year = str(match.group("year") or "")
    if kind == "nghi dinh":
        return f"nghi-dinh-{number}-{year}-nd-cp"
    if kind == "thong tu":
        return f"thong-tu-{int(number):02d}-{year}-tt-bldtbxh"
    return None


def _target_node_id(
    document_id: str,
    article: str | None,
    clause: str | None,
    point: str | None,
) -> tuple[str, str]:
    if article and clause and point:
        return point_node_id(document_id, article, clause, point), "point"
    if article and clause:
        return clause_node_id(document_id, article, clause), "clause"
    if article:
        return article_node_id(document_id, article), "article"
    return document_node_id(document_id), "document"


def parse_legal_references(chunk: Mapping[str, object]) -> list[ParsedLegalReference]:
    source_document_id = str(chunk.get("document_id") or "")
    source_text_parts = [
        extract_chunk_body(str(chunk.get("text") or ""), str(chunk.get("heading") or "")),
        _retrieval_body(str(chunk.get("retrieval_text") or "")),
    ]
    references: list[ParsedLegalReference] = []

    for raw_text in source_text_parts:
        normalized = _text_for_matching(raw_text)
        multi_clause_spans: list[tuple[int, int]] = []
        for match in MULTI_CLAUSE_REFERENCE_RE.finditer(normalized):
            if _is_preceded_by_point_marker(normalized, match.start()):
                continue
            multi_clause_spans.append((match.start(), match.end()))
            article = str(match.group("article") or "")
            point = None
            document_text = str(match.group("document") or "")
            target_document_id = (
                _resolve_article_reference_document(
                    document_text,
                    normalized,
                    match.start(),
                    match.end(),
                    chunk,
                )
                or source_document_id
            )
            matched_text = match.group(0).strip()
            for clause in _parse_clause_numbers(str(match.group("clauses") or "")):
                target_id, reference_level = _target_node_id(
                    target_document_id,
                    article,
                    clause,
                    point,
                )
                references.append(
                    ParsedLegalReference(
                        target_document_id=target_document_id,
                        target_id=target_id,
                        reference_level=reference_level,
                        article_number=article,
                        clause_ref=clause,
                        point_ref=point,
                        original_matched_text=matched_text,
                        normalized_matched_text=f"{matched_text}#khoan-{clause}",
                        extraction_method="regex:multi_clause_reference",
                        confidence=0.92 if target_document_id != source_document_id else 0.82,
                    )
                )

        for match in ARTICLE_REFERENCE_RE.finditer(normalized):
            if _inside_span((match.start(), match.end()), multi_clause_spans):
                continue
            article = str(match.group("article") or "")
            clause = str(match.group("clause") or "") or None
            point = str(match.group("point") or "") or None
            document_text = str(match.group("document") or "")
            target_document_id = (
                _resolve_article_reference_document(
                    document_text,
                    normalized,
                    match.start(),
                    match.end(),
                    chunk,
                )
                or source_document_id
            )
            target_id, reference_level = _target_node_id(target_document_id, article, clause, point)
            matched_text = match.group(0).strip()
            references.append(
                ParsedLegalReference(
                    target_document_id=target_document_id,
                    target_id=target_id,
                    reference_level=reference_level,
                    article_number=article,
                    clause_ref=clause,
                    point_ref=point,
                    original_matched_text=matched_text,
                    normalized_matched_text=matched_text,
                    extraction_method="regex:article_reference",
                    confidence=0.92 if target_document_id != source_document_id else 0.82,
                )
            )

        for match in DOCUMENT_REFERENCE_RE.finditer(normalized):
            target_document_id = _resolve_document_reference(match)
            if not target_document_id:
                continue
            matched_text = match.group(0).strip()
            references.append(
                ParsedLegalReference(
                    target_document_id=target_document_id,
                    target_id=document_node_id(target_document_id),
                    reference_level="document",
                    article_number=None,
                    clause_ref=None,
                    point_ref=None,
                    original_matched_text=matched_text,
                    normalized_matched_text=matched_text,
                    extraction_method="regex:document_reference",
                    confidence=0.95,
                )
            )
    return references


def _source_node_id(chunk: Mapping[str, object]) -> str:
    return most_specific_legal_unit_node_id(dict(chunk)) or evidence_chunk_node_id(
        str(chunk.get("chunk_id") or "")
    )


def _rank(document_type: object) -> int | None:
    metadata = DOCUMENT_METADATA_BY_TYPE.get(str(document_type or ""))
    return int(metadata["normative_rank"]) if metadata and metadata.get("normative_rank") else None


def _edge_type_for_reference(
    source_chunk: Mapping[str, object],
    reference: ParsedLegalReference,
    inverse: bool = False,
) -> str:
    source_rank = _rank(source_chunk.get("document_type"))
    target_rank = _rank(_document_type_from_id(reference.target_document_id))
    source_document_id = str(source_chunk.get("document_id") or "")

    if inverse:
        return "GUIDED_BY"
    if source_rank and target_rank and source_rank > target_rank:
        return "DETAILS"
    if source_rank and target_rank and source_rank < target_rank:
        return "GUIDED_BY"
    if source_document_id != reference.target_document_id and reference.reference_level == "document":
        return "REFERENCES"
    return "REFERENCES"


def _document_type_from_id(document_id: str) -> str:
    if document_id in {"45-2019-qh14", "92-2015-qh13-labor-only"}:
        return "bo_luat"
    if document_id.startswith("nghi-dinh"):
        return "nghi_dinh"
    if document_id.startswith("thong-tu"):
        return "thong_tu"
    return ""


def _edge_id(parts: Sequence[object]) -> str:
    raw = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"ref_edge:{digest}"


def _edge_record(
    *,
    source_id: str,
    target_id: str,
    edge_type: str,
    source_chunk: Mapping[str, object],
    target_document_id: str,
    reference: ParsedLegalReference,
    resolved: bool,
) -> dict[str, object]:
    source_chunk_id = str(source_chunk.get("chunk_id") or "")
    source_document_id = str(source_chunk.get("document_id") or "")
    return {
        "edge_id": _edge_id(
            [
                source_id,
                edge_type,
                target_id,
                source_chunk_id,
                reference.normalized_matched_text,
            ]
        ),
        "source_id": source_id,
        "target_id": target_id,
        "edge_type": edge_type,
        "source_chunk_id": source_chunk_id,
        "source_document_id": source_document_id,
        "target_document_id": target_document_id,
        "citation_text": str(source_chunk.get("citation_text") or ""),
        "original_matched_text": reference.original_matched_text,
        "normalized_matched_text": reference.normalized_matched_text,
        "extraction_method": reference.extraction_method,
        "confidence": reference.confidence,
        "resolved": resolved,
        "target_article": reference.article_number,
        "target_clause": reference.clause_ref,
        "target_point": reference.point_ref,
        "reference_level": reference.reference_level,
    }


def build_node_id_index(chunks: Sequence[Mapping[str, object]]) -> set[str]:
    node_ids: set[str] = set()
    for chunk in chunks:
        document_id = str(chunk.get("document_id") or "")
        chunk_id = str(chunk.get("chunk_id") or "")
        if document_id:
            node_ids.add(document_node_id(document_id))
        if chunk_id:
            node_ids.add(evidence_chunk_node_id(chunk_id))
        node_ids.update(legal_unit_node_ids_for_payload(dict(chunk)))
    return node_ids


def build_reference_edge_records(
    chunks: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], int]:
    existing_node_ids = build_node_id_index(chunks)
    edges_by_key: dict[tuple[str, str, str, str], dict[str, object]] = {}
    duplicate_count = 0

    for chunk in chunks:
        source_id = _source_node_id(chunk)
        source_document_id = str(chunk.get("document_id") or "")
        if not source_id:
            continue

        for reference in parse_legal_references(chunk):
            if reference.target_id == source_id:
                continue
            edge_type = _edge_type_for_reference(chunk, reference)
            resolved = reference.target_id in existing_node_ids
            edge = _edge_record(
                source_id=source_id,
                target_id=reference.target_id,
                edge_type=edge_type,
                source_chunk=chunk,
                target_document_id=reference.target_document_id,
                reference=reference,
                resolved=resolved,
            )
            key = (
                str(edge["source_id"]),
                str(edge["target_id"]),
                str(edge["edge_type"]),
                str(edge["normalized_matched_text"]),
            )
            if key in edges_by_key:
                duplicate_count += 1
            else:
                edges_by_key[key] = edge

            if (
                edge_type == "DETAILS"
                and source_document_id != reference.target_document_id
                and reference.target_id != source_id
            ):
                inverse_resolved = source_id in existing_node_ids
                inverse_article = str(chunk.get("article_number") or "") or None
                inverse_clause = str(chunk.get("clause_ref") or "") or None
                inverse_point = str(chunk.get("point_ref") or "") or None
                inverse_reference_level = str(chunk.get("level") or "") or "chunk"
                inverse_reference = ParsedLegalReference(
                    target_document_id=source_document_id,
                    target_id=source_id,
                    reference_level=inverse_reference_level,
                    article_number=inverse_article,
                    clause_ref=inverse_clause,
                    point_ref=inverse_point,
                    original_matched_text=reference.original_matched_text,
                    normalized_matched_text=reference.normalized_matched_text,
                    extraction_method="inverse:guided_by",
                    confidence=reference.confidence,
                )
                inverse_edge = _edge_record(
                    source_id=reference.target_id,
                    target_id=source_id,
                    edge_type="GUIDED_BY",
                    source_chunk=chunk,
                    target_document_id=source_document_id,
                    reference=inverse_reference,
                    resolved=resolved and inverse_resolved,
                )
                inverse_key = (
                    str(inverse_edge["source_id"]),
                    str(inverse_edge["target_id"]),
                    str(inverse_edge["edge_type"]),
                    str(inverse_edge["normalized_matched_text"]),
                )
                if inverse_key in edges_by_key:
                    duplicate_count += 1
                else:
                    edges_by_key[inverse_key] = inverse_edge

    return list(edges_by_key.values()), duplicate_count


def _distribution(edges: Sequence[Mapping[str, object]], field: str) -> dict[str, int]:
    counter = Counter(str(edge.get(field) or "") for edge in edges if edge.get(field) not in (None, ""))
    return dict(sorted(counter.items()))


def _article_key(edge: Mapping[str, object]) -> str | None:
    target_id = str(edge.get("target_id") or "")
    parts = target_id.split(":")
    if len(parts) >= 3 and parts[0] in {"article", "clause", "point"}:
        return f"{parts[1]}:Điều {parts[2]}"
    return None


def summarize_reference_edges(
    edges: Sequence[Mapping[str, object]],
    duplicate_edges_removed: int,
) -> dict[str, object]:
    top_counter: Counter[str] = Counter()
    for edge in edges:
        article = _article_key(edge)
        if article:
            top_counter[article] += 1

    resolved_edges = [edge for edge in edges if edge.get("resolved") is True]
    unresolved_edges = [edge for edge in edges if edge.get("resolved") is False]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_reference_edges": len(edges),
        "edges_by_type": _distribution(edges, "edge_type"),
        "resolved_edges": len(resolved_edges),
        "unresolved_edges": len(unresolved_edges),
        "edges_by_source_document": _distribution(edges, "source_document_id"),
        "edges_by_target_document": _distribution(edges, "target_document_id"),
        "top_referenced_articles": dict(top_counter.most_common(25)),
        "duplicate_edges_removed": duplicate_edges_removed,
        "unresolved_edge_ids": [str(edge.get("edge_id") or "") for edge in unresolved_edges],
    }


def render_reference_edges_summary_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# Reference Edges Summary",
        "",
        f"- Total reference edges: {summary['total_reference_edges']}",
        f"- Resolved edges: {summary['resolved_edges']}",
        f"- Unresolved edges: {summary['unresolved_edges']}",
        f"- Duplicate edges removed: {summary['duplicate_edges_removed']}",
    ]
    tables = [
        ("Edges By Type", "edges_by_type", "Edge type"),
        ("Edges By Source Document", "edges_by_source_document", "Source document"),
        ("Edges By Target Document", "edges_by_target_document", "Target document"),
        ("Top Referenced Articles", "top_referenced_articles", "Article"),
    ]
    for title, key, label in tables:
        lines.extend(["", f"## {title}", "", f"| {label} | Edges |", "| --- | ---: |"])
        values = summary.get(key)
        if isinstance(values, Mapping):
            for item, count in values.items():
                lines.append(f"| {item} | {count} |")
    return "\n".join(lines).strip() + "\n"


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_reference_edge_artifacts(
    edges: Sequence[Mapping[str, object]],
    duplicate_edges_removed: int,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    edges_path = output_dir / "reference_edges.jsonl"
    summary_json_path = output_dir / "reference_edges_summary.json"
    summary_md_path = output_dir / "reference_edges_summary.md"

    with edges_path.open("w", encoding="utf-8", newline="\n") as handle:
        for edge in edges:
            handle.write(json.dumps(edge, ensure_ascii=False) + "\n")

    summary = summarize_reference_edges(edges, duplicate_edges_removed)
    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_md_path.write_text(
        render_reference_edges_summary_markdown(summary),
        encoding="utf-8",
    )
    return edges_path, summary_json_path, summary_md_path


__all__ = [
    "ParsedLegalReference",
    "build_node_id_index",
    "build_reference_edge_records",
    "parse_legal_references",
    "read_jsonl",
    "render_reference_edges_summary_markdown",
    "summarize_reference_edges",
    "write_reference_edge_artifacts",
]
