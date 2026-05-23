from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
import unicodedata

from ...corpus_pipeline import extract_chunk_body, normalize_for_matching
from ..retrieval.models import RetrievedRecord
from .models import LegalGraphEdge
from .ontology import EdgeType
from .structural_parser import (
    article_node_id,
    clause_node_id,
    document_node_id,
    most_specific_legal_unit_node_id,
    point_node_id,
    record_to_payload,
)


ARTICLE_REFERENCE_RE = re.compile(
    r"(?P<prefix>theo quy dinh tai|tru truong hop quy dinh tai|quy dinh tai)?\s*"
    r"(?:(?:diem)\s+(?P<point>[a-z](?:\.\d+)?)\s+)?"
    r"(?:(?:khoan)\s+(?P<clause>\d+)\s+)?"
    r"(?:dieu)\s+(?P<article>\d+[a-z]?)",
    re.IGNORECASE,
)
DECREE_REFERENCE_RE = re.compile(
    r"nghi dinh\s+(?P<number>\d+)\s*/\s*(?P<year>\d{4})\s*/\s*nd\s*-?\s*cp",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedReference:
    target_node_id: str
    reference_type: str
    normalized_matched_text: str
    normalized_source_span: tuple[int, int]
    original_matched_text: str = ""
    raw_source_span: tuple[int, int] | None = None
    edge_type: EdgeType = EdgeType.REFERENCES


@dataclass(frozen=True)
class NormalizedText:
    text: str
    raw_offsets: tuple[int, ...]


def normalize_with_raw_offsets(value: str) -> NormalizedText:
    normalized_chars: list[str] = []
    raw_offsets: list[int] = []
    previous_was_space = True
    for raw_index, raw_char in enumerate(value):
        current = raw_char.replace("đ", "d").replace("Đ", "D")
        current = unicodedata.normalize("NFKD", current)
        current = "".join(ch for ch in current if not unicodedata.combining(ch)).lower()
        for char in current:
            if char.isspace():
                if previous_was_space:
                    continue
                normalized_chars.append(" ")
                raw_offsets.append(raw_index)
                previous_was_space = True
                continue
            normalized_chars.append(char)
            raw_offsets.append(raw_index)
            previous_was_space = False

    if normalized_chars and normalized_chars[-1] == " ":
        normalized_chars.pop()
        raw_offsets.pop()
    return NormalizedText(text="".join(normalized_chars), raw_offsets=tuple(raw_offsets))


def _raw_span_for_normalized_span(
    normalized: NormalizedText,
    span: tuple[int, int],
) -> tuple[int, int] | None:
    start, end = span
    if start < 0 or end <= start or end > len(normalized.raw_offsets):
        return None
    raw_start = normalized.raw_offsets[start]
    raw_end = normalized.raw_offsets[end - 1] + 1
    return raw_start, raw_end


def _target_document_id(normalized_text: str, source_document_id: str) -> str:
    if "bo luat lao dong" in normalized_text or source_document_id.startswith("nghi-dinh"):
        return "45-2019-qh14"
    return source_document_id


def parse_cross_references(
    record: RetrievedRecord | dict[str, Any],
) -> tuple[ParsedReference, ...]:
    payload = record_to_payload(record)
    source_document_id = str(payload.get("document_id") or "").strip()
    text = extract_chunk_body(
        str(payload.get("text") or ""),
        str(payload.get("heading") or ""),
    )
    normalized = normalize_with_raw_offsets(text)
    normalized_text = normalized.text
    references: list[ParsedReference] = []

    for match in ARTICLE_REFERENCE_RE.finditer(normalized_text):
        article = str(match.group("article") or "").strip()
        clause = str(match.group("clause") or "").strip()
        point = str(match.group("point") or "").strip()
        if not article:
            continue
        target_document_id = _target_document_id(normalized_text, source_document_id)
        if point and clause:
            target_id = point_node_id(target_document_id, article, clause, point)
            reference_type = "point"
        elif clause:
            target_id = clause_node_id(target_document_id, article, clause)
            reference_type = "clause"
        else:
            target_id = article_node_id(target_document_id, article)
            reference_type = "article"
        normalized_span = (match.start(), match.end())
        raw_span = _raw_span_for_normalized_span(normalized, normalized_span)
        references.append(
            ParsedReference(
                target_node_id=target_id,
                reference_type=reference_type,
                normalized_matched_text=match.group(0).strip(),
                normalized_source_span=normalized_span,
                original_matched_text=text[raw_span[0] : raw_span[1]].strip() if raw_span else "",
                raw_source_span=raw_span,
            )
        )

    for match in DECREE_REFERENCE_RE.finditer(normalized_text):
        number = str(match.group("number") or "")
        year = str(match.group("year") or "")
        target_document_id = f"nghi-dinh-{number}-{year}-nd-cp"
        if source_document_id == target_document_id:
            continue
        normalized_span = (match.start(), match.end())
        raw_span = _raw_span_for_normalized_span(normalized, normalized_span)
        references.append(
            ParsedReference(
                target_node_id=document_node_id(target_document_id),
                reference_type="document",
                normalized_matched_text=match.group(0).strip(),
                normalized_source_span=normalized_span,
                original_matched_text=text[raw_span[0] : raw_span[1]].strip() if raw_span else "",
                raw_source_span=raw_span,
                edge_type=EdgeType.GUIDED_BY,
            )
        )

    return tuple(references)


def build_reference_edges(
    record: RetrievedRecord | dict[str, Any],
) -> tuple[LegalGraphEdge, ...]:
    payload = record_to_payload(record)
    source_id = most_specific_legal_unit_node_id(payload)
    source_chunk_id = str(payload.get("chunk_id") or "")
    citation_text = str(payload.get("citation_text") or "")
    source_document_id = str(payload.get("document_id") or "")
    if not source_id or not source_chunk_id:
        return ()

    edges: list[LegalGraphEdge] = []
    for reference in parse_cross_references(payload):
        if reference.target_node_id == source_id:
            continue
        edge_id = (
            f"{source_id}|{reference.edge_type.value}|{reference.target_node_id}|"
            f"{source_chunk_id}|{reference.normalized_source_span[0]}-"
            f"{reference.normalized_source_span[1]}"
        )
        properties: dict[str, Any] = {
            "normalized_matched_text": reference.normalized_matched_text,
            "normalized_source_span": list(reference.normalized_source_span),
            "reference_type": reference.reference_type,
            "citation_text": citation_text,
            "confidence": 1.0,
            "extraction_method": "regex",
        }
        if reference.original_matched_text:
            properties["original_matched_text"] = reference.original_matched_text
        if reference.raw_source_span is not None:
            properties["raw_source_span"] = list(reference.raw_source_span)
        edges.append(
            LegalGraphEdge(
                edge_id=edge_id,
                source_id=source_id,
                target_id=reference.target_node_id,
                edge_type=reference.edge_type,
                confidence=1.0,
                source_chunk_id=source_chunk_id,
                extraction_method="regex",
                properties=properties,
            )
        )
    text = extract_chunk_body(
        str(payload.get("text") or ""),
        str(payload.get("heading") or ""),
    )
    normalized_text = normalize_for_matching(text)
    if source_document_id.startswith("nghi-dinh") and "bo luat lao dong" in normalized_text:
        labor_code_document_id = "45-2019-qh14"
        edges.append(
            LegalGraphEdge(
                edge_id=(
                    f"{document_node_id(labor_code_document_id)}|{EdgeType.GUIDED_BY.value}|"
                    f"{document_node_id(source_document_id)}|{source_chunk_id}"
                ),
                source_id=document_node_id(labor_code_document_id),
                target_id=document_node_id(source_document_id),
                edge_type=EdgeType.GUIDED_BY,
                confidence=1.0,
                source_chunk_id=source_chunk_id,
                extraction_method="regex",
                properties={
                    "normalized_matched_text": "bo luat lao dong",
                    "normalized_source_span": [],
                    "reference_type": "document_guidance",
                    "citation_text": citation_text,
                    "confidence": 1.0,
                    "extraction_method": "regex",
                },
            )
        )
    return tuple(edges)


__all__ = [
    "NormalizedText",
    "ParsedReference",
    "build_reference_edges",
    "normalize_with_raw_offsets",
    "parse_cross_references",
]
