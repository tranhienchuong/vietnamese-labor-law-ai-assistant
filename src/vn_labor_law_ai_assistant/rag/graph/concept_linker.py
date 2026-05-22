from __future__ import annotations

from typing import Any

from ...corpus_pipeline import normalize_for_matching
from ..retrieval.models import RetrievedRecord
from .models import LegalGraphEdge, LegalGraphNode
from .ontology import EdgeType, NodeType
from .structural_parser import (
    concept_node_id,
    evidence_chunk_node_id,
    legal_unit_node_ids_for_payload,
    normalized_name,
    record_to_payload,
)


LEGAL_CONCEPTS: dict[str, tuple[str, ...]] = {
    "đơn phương chấm dứt hợp đồng lao động": (
        "đơn phương chấm dứt",
        "đơn phương chấm dứt hợp đồng",
        "đơn phương chấm dứt HĐLĐ",
        "nghỉ việc không báo trước",
    ),
    "thời hạn báo trước": ("báo trước", "thời hạn báo trước", "không cần báo trước"),
    "trợ cấp thôi việc": ("trợ cấp thôi việc", "tính trợ cấp thôi việc"),
    "trợ cấp mất việc làm": ("trợ cấp mất việc làm",),
    "tiền lương": (
        "tiền lương",
        "trả lương",
        "trả thiếu lương",
        "trả lương không đúng hạn",
    ),
    "kỷ luật sa thải": ("sa thải", "kỷ luật sa thải"),
    "hợp đồng lao động": ("hợp đồng lao động", "HĐLĐ"),
}

SUBJECTS: dict[str, tuple[str, ...]] = {
    "người lao động": ("người lao động", "NLĐ"),
    "người sử dụng lao động": (
        "người sử dụng lao động",
        "NSDLĐ",
        "công ty",
        "doanh nghiệp",
    ),
    "tổ chức đại diện người lao động": (
        "công đoàn",
        "tổ chức đại diện người lao động",
    ),
}

ACTIONS: dict[str, tuple[str, ...]] = {
    "đơn phương chấm dứt hợp đồng lao động": (
        "đơn phương chấm dứt hợp đồng lao động",
        "đơn phương chấm dứt HĐLĐ",
    ),
    "trả lương chậm": ("trả lương chậm", "trả lương không đúng hạn"),
    "tự ý bỏ việc": ("tự ý bỏ việc", "bỏ việc"),
}


def _contains_alias(text: str, aliases: tuple[str, ...]) -> bool:
    return any(normalize_for_matching(alias) in text for alias in aliases)


def _provenance_properties(
    *,
    citation_text: str,
    matched_aliases: tuple[str, ...],
    confidence: float = 0.85,
) -> dict[str, Any]:
    return {
        "citation_text": citation_text,
        "extraction_method": "dictionary",
        "confidence": confidence,
        "matched_aliases": list(matched_aliases),
    }


def _make_node(
    *,
    kind: str,
    node_type: NodeType,
    name: str,
    source_chunk_id: str,
    citation_text: str,
    matched_aliases: tuple[str, ...],
) -> LegalGraphNode:
    node_normalized_name = normalized_name(name)
    return LegalGraphNode(
        node_id=concept_node_id(kind, node_normalized_name),
        node_type=node_type,
        name=name,
        normalized_name=node_normalized_name,
        source_chunk_id=source_chunk_id,
        properties=_provenance_properties(
            citation_text=citation_text,
            matched_aliases=matched_aliases,
        ),
    )


def _make_edge(
    *,
    source_id: str,
    target_id: str,
    edge_type: EdgeType,
    source_chunk_id: str,
    citation_text: str,
    matched_aliases: tuple[str, ...],
) -> LegalGraphEdge:
    return LegalGraphEdge(
        edge_id=(
            f"{source_id}|{edge_type.value}|{target_id}|"
            f"{source_chunk_id}|{'_'.join(matched_aliases)}"
        ),
        source_id=source_id,
        target_id=target_id,
        edge_type=edge_type,
        confidence=0.85,
        source_chunk_id=source_chunk_id,
        extraction_method="dictionary",
        properties=_provenance_properties(
            citation_text=citation_text,
            matched_aliases=matched_aliases,
        ),
    )


def link_concepts_for_record(
    record: RetrievedRecord | dict[str, Any],
) -> tuple[tuple[LegalGraphNode, ...], tuple[LegalGraphEdge, ...]]:
    payload = record_to_payload(record)
    chunk_id = str(payload.get("chunk_id") or "")
    if not chunk_id:
        return (), ()

    citation_text = str(payload.get("citation_text") or "")
    text = normalize_for_matching(
        " ".join(
            str(payload.get(key) or "")
            for key in ("citation_text", "heading", "article_title", "text")
        )
    )
    evidence_id = evidence_chunk_node_id(chunk_id)
    unit_ids = legal_unit_node_ids_for_payload(payload)

    nodes: list[LegalGraphNode] = []
    edges: list[LegalGraphEdge] = []

    def add_links(
        dictionary: dict[str, tuple[str, ...]],
        *,
        kind: str,
        node_type: NodeType,
        edge_type: EdgeType,
    ) -> None:
        for name, aliases in dictionary.items():
            matched_aliases = tuple(
                alias for alias in aliases if normalize_for_matching(alias) in text
            )
            if not matched_aliases and not _contains_alias(text, aliases):
                continue
            node = _make_node(
                kind=kind,
                node_type=node_type,
                name=name,
                source_chunk_id=chunk_id,
                citation_text=citation_text,
                matched_aliases=matched_aliases or aliases[:1],
            )
            nodes.append(node)
            edges.append(
                _make_edge(
                    source_id=evidence_id,
                    target_id=node.node_id,
                    edge_type=edge_type,
                    source_chunk_id=chunk_id,
                    citation_text=citation_text,
                    matched_aliases=matched_aliases or aliases[:1],
                )
            )
            for unit_id in unit_ids:
                edges.append(
                    _make_edge(
                        source_id=unit_id,
                        target_id=node.node_id,
                        edge_type=edge_type,
                        source_chunk_id=chunk_id,
                        citation_text=citation_text,
                        matched_aliases=matched_aliases or aliases[:1],
                    )
                )

    add_links(
        LEGAL_CONCEPTS,
        kind="concept",
        node_type=NodeType.LEGAL_CONCEPT,
        edge_type=EdgeType.MENTIONS_CONCEPT,
    )
    add_links(
        SUBJECTS,
        kind="subject",
        node_type=NodeType.SUBJECT,
        edge_type=EdgeType.APPLIES_TO,
    )
    add_links(
        ACTIONS,
        kind="action",
        node_type=NodeType.ACTION,
        edge_type=EdgeType.REGULATES_ACTION,
    )
    return tuple(nodes), tuple(edges)


__all__ = [
    "ACTIONS",
    "LEGAL_CONCEPTS",
    "SUBJECTS",
    "link_concepts_for_record",
]
