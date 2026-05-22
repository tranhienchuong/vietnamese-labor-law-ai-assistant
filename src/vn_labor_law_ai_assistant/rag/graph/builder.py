from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from ..retrieval.models import RetrievedRecord
from .concept_linker import link_concepts_for_record
from .cross_reference_parser import build_reference_edges
from .models import LegalGraphEdge, LegalGraphNode
from .ontology import EdgeType, NodeType
from .store import LegalGraphStore
from .structural_parser import (
    article_node_id,
    clause_node_id,
    dedupe_edges_by_id,
    dedupe_nodes_by_id,
    document_node_id,
    evidence_chunk_node_id,
    normalized_name,
    point_node_id,
    record_to_payload,
)


@dataclass(frozen=True)
class LegalGraphBuildResult:
    nodes: tuple[LegalGraphNode, ...]
    edges: tuple[LegalGraphEdge, ...]
    summary: dict[str, object]


def _node_provenance(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "citation_text": str(payload.get("citation_text") or ""),
        "extraction_method": "structural_metadata",
        "confidence": 1.0,
    }


def _edge(
    *,
    source_id: str,
    target_id: str,
    edge_type: EdgeType,
    payload: dict[str, Any],
) -> LegalGraphEdge:
    source_chunk_id = str(payload.get("chunk_id") or "")
    return LegalGraphEdge(
        edge_id=f"{source_id}|{edge_type.value}|{target_id}|{source_chunk_id}",
        source_id=source_id,
        target_id=target_id,
        edge_type=edge_type,
        confidence=1.0,
        source_chunk_id=source_chunk_id,
        extraction_method="structural_metadata",
        properties=_node_provenance(payload),
    )


class LegalGraphBuilder:
    def __init__(
        self,
        *,
        with_concepts: bool = True,
        with_references: bool = True,
    ) -> None:
        self.with_concepts = with_concepts
        self.with_references = with_references

    def build(
        self,
        records: Sequence[RetrievedRecord | dict[str, Any]],
        *,
        build_metadata: dict[str, object] | None = None,
    ) -> LegalGraphBuildResult:
        nodes: list[LegalGraphNode] = []
        edges: list[LegalGraphEdge] = []
        documents: dict[str, dict[str, Any]] = {}

        for record in records:
            payload = record_to_payload(record)
            chunk_id = str(payload.get("chunk_id") or "").strip()
            document_id = str(payload.get("document_id") or "").strip()
            article_number = str(payload.get("article_number") or "").strip()
            if not chunk_id or not document_id:
                continue

            citation_text = str(payload.get("citation_text") or "")
            document_title = str(payload.get("document_title") or document_id)
            doc_id = document_node_id(document_id)
            evidence_id = evidence_chunk_node_id(chunk_id)
            document_state = documents.setdefault(
                document_id,
                {
                    "node_id": doc_id,
                    "document_id": document_id,
                    "document_title": document_title,
                    "source_chunk_ids": [],
                    "citation_texts": [],
                },
            )
            document_state["source_chunk_ids"].append(chunk_id)
            if citation_text:
                document_state["citation_texts"].append(citation_text)
            nodes.append(
                LegalGraphNode(
                    node_id=evidence_id,
                    node_type=NodeType.EVIDENCE_CHUNK,
                    name=citation_text or chunk_id,
                    normalized_name=normalized_name(citation_text or chunk_id),
                    source_chunk_id=chunk_id,
                    properties={
                        **_node_provenance(payload),
                        "chunk_id": chunk_id,
                        "qdrant_point_id": payload.get("qdrant_point_id"),
                        "document_id": document_id,
                        "article_number": payload.get("article_number"),
                        "clause_ref": payload.get("clause_ref"),
                        "point_ref": payload.get("point_ref"),
                        "point_refs": list(payload.get("point_refs") or []),
                        "level": payload.get("level"),
                        "chunk_type": payload.get("chunk_type"),
                    },
                )
            )

            if not article_number:
                continue

            article_id = article_node_id(document_id, article_number)
            article_name = (
                str(payload.get("heading") or "").strip()
                or f"Dieu {article_number}"
            )
            nodes.append(
                LegalGraphNode(
                    node_id=article_id,
                    node_type=NodeType.LEGAL_ARTICLE,
                    name=article_name,
                    normalized_name=normalized_name(article_name),
                    source_chunk_id=chunk_id,
                    properties={
                        **_node_provenance(payload),
                        "document_id": document_id,
                        "article_number": article_number,
                        "article_title": payload.get("article_title"),
                        "chapter_heading": payload.get("chapter_heading"),
                        "section_heading": payload.get("section_heading"),
                    },
                )
            )
            edges.append(
                _edge(
                    source_id=doc_id,
                    target_id=article_id,
                    edge_type=EdgeType.HAS_ARTICLE,
                    payload=payload,
                )
            )

            level = str(payload.get("level") or "")
            clause_ref = str(payload.get("clause_ref") or "").strip()
            point_ref = str(payload.get("point_ref") or "").strip()
            point_refs = tuple(str(value).strip() for value in payload.get("point_refs") or [] if value)
            source_unit_id = article_id

            if clause_ref:
                clause_id = clause_node_id(document_id, article_number, clause_ref)
                clause_name = f"Dieu {article_number} khoan {clause_ref}"
                nodes.append(
                    LegalGraphNode(
                        node_id=clause_id,
                        node_type=NodeType.LEGAL_CLAUSE,
                        name=clause_name,
                        normalized_name=normalized_name(clause_name),
                        source_chunk_id=chunk_id,
                        properties={
                            **_node_provenance(payload),
                            "document_id": document_id,
                            "article_number": article_number,
                            "clause_ref": clause_ref,
                        },
                    )
                )
                edges.append(
                    _edge(
                        source_id=article_id,
                        target_id=clause_id,
                        edge_type=EdgeType.HAS_CLAUSE,
                        payload=payload,
                    )
                )
                source_unit_id = clause_id

                for current_point_ref in (point_ref,) + point_refs if point_ref else point_refs:
                    point_id = point_node_id(
                        document_id,
                        article_number,
                        clause_ref,
                        current_point_ref,
                    )
                    point_name = (
                        f"Dieu {article_number} khoan {clause_ref} diem {current_point_ref}"
                    )
                    nodes.append(
                        LegalGraphNode(
                            node_id=point_id,
                            node_type=NodeType.LEGAL_POINT,
                            name=point_name,
                            normalized_name=normalized_name(point_name),
                            source_chunk_id=chunk_id,
                            properties={
                                **_node_provenance(payload),
                                "document_id": document_id,
                                "article_number": article_number,
                                "clause_ref": clause_ref,
                                "point_ref": current_point_ref,
                            },
                        )
                    )
                    edges.append(
                        _edge(
                            source_id=clause_id,
                            target_id=point_id,
                            edge_type=EdgeType.HAS_POINT,
                            payload=payload,
                        )
                    )
                    if level == "point" or current_point_ref == point_ref:
                        source_unit_id = point_id

            edges.append(
                _edge(
                    source_id=source_unit_id,
                    target_id=evidence_id,
                    edge_type=EdgeType.HAS_SOURCE_CHUNK,
                    payload=payload,
                )
            )
            edges.append(
                _edge(
                    source_id=evidence_id,
                    target_id=source_unit_id,
                    edge_type=EdgeType.SOURCE_OF,
                    payload=payload,
                )
            )

            if self.with_concepts:
                concept_nodes, concept_edges = link_concepts_for_record(payload)
                nodes.extend(concept_nodes)
                edges.extend(concept_edges)
            if self.with_references:
                edges.extend(build_reference_edges(payload))

        for document in documents.values():
            source_chunk_ids = tuple(dict.fromkeys(document["source_chunk_ids"]))
            citation_texts = tuple(dict.fromkeys(document["citation_texts"]))
            metadata_properties = dict(build_metadata or {})
            nodes.append(
                LegalGraphNode(
                    node_id=str(document["node_id"]),
                    node_type=NodeType.LEGAL_DOCUMENT,
                    name=str(document["document_title"]),
                    normalized_name=normalized_name(document["document_title"]),
                    source_chunk_id="",
                    properties={
                        "document_id": document["document_id"],
                        "citation_text": "",
                        "source_chunk_ids": list(source_chunk_ids),
                        "source_chunk_count": len(source_chunk_ids),
                        "citation_texts": list(citation_texts[:50]),
                        "extraction_method": "structural_metadata",
                        "confidence": 1.0,
                        **metadata_properties,
                    },
                )
            )

        deduped_nodes = dedupe_nodes_by_id(nodes)
        deduped_edges = dedupe_edges_by_id(edges)
        type_counts = Counter(node.node_type.value for node in deduped_nodes)
        edge_counts = Counter(edge.edge_type.value for edge in deduped_edges)
        summary: dict[str, object] = {
            "documents": type_counts[NodeType.LEGAL_DOCUMENT.value],
            "articles": type_counts[NodeType.LEGAL_ARTICLE.value],
            "clauses": type_counts[NodeType.LEGAL_CLAUSE.value],
            "points": type_counts[NodeType.LEGAL_POINT.value],
            "evidence_chunks": type_counts[NodeType.EVIDENCE_CHUNK.value],
            "edges": len(deduped_edges),
            "concept_nodes": (
                type_counts[NodeType.LEGAL_CONCEPT.value]
                + type_counts[NodeType.SUBJECT.value]
                + type_counts[NodeType.ACTION.value]
            ),
            "reference_edges": edge_counts[EdgeType.REFERENCES.value],
        }
        if build_metadata:
            summary["build_metadata"] = dict(build_metadata)
        return LegalGraphBuildResult(nodes=deduped_nodes, edges=deduped_edges, summary=summary)

    def build_and_upsert(
        self,
        records: Sequence[RetrievedRecord | dict[str, Any]],
        store: LegalGraphStore,
        *,
        build_metadata: dict[str, object] | None = None,
    ) -> LegalGraphBuildResult:
        result = self.build(records, build_metadata=build_metadata)
        store.upsert_nodes(result.nodes)
        store.upsert_edges(result.edges)
        return result


__all__ = [
    "LegalGraphBuildResult",
    "LegalGraphBuilder",
]
