from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from ..retrieval.models import RetrievedRecord
from .concept_linker import link_concepts_for_record
from .models import LegalGraphEdge, LegalGraphNode
from .ontology import EdgeType, NodeType
from .store import LegalGraphStore
from .structural_parser import (
    appendix_node_id,
    article_node_id,
    canonical_legal_node_id,
    clause_node_id,
    dedupe_edges_by_id,
    dedupe_nodes_by_id,
    document_node_id,
    evidence_chunk_node_id,
    normalized_name,
    point_node_id,
    record_to_payload,
    taxonomy_node_id,
)


EVIDENCE_CHUNK_PAYLOAD_FIELDS = (
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
    "level",
    "chunk_type",
    "parent_chunk_id",
    "topic",
    "actor",
    "issue_type",
    "citation_text",
    "retrieval_text",
    "source_file",
    "document_hierarchy",
)

LABOR_CODE_DOCUMENT_ID = "45-2019-qh14"
EXPECTED_DOCUMENT_NORMATIVE_RANKS = {
    "45-2019-qh14": 1,
    "92-2015-qh13-labor-only": 1,
    "nghi-dinh-145-2020-nd-cp": 2,
    "nghi-dinh-135-2020-nd-cp": 2,
    "thong-tu-09-2020-tt-bldtbxh": 3,
    "thong-tu-10-2020-tt-bldtbxh": 3,
}
EXPECTED_LABOR_CODE_HIERARCHY_DOCUMENTS = (
    "nghi-dinh-135-2020-nd-cp",
    "nghi-dinh-145-2020-nd-cp",
    "thong-tu-09-2020-tt-bldtbxh",
    "thong-tu-10-2020-tt-bldtbxh",
)
EXPECTED_REFERENCE_ARTICLE_LINKS = {
    "thong-tu-09-2020-tt-bldtbxh": ("143", "145", "146", "147"),
    "thong-tu-10-2020-tt-bldtbxh": ("21", "73", "142"),
    "nghi-dinh-135-2020-nd-cp": ("169",),
}
REFERENCE_EDGE_TYPES = {
    EdgeType.REFERENCES,
    EdgeType.DETAILS,
    EdgeType.GUIDED_BY,
    EdgeType.GUIDES,
}


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


def _custom_edge(
    *,
    source_id: str,
    target_id: str,
    edge_type: EdgeType,
    edge_id: str,
    confidence: float = 1.0,
    source_chunk_id: str = "",
    extraction_method: str = "graph_builder",
    properties: dict[str, Any] | None = None,
) -> LegalGraphEdge:
    return LegalGraphEdge(
        edge_id=edge_id,
        source_id=source_id,
        target_id=target_id,
        edge_type=edge_type,
        confidence=confidence,
        source_chunk_id=source_chunk_id,
        extraction_method=extraction_method,
        properties=dict(properties or {}),
    )


def _source_file(payload: dict[str, Any]) -> str:
    source_file = str(payload.get("source_file") or "").strip()
    if source_file:
        return source_file
    source_path = str(payload.get("source_path") or "").strip()
    return source_path.replace("\\", "/").rsplit("/", 1)[-1] if source_path else ""


def _issuing_authority(document_type: str, existing: object = None) -> str:
    if existing:
        return str(existing)
    if document_type in {"bo_luat", "luat"}:
        return "Quốc hội"
    if document_type == "nghi_dinh":
        return "Chính phủ"
    if document_type == "thong_tu":
        return "Bộ trưởng / Thủ trưởng cơ quan ngang Bộ"
    return ""


def _rank_label(document_type: str, normative_rank: object, existing: object = None) -> str:
    if existing:
        return str(existing)
    if int(normative_rank or 0) == 1:
        return "highest"
    if int(normative_rank or 0) == 2:
        return "middle"
    if int(normative_rank or 0) == 3:
        return "lowest"
    if document_type in {"bo_luat", "luat"}:
        return "highest"
    if document_type == "nghi_dinh":
        return "middle"
    if document_type == "thong_tu":
        return "lowest"
    return ""


def _taxonomy_values(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    values = payload.get(key) or []
    if isinstance(values, str):
        values = [values]
    return tuple(str(value).strip() for value in values if str(value).strip())


def _node_type_for_taxonomy(kind: str) -> NodeType:
    if kind == "topic":
        return NodeType.LEGAL_TOPIC
    if kind == "actor":
        return NodeType.LEGAL_ACTOR
    if kind == "issue_type":
        return NodeType.LEGAL_ISSUE_TYPE
    raise ValueError(f"Unsupported taxonomy kind: {kind}")


def _edge_type_for_taxonomy(kind: str) -> EdgeType:
    if kind == "topic":
        return EdgeType.MENTIONS_TOPIC
    if kind == "actor":
        return EdgeType.APPLIES_TO_ACTOR
    if kind == "issue_type":
        return EdgeType.HAS_ISSUE_TYPE
    raise ValueError(f"Unsupported taxonomy kind: {kind}")


def _reference_edge_record_to_graph_edge(edge: dict[str, Any]) -> LegalGraphEdge | None:
    if edge.get("resolved") is not True:
        return None
    try:
        edge_type = EdgeType(str(edge.get("edge_type") or ""))
    except ValueError:
        return None
    if edge_type not in REFERENCE_EDGE_TYPES:
        return None
    artifact_source_id = str(edge.get("source_id") or "")
    artifact_target_id = str(edge.get("target_id") or "")
    source_id = canonical_legal_node_id(artifact_source_id)
    target_id = canonical_legal_node_id(artifact_target_id)
    if not source_id or not target_id:
        return None
    return _custom_edge(
        source_id=source_id,
        target_id=target_id,
        edge_type=edge_type,
        edge_id=str(edge.get("edge_id") or f"{source_id}|{edge_type.value}|{target_id}"),
        confidence=float(edge.get("confidence") or 1.0),
        source_chunk_id=str(edge.get("source_chunk_id") or ""),
        extraction_method=str(edge.get("extraction_method") or "reference_edges_artifact"),
        properties={
            "edge_id": edge.get("edge_id"),
            "source_id": source_id,
            "target_id": target_id,
            "artifact_source_id": artifact_source_id,
            "artifact_target_id": artifact_target_id,
            "source_chunk_id": edge.get("source_chunk_id"),
            "source_document_id": edge.get("source_document_id"),
            "target_document_id": edge.get("target_document_id"),
            "citation_text": edge.get("citation_text"),
            "original_matched_text": edge.get("original_matched_text"),
            "normalized_matched_text": edge.get("normalized_matched_text"),
            "extraction_method": edge.get("extraction_method"),
            "confidence": edge.get("confidence"),
            "resolved": edge.get("resolved"),
            "target_article": edge.get("target_article"),
            "target_clause": edge.get("target_clause"),
            "target_point": edge.get("target_point"),
            "reference_level": edge.get("reference_level"),
            "source_artifact": "reference_edges",
        },
    )


def _add_normative_edge(
    edges: list[LegalGraphEdge],
    *,
    source_document_id: str,
    target_document_id: str,
    edge_type: EdgeType,
    reason: str,
) -> None:
    source_id = document_node_id(source_document_id)
    target_id = document_node_id(target_document_id)
    edges.append(
        _custom_edge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            edge_id=f"{source_id}|{edge_type.value}|{target_id}|normative_hierarchy",
            extraction_method="normative_hierarchy",
            properties={
                "source_document_id": source_document_id,
                "target_document_id": target_document_id,
                "reason": reason,
                "confidence": 1.0,
                "edge_category": "normative_hierarchy",
            },
        )
    )


def _add_normative_hierarchy_edges(
    edges: list[LegalGraphEdge],
    document_ids: set[str],
) -> None:
    if LABOR_CODE_DOCUMENT_ID not in document_ids:
        return
    for lower_document_id in EXPECTED_LABOR_CODE_HIERARCHY_DOCUMENTS:
        if lower_document_id not in document_ids:
            continue
        _add_normative_edge(
            edges,
            source_document_id=LABOR_CODE_DOCUMENT_ID,
            target_document_id=lower_document_id,
            edge_type=EdgeType.SUPERIOR_TO,
            reason="labor_code_superior_to_implementing_document",
        )
        _add_normative_edge(
            edges,
            source_document_id=lower_document_id,
            target_document_id=LABOR_CODE_DOCUMENT_ID,
            edge_type=EdgeType.SUBORDINATE_TO,
            reason="implementing_document_subordinate_to_labor_code",
        )
        _add_normative_edge(
            edges,
            source_document_id=lower_document_id,
            target_document_id=LABOR_CODE_DOCUMENT_ID,
            edge_type=EdgeType.MUST_COMPLY_WITH,
            reason="lower_rank_document_must_comply_with_labor_code",
        )
        _add_normative_edge(
            edges,
            source_document_id=lower_document_id,
            target_document_id=LABOR_CODE_DOCUMENT_ID,
            edge_type=EdgeType.IMPLEMENTS,
            reason="document_implements_labor_code",
        )
        _add_normative_edge(
            edges,
            source_document_id=lower_document_id,
            target_document_id=LABOR_CODE_DOCUMENT_ID,
            edge_type=EdgeType.GUIDES,
            reason="document_guides_labor_code",
        )


def _loaded_reference_summary(reference_edges: Sequence[dict[str, Any]]) -> dict[str, int]:
    resolved = [edge for edge in reference_edges if edge.get("resolved") is True]
    unresolved = [edge for edge in reference_edges if edge.get("resolved") is False]
    return {
        "resolved": len(resolved),
        "unresolved": len(unresolved),
    }


def _coerce_int(value: object) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _document_id_from_node_id(node_id: str) -> str:
    parts = str(node_id or "").split(":")
    if parts[0] in {"document", "article", "clause", "point", "appendix"} and len(parts) >= 2:
        return parts[1]
    if parts[0] == "chunk":
        return ""
    return ""


def _article_number_from_node_id(node_id: str) -> str:
    parts = str(node_id or "").split(":")
    if parts[0] in {"article", "clause", "point"} and len(parts) >= 3:
        return parts[2].removeprefix("dieu-")
    return ""


def _edge_connects_documents(edge: LegalGraphEdge, left_document_id: str, right_document_id: str) -> bool:
    source_document_id = _document_id_from_node_id(edge.source_id)
    target_document_id = _document_id_from_node_id(edge.target_id)
    return {source_document_id, target_document_id} == {left_document_id, right_document_id}


def _has_reference_article_connection(
    edges: Sequence[LegalGraphEdge],
    *,
    source_document_id: str,
    target_document_id: str,
    target_article_number: str,
) -> bool:
    for edge in edges:
        if edge.properties.get("source_artifact") != "reference_edges":
            continue
        if edge.edge_type not in REFERENCE_EDGE_TYPES:
            continue
        if not _edge_connects_documents(edge, source_document_id, target_document_id):
            continue

        source_doc = _document_id_from_node_id(edge.source_id)
        target_doc = _document_id_from_node_id(edge.target_id)
        source_article = _article_number_from_node_id(edge.source_id)
        target_article = _article_number_from_node_id(edge.target_id)
        if source_doc == target_document_id and source_article == target_article_number:
            return True
        if target_doc == target_document_id and target_article == target_article_number:
            return True
    return False


def _distinct_reference_articles_between_documents(
    edges: Sequence[LegalGraphEdge],
    *,
    source_document_id: str,
    target_document_id: str,
) -> tuple[str, ...]:
    article_numbers: set[str] = set()
    for edge in edges:
        if edge.properties.get("source_artifact") != "reference_edges":
            continue
        if edge.edge_type not in REFERENCE_EDGE_TYPES:
            continue
        if not _edge_connects_documents(edge, source_document_id, target_document_id):
            continue
        for node_id in (edge.source_id, edge.target_id):
            if _document_id_from_node_id(node_id) == target_document_id:
                article_number = _article_number_from_node_id(node_id)
                if article_number:
                    article_numbers.add(article_number)
    return tuple(
        sorted(
            article_numbers,
            key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
        )
    )


def _normative_rank_mismatches(
    documents: dict[str, dict[str, Any]]
) -> dict[str, dict[str, int | None]]:
    mismatches: dict[str, dict[str, int | None]] = {}
    for document_id, expected_rank in EXPECTED_DOCUMENT_NORMATIVE_RANKS.items():
        if document_id not in documents:
            continue
        actual_rank = _coerce_int(documents[document_id].get("normative_rank"))
        if actual_rank != expected_rank:
            mismatches[document_id] = {
                "expected": expected_rank,
                "actual": actual_rank,
            }
    return mismatches


def _labor_code_hierarchy_missing_edges(edges: Sequence[LegalGraphEdge]) -> list[str]:
    edge_set = {(edge.source_id, edge.edge_type, edge.target_id) for edge in edges}
    missing: list[str] = []
    for lower_document_id in EXPECTED_LABOR_CODE_HIERARCHY_DOCUMENTS:
        source_id = document_node_id(LABOR_CODE_DOCUMENT_ID)
        target_id = document_node_id(lower_document_id)
        if (source_id, EdgeType.SUPERIOR_TO, target_id) not in edge_set:
            missing.append(f"{LABOR_CODE_DOCUMENT_ID} SUPERIOR_TO {lower_document_id}")
    return missing


def _expected_reference_article_link_validation(
    edges: Sequence[LegalGraphEdge],
) -> dict[str, dict[str, object]]:
    validation: dict[str, dict[str, object]] = {}
    for source_document_id, article_numbers in EXPECTED_REFERENCE_ARTICLE_LINKS.items():
        missing_articles = [
            article_number
            for article_number in article_numbers
            if not _has_reference_article_connection(
                edges,
                source_document_id=source_document_id,
                target_document_id=LABOR_CODE_DOCUMENT_ID,
                target_article_number=article_number,
            )
        ]
        validation[source_document_id] = {
            "expected_articles": list(article_numbers),
            "missing_articles": missing_articles,
            "passed": not missing_articles,
        }
    return validation


def _blttds_labor_taxonomy_present(nodes: Sequence[LegalGraphNode]) -> bool:
    for node in nodes:
        if node.node_type != NodeType.EVIDENCE_CHUNK:
            continue
        if node.properties.get("document_id") != "92-2015-qh13-labor-only":
            continue
        topics = set(_taxonomy_values(node.properties, "topic"))
        actors = set(_taxonomy_values(node.properties, "actor"))
        issue_types = set(_taxonomy_values(node.properties, "issue_type"))
        if (
            {"to_tung_lao_dong", "tranh_chap_lao_dong"} & topics
            and "toa_an" in actors
            and {"tham_quyen_toa_an", "tranh_chap_lao_dong"} & issue_types
        ):
            return True
    return False


class LegalGraphBuilder:
    def __init__(
        self,
        *,
        with_concepts: bool = True,
        with_references: bool = True,
        with_normative_hierarchy: bool = True,
        reference_edges: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self.with_concepts = with_concepts
        self.with_references = with_references
        self.with_normative_hierarchy = with_normative_hierarchy
        self.reference_edges = tuple(dict(edge) for edge in (reference_edges or ()))

    def build(
        self,
        records: Sequence[RetrievedRecord | dict[str, Any]],
        *,
        build_metadata: dict[str, object] | None = None,
    ) -> LegalGraphBuildResult:
        nodes: list[LegalGraphNode] = []
        edges: list[LegalGraphEdge] = []
        documents: dict[str, dict[str, Any]] = {}
        warnings: list[str] = []

        for record in records:
            payload = record_to_payload(record)
            chunk_id = str(payload.get("chunk_id") or "").strip()
            document_id = str(payload.get("document_id") or "").strip()
            article_number = str(payload.get("article_number") or "").strip()
            if not chunk_id or not document_id:
                continue

            citation_text = str(payload.get("citation_text") or "")
            document_title = str(payload.get("document_title") or document_id)
            document_type = str(payload.get("document_type") or "")
            normative_rank = payload.get("normative_rank")
            rank_label = _rank_label(document_type, normative_rank, payload.get("rank_label"))
            issuing_authority = _issuing_authority(document_type, payload.get("issuing_authority"))
            doc_id = document_node_id(document_id)
            evidence_id = evidence_chunk_node_id(chunk_id)
            document_state = documents.setdefault(
                document_id,
                {
                    "node_id": doc_id,
                    "document_id": document_id,
                    "document_title": document_title,
                    "document_type": document_type,
                    "normative_rank": normative_rank,
                    "rank_label": rank_label,
                    "issuing_authority": issuing_authority,
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
                        **{
                            field: (
                                _source_file(payload)
                                if field == "source_file"
                                else dict(payload.get("document_hierarchy") or {})
                                if field == "document_hierarchy"
                                else list(payload.get(field) or [])
                                if field in {"point_refs", "topic", "actor", "issue_type"}
                                else payload.get(field)
                            )
                            for field in EVIDENCE_CHUNK_PAYLOAD_FIELDS
                        },
                        "qdrant_point_id": payload.get("qdrant_point_id"),
                        "source_path": payload.get("source_path"),
                        "text": payload.get("text"),
                    },
                )
            )

            document_hierarchy = dict(payload.get("document_hierarchy") or {})
            appendix_id = str(document_hierarchy.get("appendix_id") or "").strip()
            appendix_heading = str(document_hierarchy.get("appendix_heading") or payload.get("heading") or "").strip()
            source_unit_id = ""
            if not article_number and appendix_id:
                current_appendix_id = appendix_node_id(document_id, appendix_id)
                nodes.append(
                    LegalGraphNode(
                        node_id=current_appendix_id,
                        node_type=NodeType.LEGAL_APPENDIX,
                        name=appendix_heading or appendix_id,
                        normalized_name=normalized_name(appendix_heading or appendix_id),
                        source_chunk_id=chunk_id,
                        properties={
                            **_node_provenance(payload),
                            "document_id": document_id,
                            "appendix_id": appendix_id,
                            "appendix_heading": appendix_heading,
                            "chunk_type": payload.get("chunk_type"),
                        },
                    )
                )
                edges.append(
                    _edge(
                        source_id=doc_id,
                        target_id=current_appendix_id,
                        edge_type=EdgeType.HAS_APPENDIX,
                        payload=payload,
                    )
                )
                source_unit_id = current_appendix_id

            if not article_number and not source_unit_id:
                warnings.append(f"Chunk has no article_number or appendix_id: {chunk_id}")
                source_unit_id = doc_id

            if not article_number:
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
                for kind, values in (
                    ("topic", _taxonomy_values(payload, "topic")),
                    ("actor", _taxonomy_values(payload, "actor")),
                    ("issue_type", _taxonomy_values(payload, "issue_type")),
                ):
                    for value in values:
                        taxonomy_id = taxonomy_node_id(kind, value)
                        taxonomy_type = _node_type_for_taxonomy(kind)
                        nodes.append(
                            LegalGraphNode(
                                node_id=taxonomy_id,
                                node_type=taxonomy_type,
                                name=value,
                                normalized_name=normalized_name(value),
                                source_chunk_id=chunk_id,
                                properties={"value": value, "kind": kind},
                            )
                        )
                        edges.append(
                            _custom_edge(
                                source_id=evidence_id,
                                target_id=taxonomy_id,
                                edge_type=_edge_type_for_taxonomy(kind),
                                edge_id=(
                                    f"{evidence_id}|{_edge_type_for_taxonomy(kind).value}|"
                                    f"{taxonomy_id}|{chunk_id}"
                                ),
                                source_chunk_id=chunk_id,
                                extraction_method="chunk_metadata_taxonomy",
                                properties={
                                    "chunk_id": chunk_id,
                                    "value": value,
                                    "kind": kind,
                                    "citation_text": citation_text,
                                },
                            )
                        )
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
            for kind, values in (
                ("topic", _taxonomy_values(payload, "topic")),
                ("actor", _taxonomy_values(payload, "actor")),
                ("issue_type", _taxonomy_values(payload, "issue_type")),
            ):
                for value in values:
                    taxonomy_id = taxonomy_node_id(kind, value)
                    taxonomy_type = _node_type_for_taxonomy(kind)
                    nodes.append(
                        LegalGraphNode(
                            node_id=taxonomy_id,
                            node_type=taxonomy_type,
                            name=value,
                            normalized_name=normalized_name(value),
                            source_chunk_id=chunk_id,
                            properties={"value": value, "kind": kind},
                        )
                    )
                    edges.append(
                        _custom_edge(
                            source_id=evidence_id,
                            target_id=taxonomy_id,
                            edge_type=_edge_type_for_taxonomy(kind),
                            edge_id=(
                                f"{evidence_id}|{_edge_type_for_taxonomy(kind).value}|"
                                f"{taxonomy_id}|{chunk_id}"
                            ),
                            source_chunk_id=chunk_id,
                            extraction_method="chunk_metadata_taxonomy",
                            properties={
                                "chunk_id": chunk_id,
                                "value": value,
                                "kind": kind,
                                "citation_text": citation_text,
                            },
                        )
                    )

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
                        "document_type": document.get("document_type"),
                        "normative_rank": document.get("normative_rank"),
                        "rank_label": document.get("rank_label"),
                        "issuing_authority": document.get("issuing_authority"),
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

        if self.with_references:
            for reference_edge in self.reference_edges:
                graph_edge = _reference_edge_record_to_graph_edge(reference_edge)
                if graph_edge is not None:
                    edges.append(graph_edge)
        if self.with_normative_hierarchy:
            _add_normative_hierarchy_edges(edges, set(documents))

        deduped_nodes = dedupe_nodes_by_id(nodes)
        deduped_edges = dedupe_edges_by_id(edges)
        type_counts = Counter(node.node_type.value for node in deduped_nodes)
        edge_counts = Counter(edge.edge_type.value for edge in deduped_edges)
        reference_artifact_edges = tuple(
            edge
            for edge in deduped_edges
            if edge.properties.get("source_artifact") == "reference_edges"
        )
        reference_edge_counts = Counter(edge.edge_type.value for edge in reference_artifact_edges)
        normative_hierarchy_edges = tuple(
            edge
            for edge in deduped_edges
            if edge.properties.get("edge_category") == "normative_hierarchy"
        )
        normative_hierarchy_edge_counts = Counter(
            edge.edge_type.value for edge in normative_hierarchy_edges
        )
        reference_summary = _loaded_reference_summary(self.reference_edges)
        source_of_targets = {edge.source_id for edge in deduped_edges if edge.edge_type == EdgeType.SOURCE_OF}
        evidence_node_ids = {
            node.node_id for node in deduped_nodes if node.node_type == NodeType.EVIDENCE_CHUNK
        }
        orphan_chunk_count = len(evidence_node_ids - source_of_targets)
        document_chunk_counts = Counter(
            node.properties.get("document_id")
            for node in deduped_nodes
            if node.node_type == NodeType.EVIDENCE_CHUNK
        )
        documents_without_chunks = [
            document_id for document_id in documents if document_chunk_counts.get(document_id, 0) == 0
        ]
        normative_rank_mismatches = _normative_rank_mismatches(documents)
        missing_hierarchy_edges = _labor_code_hierarchy_missing_edges(deduped_edges)
        reference_article_validation = _expected_reference_article_link_validation(deduped_edges)
        nd145_labor_code_articles = _distinct_reference_articles_between_documents(
            deduped_edges,
            source_document_id="nghi-dinh-145-2020-nd-cp",
            target_document_id=LABOR_CODE_DOCUMENT_ID,
        )
        metadata_record_count = _coerce_int(
            (build_metadata or {}).get("manifest_record_count") if build_metadata else None
        )
        expected_chunk_count = metadata_record_count or len(records)
        summary: dict[str, object] = {
            "documents": type_counts[NodeType.LEGAL_DOCUMENT.value],
            "articles": type_counts[NodeType.LEGAL_ARTICLE.value],
            "clauses": type_counts[NodeType.LEGAL_CLAUSE.value],
            "points": type_counts[NodeType.LEGAL_POINT.value],
            "appendices": type_counts[NodeType.LEGAL_APPENDIX.value],
            "evidence_chunks": type_counts[NodeType.EVIDENCE_CHUNK.value],
            "topic_nodes": type_counts[NodeType.LEGAL_TOPIC.value],
            "actor_nodes": type_counts[NodeType.LEGAL_ACTOR.value],
            "issue_type_nodes": type_counts[NodeType.LEGAL_ISSUE_TYPE.value],
            "edges": len(deduped_edges),
            "concept_nodes": (
                type_counts[NodeType.LEGAL_CONCEPT.value]
                + type_counts[NodeType.SUBJECT.value]
                + type_counts[NodeType.ACTION.value]
            ),
            "reference_edges": (
                reference_edge_counts[EdgeType.REFERENCES.value]
                + reference_edge_counts[EdgeType.DETAILS.value]
                + reference_edge_counts[EdgeType.GUIDED_BY.value]
                + reference_edge_counts[EdgeType.GUIDES.value]
            ),
            "references_edges": reference_edge_counts[EdgeType.REFERENCES.value],
            "details_edges": reference_edge_counts[EdgeType.DETAILS.value],
            "guided_by_edges": reference_edge_counts[EdgeType.GUIDED_BY.value],
            "guides_edges": reference_edge_counts[EdgeType.GUIDES.value],
            "taxonomy_edges": (
                edge_counts[EdgeType.MENTIONS_TOPIC.value]
                + edge_counts[EdgeType.APPLIES_TO_ACTOR.value]
                + edge_counts[EdgeType.HAS_ISSUE_TYPE.value]
            ),
            "normative_hierarchy_edges": len(normative_hierarchy_edges),
            "normative_hierarchy_edges_by_type": dict(
                sorted(normative_hierarchy_edge_counts.items())
            ),
            "unresolved_edges_skipped": reference_summary["unresolved"],
            "duplicate_edges_skipped": len(edges) - len(deduped_edges),
            "documents_without_chunks": documents_without_chunks,
            "orphan_evidence_chunks": orphan_chunk_count,
            "details_guided_by_balanced": (
                reference_edge_counts[EdgeType.DETAILS.value]
                == reference_edge_counts[EdgeType.GUIDED_BY.value]
            ),
            "normative_rank_mismatches": normative_rank_mismatches,
            "missing_hierarchy_edges": missing_hierarchy_edges,
            "expected_reference_article_links": reference_article_validation,
            "nd145_labor_code_article_count": len(nd145_labor_code_articles),
            "nd145_labor_code_articles": list(nd145_labor_code_articles),
            "warnings": sorted(set(warnings)),
            "validation": {
                "evidence_chunk_count_expected": expected_chunk_count,
                "evidence_chunk_count_matches_input": (
                    type_counts[NodeType.EVIDENCE_CHUNK.value] == len(records)
                ),
                "evidence_chunk_count_matches_index": (
                    type_counts[NodeType.EVIDENCE_CHUNK.value] == expected_chunk_count
                ),
                "legal_document_count_is_6": type_counts[NodeType.LEGAL_DOCUMENT.value] == 6,
                "all_documents_have_chunks": not documents_without_chunks,
                "all_documents_have_correct_normative_rank": not normative_rank_mismatches,
                "no_orphan_evidence_chunks": orphan_chunk_count == 0,
                "no_unresolved_reference_edges_loaded": not any(
                    edge.properties.get("resolved") is False for edge in reference_artifact_edges
                ),
                "details_guided_by_balanced": (
                    reference_edge_counts[EdgeType.DETAILS.value]
                    == reference_edge_counts[EdgeType.GUIDED_BY.value]
                ),
                "labor_code_hierarchy_connections_present": not missing_hierarchy_edges,
                "tt09_labor_code_article_links_present": reference_article_validation.get(
                    "thong-tu-09-2020-tt-bldtbxh", {}
                ).get("passed")
                is True,
                "tt10_labor_code_article_links_present": reference_article_validation.get(
                    "thong-tu-10-2020-tt-bldtbxh", {}
                ).get("passed")
                is True,
                "nd135_labor_code_article_169_link_present": reference_article_validation.get(
                    "nghi-dinh-135-2020-nd-cp", {}
                ).get("passed")
                is True,
                "nd145_connects_to_multiple_labor_code_articles": (
                    len(nd145_labor_code_articles) > 1
                ),
                "blttds_labor_litigation_taxonomy_present": _blttds_labor_taxonomy_present(
                    deduped_nodes
                ),
            },
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
