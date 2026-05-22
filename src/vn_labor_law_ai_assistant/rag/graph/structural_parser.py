from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Sequence

from ...corpus_pipeline import normalize_for_matching
from ...heuristic_router import dedupe_preserve_order
from ..retrieval.manifest import load_manifest
from ..retrieval.models import RetrievedRecord


def document_node_id(document_id: str) -> str:
    return f"document:{document_id}"


def article_node_id(document_id: str, article_number: str) -> str:
    return f"article:{document_id}:{article_number}"


def clause_node_id(document_id: str, article_number: str, clause_ref: str) -> str:
    return f"clause:{document_id}:{article_number}:{clause_ref}"


def point_node_id(
    document_id: str,
    article_number: str,
    clause_ref: str,
    point_ref: str,
) -> str:
    return f"point:{document_id}:{article_number}:{clause_ref}:{point_ref}"


def evidence_chunk_node_id(chunk_id: str) -> str:
    return f"chunk:{chunk_id}"


def concept_node_id(kind: str, normalized_name: str) -> str:
    return f"{kind}:{normalized_name.replace(' ', '_')}"


def legal_unit_node_ids_for_payload(payload: dict[str, Any]) -> tuple[str, ...]:
    document_id = str(payload.get("document_id") or "").strip()
    article_number = str(payload.get("article_number") or "").strip()
    clause_ref = str(payload.get("clause_ref") or "").strip()
    point_ref = str(payload.get("point_ref") or "").strip()
    point_refs = tuple(str(value).strip() for value in payload.get("point_refs") or [] if value)
    if not document_id or not article_number:
        return ()

    node_ids: list[str] = [article_node_id(document_id, article_number)]
    if clause_ref:
        node_ids.append(clause_node_id(document_id, article_number, clause_ref))
        for current_point_ref in dedupe_preserve_order(
            (point_ref,) + point_refs if point_ref else point_refs
        ):
            node_ids.append(
                point_node_id(document_id, article_number, clause_ref, current_point_ref)
            )
    return tuple(node_ids)


def most_specific_legal_unit_node_id(payload: dict[str, Any]) -> str | None:
    document_id = str(payload.get("document_id") or "").strip()
    article_number = str(payload.get("article_number") or "").strip()
    clause_ref = str(payload.get("clause_ref") or "").strip()
    point_ref = str(payload.get("point_ref") or "").strip()
    level = str(payload.get("level") or "").strip()
    if not document_id or not article_number:
        return None
    if level == "point" and clause_ref and point_ref:
        return point_node_id(document_id, article_number, clause_ref, point_ref)
    if clause_ref:
        return clause_node_id(document_id, article_number, clause_ref)
    return article_node_id(document_id, article_number)


def record_to_payload(record: RetrievedRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, RetrievedRecord):
        payload = dict(record.payload)
        payload.setdefault("chunk_id", record.chunk_id)
        payload.setdefault("citation_text", record.citation_text)
        payload.setdefault("text", record.text)
        payload.setdefault("parent_chunk_id", record.parent_chunk_id)
        payload.setdefault("dense_text", record.dense_text)
        payload.setdefault("sparse_text", record.sparse_text)
        return payload

    if "payload" in record and isinstance(record["payload"], dict):
        payload = dict(record["payload"])
        payload.setdefault("chunk_id", record.get("chunk_id"))
        payload.setdefault("citation_text", record.get("citation_text"))
        payload.setdefault("text", record.get("text"))
        payload.setdefault("parent_chunk_id", record.get("parent_chunk_id"))
        payload.setdefault("dense_text", record.get("dense_text"))
        payload.setdefault("sparse_text", record.get("sparse_text"))
        return payload

    return dict(record)


def payload_to_record(payload: dict[str, Any]) -> RetrievedRecord:
    return RetrievedRecord(
        chunk_id=str(payload.get("chunk_id") or ""),
        parent_chunk_id=(
            str(payload.get("parent_chunk_id")) if payload.get("parent_chunk_id") else None
        ),
        citation_text=str(payload.get("citation_text") or ""),
        text=str(payload.get("text") or ""),
        dense_text=str(payload.get("dense_text") or ""),
        sparse_text=str(payload.get("sparse_text") or ""),
        payload=dict(payload),
    )


def iter_records_from_jsonl(path: Path) -> Iterable[RetrievedRecord]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = record_to_payload(json.loads(line))
        if payload.get("chunk_id"):
            yield payload_to_record(payload)


def iter_records_from_sqlite(path: Path) -> Iterable[RetrievedRecord]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT chunk_id, parent_chunk_id, citation_text, text, dense_text, sparse_text,
                   payload_json
            FROM records
            ORDER BY chunk_id
            """
        ).fetchall()
    finally:
        connection.close()

    for row in rows:
        payload = json.loads(row["payload_json"])
        payload.setdefault("chunk_id", row["chunk_id"])
        payload.setdefault("citation_text", row["citation_text"])
        payload.setdefault("text", row["text"])
        payload.setdefault("parent_chunk_id", row["parent_chunk_id"])
        payload.setdefault("dense_text", row["dense_text"])
        payload.setdefault("sparse_text", row["sparse_text"])
        yield payload_to_record(payload)


def resolve_index_artifact_path(index_path: Path, artifact_path: str) -> Path:
    path = Path(artifact_path)
    if path.is_absolute():
        return path
    return index_path.resolve().parents[1] / path if artifact_path.startswith("artifacts/") else path


def load_index_records(index_path: Path) -> tuple[RetrievedRecord, ...]:
    manifest = load_manifest(index_path)
    records_jsonl_path = str(manifest.get("records_jsonl_path") or "").strip()
    records_db_path = str(manifest.get("records_db_path") or "").strip()

    if records_jsonl_path:
        jsonl_path = resolve_index_artifact_path(index_path, records_jsonl_path)
        if jsonl_path.exists():
            return tuple(iter_records_from_jsonl(jsonl_path))

    if records_db_path:
        sqlite_path = resolve_index_artifact_path(index_path, records_db_path)
        if sqlite_path.exists():
            return tuple(iter_records_from_sqlite(sqlite_path))

    raise FileNotFoundError(
        "Could not find records_jsonl_path or records_db_path from index manifest."
    )


def normalized_name(value: object) -> str:
    return normalize_for_matching(str(value or "")).strip()


def compact_id_part(value: object) -> str:
    return normalized_name(value).replace(" ", "_")


def dedupe_nodes_by_id(nodes: Sequence[Any]) -> tuple[Any, ...]:
    seen: set[str] = set()
    ordered: list[Any] = []
    for node in nodes:
        if node.node_id in seen:
            continue
        seen.add(node.node_id)
        ordered.append(node)
    return tuple(ordered)


def dedupe_edges_by_id(edges: Sequence[Any]) -> tuple[Any, ...]:
    seen: set[str] = set()
    ordered: list[Any] = []
    for edge in edges:
        if edge.edge_id in seen:
            continue
        seen.add(edge.edge_id)
        ordered.append(edge)
    return tuple(ordered)


__all__ = [
    "article_node_id",
    "clause_node_id",
    "compact_id_part",
    "concept_node_id",
    "dedupe_edges_by_id",
    "dedupe_nodes_by_id",
    "document_node_id",
    "evidence_chunk_node_id",
    "iter_records_from_jsonl",
    "iter_records_from_sqlite",
    "legal_unit_node_ids_for_payload",
    "load_index_records",
    "most_specific_legal_unit_node_id",
    "normalized_name",
    "payload_to_record",
    "point_node_id",
    "record_to_payload",
]
