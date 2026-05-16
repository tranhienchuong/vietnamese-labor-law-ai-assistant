from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Sequence

from ...heuristic_router import dedupe_preserve_order
from ...indexing import make_qdrant_point_id
from .models import RetrievedRecord, SearchHit
from .utils import record_from_qdrant_payload, record_reference_sort_key


class RecordStore:
    def fetch_records(self, chunk_ids: Sequence[str]) -> dict[str, RetrievedRecord]:
        raise NotImplementedError

    def fetch_records_from_hits(self, hits: Sequence[SearchHit]) -> dict[str, RetrievedRecord]:
        return self.fetch_records([hit.chunk_id for hit in hits])

    def fetch_records_by_reference(
        self,
        *,
        document_ids: Sequence[str] = (),
        article_numbers: Sequence[str] = (),
        clause_refs: Sequence[str] = (),
        point_refs: Sequence[str] = (),
        exclude_chunk_ids: Sequence[str] = (),
        limit: int = 12,
    ) -> tuple[RetrievedRecord, ...]:
        raise NotImplementedError

    def fetch_article_siblings(
        self,
        *,
        document_id: str,
        article_number: str,
        exclude_chunk_ids: Sequence[str] = (),
        limit: int = 6,
    ) -> tuple[RetrievedRecord, ...]:
        return self.fetch_records_by_reference(
            document_ids=(document_id,),
            article_numbers=(article_number,),
            exclude_chunk_ids=exclude_chunk_ids,
            limit=limit,
        )

    def close(self) -> None:
        return None


class SQLiteRecordStore(RecordStore):
    def __init__(self, records_db_path: Path) -> None:
        self.sqlite = sqlite3.connect(records_db_path)
        self.sqlite.row_factory = sqlite3.Row

    @staticmethod
    def records_from_rows(rows: Sequence[sqlite3.Row]) -> dict[str, RetrievedRecord]:
        records: dict[str, RetrievedRecord] = {}
        for row in rows:
            payload = json.loads(row["payload_json"])
            records[str(row["chunk_id"])] = RetrievedRecord(
                chunk_id=str(row["chunk_id"]),
                parent_chunk_id=str(row["parent_chunk_id"]) if row["parent_chunk_id"] else None,
                citation_text=str(row["citation_text"]),
                text=str(row["text"]),
                dense_text=str(row["dense_text"]),
                sparse_text=str(row["sparse_text"]),
                payload=payload,
            )
        return records

    def fetch_records(self, chunk_ids: Sequence[str]) -> dict[str, RetrievedRecord]:
        ordered_ids = dedupe_preserve_order(chunk_ids)
        if not ordered_ids:
            return {}

        placeholders = ", ".join("?" for _ in ordered_ids)
        rows = self.sqlite.execute(
            f"""
            SELECT chunk_id, parent_chunk_id, citation_text, text, dense_text, sparse_text, payload_json
            FROM records
            WHERE chunk_id IN ({placeholders})
            """,
            ordered_ids,
        ).fetchall()

        return self.records_from_rows(rows)

    def fetch_records_by_reference(
        self,
        *,
        document_ids: Sequence[str] = (),
        article_numbers: Sequence[str] = (),
        clause_refs: Sequence[str] = (),
        point_refs: Sequence[str] = (),
        exclude_chunk_ids: Sequence[str] = (),
        limit: int = 12,
    ) -> tuple[RetrievedRecord, ...]:
        where_parts: list[str] = []
        params: list[object] = []

        def add_in_filter(field_name: str, values: Sequence[str]) -> None:
            ordered_values = dedupe_preserve_order(tuple(value for value in values if value))
            if not ordered_values:
                return
            placeholders = ", ".join("?" for _ in ordered_values)
            where_parts.append(f"{field_name} IN ({placeholders})")
            params.extend(ordered_values)

        add_in_filter("document_id", document_ids)
        add_in_filter("article_number", article_numbers)
        add_in_filter("clause_ref", clause_refs)
        ordered_point_refs = dedupe_preserve_order(tuple(value for value in point_refs if value))
        if ordered_point_refs:
            placeholders = ", ".join("?" for _ in ordered_point_refs)
            point_conditions = [f"point_ref IN ({placeholders})"]
            params.extend(ordered_point_refs)
            for point_ref in ordered_point_refs:
                point_conditions.append("point_refs LIKE ?")
                params.append(f"%|{point_ref}|%")
            where_parts.append("(" + " OR ".join(point_conditions) + ")")

        excluded_ids = dedupe_preserve_order(tuple(value for value in exclude_chunk_ids if value))
        if excluded_ids:
            placeholders = ", ".join("?" for _ in excluded_ids)
            where_parts.append(f"chunk_id NOT IN ({placeholders})")
            params.extend(excluded_ids)

        if not where_parts:
            return ()

        params.append(max(1, int(limit)))
        rows = self.sqlite.execute(
            f"""
            SELECT chunk_id, parent_chunk_id, citation_text, text, dense_text, sparse_text, payload_json
            FROM records
            WHERE {" AND ".join(where_parts)}
            ORDER BY
                CASE level
                    WHEN 'article' THEN 0
                    WHEN 'clause' THEN 1
                    WHEN 'point' THEN 2
                    ELSE 3
                END,
                CAST(NULLIF(clause_ref, '') AS INTEGER),
                clause_ref,
                point_ref,
                point_refs,
                chunk_id
            LIMIT ?
            """,
            params,
        ).fetchall()
        return tuple(self.records_from_rows(rows).values())

    def close(self) -> None:
        self.sqlite.close()


class QdrantPayloadRecordStore(RecordStore):
    def __init__(
        self,
        *,
        qdrant_client: Any,
        qdrant_models: Any,
        collection_name: str,
    ) -> None:
        self.qdrant = qdrant_client
        self.models = qdrant_models
        self.collection_name = collection_name

    @staticmethod
    def records_from_qdrant_points(points: Sequence[object]) -> dict[str, RetrievedRecord]:
        records: dict[str, RetrievedRecord] = {}
        for point in points:
            payload = getattr(point, "payload", None)
            if not isinstance(payload, dict):
                continue
            try:
                record = record_from_qdrant_payload(payload)
            except ValueError:
                continue
            records[record.chunk_id] = record
        return records

    def fetch_records(self, chunk_ids: Sequence[str]) -> dict[str, RetrievedRecord]:
        ordered_ids = dedupe_preserve_order(chunk_ids)
        if not ordered_ids:
            return {}

        points = self.qdrant.retrieve(
            collection_name=self.collection_name,
            ids=[make_qdrant_point_id(chunk_id) for chunk_id in ordered_ids],
            with_payload=True,
            with_vectors=False,
        )
        return self.records_from_qdrant_points(points)

    def fetch_records_from_hits(self, hits: Sequence[SearchHit]) -> dict[str, RetrievedRecord]:
        records: dict[str, RetrievedRecord] = {}
        missing_chunk_ids: list[str] = []
        for hit in hits:
            try:
                record = record_from_qdrant_payload(hit.payload)
            except ValueError:
                missing_chunk_ids.append(hit.chunk_id)
                continue
            records[record.chunk_id] = record

        if missing_chunk_ids:
            records.update(self.fetch_records(missing_chunk_ids))
        return records

    def build_reference_payload_filter(
        self,
        *,
        document_ids: Sequence[str] = (),
        article_numbers: Sequence[str] = (),
        clause_refs: Sequence[str] = (),
        point_refs: Sequence[str] = (),
        exclude_chunk_ids: Sequence[str] = (),
    ):
        models = self.models
        must_conditions: list[object] = []
        must_not_conditions: list[object] = []

        def add_match_any(field_name: str, values: Sequence[str]) -> None:
            ordered_values = dedupe_preserve_order(tuple(value for value in values if value))
            if not ordered_values:
                return
            must_conditions.append(
                models.FieldCondition(
                    key=field_name,
                    match=models.MatchAny(any=list(ordered_values)),
                )
            )

        add_match_any("document_id", document_ids)
        add_match_any("article_number", article_numbers)
        add_match_any("clause_ref", clause_refs)
        add_match_any("point_refs", point_refs)

        excluded_ids = dedupe_preserve_order(tuple(value for value in exclude_chunk_ids if value))
        if excluded_ids:
            must_not_conditions.append(
                models.FieldCondition(
                    key="chunk_id",
                    match=models.MatchAny(any=list(excluded_ids)),
                )
            )

        if not must_conditions and not must_not_conditions:
            return None
        return models.Filter(
            must=must_conditions or None,
            must_not=must_not_conditions or None,
        )

    def fetch_records_by_reference(
        self,
        *,
        document_ids: Sequence[str] = (),
        article_numbers: Sequence[str] = (),
        clause_refs: Sequence[str] = (),
        point_refs: Sequence[str] = (),
        exclude_chunk_ids: Sequence[str] = (),
        limit: int = 12,
    ) -> tuple[RetrievedRecord, ...]:
        query_filter = self.build_reference_payload_filter(
            document_ids=document_ids,
            article_numbers=article_numbers,
            clause_refs=clause_refs,
            point_refs=point_refs,
            exclude_chunk_ids=exclude_chunk_ids,
        )
        if query_filter is None:
            return ()

        points, _ = self.qdrant.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=max(1, int(limit)) * 4,
            with_payload=True,
            with_vectors=False,
        )
        records = tuple(self.records_from_qdrant_points(points).values())
        return tuple(sorted(records, key=record_reference_sort_key)[: max(1, int(limit))])


__all__ = [
    "QdrantPayloadRecordStore",
    "RecordStore",
    "SQLiteRecordStore",
]

