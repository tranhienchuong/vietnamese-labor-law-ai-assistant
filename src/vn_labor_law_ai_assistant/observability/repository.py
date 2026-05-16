from __future__ import annotations

import json
import uuid
from typing import Any

from ..db.sqlite import SQLiteDatabase, utc_timestamp
from .models import ChatTraceCreate


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


class ChatTraceRepository:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def create_trace(self, trace: ChatTraceCreate) -> dict[str, Any]:
        trace_id = str(uuid.uuid4())
        created_at = utc_timestamp()
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_traces (
                    id, request_id, user_id, conversation_id, message_id,
                    question, provider, model, retrieve_only, insufficient_context,
                    latency_ms, retrieval_latency_ms, generation_latency_ms,
                    intent_json, retrieved_hits_json, selected_contexts_json,
                    citations_json, error, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    trace.request_id,
                    trace.user_id,
                    trace.conversation_id,
                    trace.message_id,
                    trace.question,
                    trace.provider,
                    trace.model,
                    1 if trace.retrieve_only else 0,
                    1 if trace.insufficient_context else 0,
                    trace.latency_ms,
                    trace.retrieval_latency_ms,
                    trace.generation_latency_ms,
                    _json_dumps(trace.intent or {}),
                    _json_dumps(trace.retrieved_hits or []),
                    _json_dumps(trace.selected_contexts or []),
                    _json_dumps(trace.citations or {}),
                    trace.error,
                    created_at,
                ),
            )
        stored = self.get_trace(trace_id)
        if stored is None:
            raise RuntimeError("Failed to create chat trace.")
        return stored

    def list_recent_traces(
        self,
        *,
        limit: int = 50,
        user_id: str | None = None,
        conversation_id: str | None = None,
        insufficient_only: bool = False,
        error_only: bool = False,
    ) -> list[dict[str, Any]]:
        where_clauses: list[str] = []
        params: list[Any] = []
        if user_id:
            where_clauses.append("user_id = ?")
            params.append(user_id)
        if conversation_id:
            where_clauses.append("conversation_id = ?")
            params.append(conversation_id)
        if insufficient_only:
            where_clauses.append("insufficient_context = 1")
        if error_only:
            where_clauses.append("error IS NOT NULL AND TRIM(error) <> ''")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        params.append(max(1, min(int(limit), 200)))
        with self.database.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT *
                FROM chat_traces
                {where_sql}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [self._row_to_summary(row) for row in rows]

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM chat_traces WHERE id = ?",
                (trace_id,),
            ).fetchone()
        return self._row_to_detail(row) if row is not None else None

    def count_traces(self) -> int:
        return self._count("SELECT COUNT(*) AS count FROM chat_traces")

    def count_traces_with_errors(self) -> int:
        return self._count(
            """
            SELECT COUNT(*) AS count
            FROM chat_traces
            WHERE error IS NOT NULL AND TRIM(error) <> ''
            """
        )

    def count_insufficient_context_traces(self) -> int:
        return self._count(
            "SELECT COUNT(*) AS count FROM chat_traces WHERE insufficient_context = 1"
        )

    def _count(self, sql: str) -> int:
        with self.database.connect() as connection:
            row = connection.execute(sql).fetchone()
        return int(row["count"]) if row is not None else 0

    def _row_to_summary(self, row: Any) -> dict[str, Any]:
        citations = _json_loads(row["citations_json"], {})
        selected_contexts = _json_loads(row["selected_contexts_json"], [])
        legal_basis = citations.get("legal_basis") if isinstance(citations, dict) else []
        return {
            "id": row["id"],
            "requestId": row["request_id"],
            "userId": row["user_id"],
            "conversationId": row["conversation_id"],
            "messageId": row["message_id"],
            "question": row["question"],
            "provider": row["provider"],
            "model": row["model"],
            "retrieveOnly": bool(row["retrieve_only"]),
            "insufficientContext": bool(row["insufficient_context"]),
            "latencyMs": row["latency_ms"],
            "retrievalLatencyMs": row["retrieval_latency_ms"],
            "generationLatencyMs": row["generation_latency_ms"],
            "citationCount": len(legal_basis) if isinstance(legal_basis, list) else 0,
            "selectedContextCount": len(selected_contexts)
            if isinstance(selected_contexts, list)
            else 0,
            "error": row["error"],
            "createdAt": row["created_at"],
        }

    def _row_to_detail(self, row: Any) -> dict[str, Any]:
        summary = self._row_to_summary(row)
        summary.update(
            {
                "intent": _json_loads(row["intent_json"], {}),
                "retrievedHits": _json_loads(row["retrieved_hits_json"], []),
                "selectedContexts": _json_loads(row["selected_contexts_json"], []),
                "citations": _json_loads(row["citations_json"], {}),
            }
        )
        return summary
