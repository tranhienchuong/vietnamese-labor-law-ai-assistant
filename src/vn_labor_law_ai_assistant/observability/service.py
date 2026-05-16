from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Iterable

from ..db.sqlite import SQLiteDatabase
from .models import ChatTraceCreate
from .repository import ChatTraceRepository


QUESTION_LIMIT = 4000
TEXT_PREVIEW_LIMIT = 1200
ERROR_LIMIT = 500
MAX_RETRIEVED_HITS = 80
MAX_SELECTED_CONTEXTS = 20


class ChatTraceService:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.repository = ChatTraceRepository(database)

    def record_chat_trace(
        self,
        *,
        user_id: str,
        question: str,
        request_id: str | None = None,
        conversation_id: str | None = None,
        message_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        retrieve_only: bool = False,
        insufficient_context: bool = False,
        latency_ms: int | None = None,
        retrieval_latency_ms: int | None = None,
        generation_latency_ms: int | None = None,
        intent: Any | None = None,
        retrieved_hits: Iterable[Any] | None = None,
        selected_contexts: Iterable[Any] | None = None,
        citations: Any | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        return self.repository.create_trace(
            ChatTraceCreate(
                request_id=_clean_optional_string(request_id, 120),
                user_id=user_id,
                conversation_id=_clean_optional_string(conversation_id, 120),
                message_id=_clean_optional_string(message_id, 120),
                question=_truncate(question, QUESTION_LIMIT),
                provider=_clean_optional_string(provider, 80),
                model=_clean_optional_string(model, 160),
                retrieve_only=retrieve_only,
                insufficient_context=insufficient_context,
                latency_ms=latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                intent=_sanitize_intent(intent),
                retrieved_hits=_sanitize_retrieved_hits(retrieved_hits),
                selected_contexts=_sanitize_selected_contexts(selected_contexts),
                citations=_sanitize_citations(citations),
                error=_clean_optional_string(error, ERROR_LIMIT),
            )
        )

    def list_recent_traces(
        self,
        *,
        limit: int = 50,
        user_id: str | None = None,
        conversation_id: str | None = None,
        insufficient_only: bool = False,
        error_only: bool = False,
    ) -> list[dict[str, Any]]:
        return self.repository.list_recent_traces(
            limit=limit,
            user_id=user_id,
            conversation_id=conversation_id,
            insufficient_only=insufficient_only,
            error_only=error_only,
        )

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        return self.repository.get_trace(trace_id)

    def count_traces(self) -> int:
        return self.repository.count_traces()

    def count_traces_with_errors(self) -> int:
        return self.repository.count_traces_with_errors()

    def count_insufficient_context_traces(self) -> int:
        return self.repository.count_insufficient_context_traces()


def _sanitize_intent(intent: Any | None) -> dict[str, Any]:
    if intent is None:
        return {}
    if isinstance(intent, dict):
        return _jsonable(intent)

    fields = (
        "raw_query",
        "normalized_query",
        "actor_filters",
        "topic_filters",
        "issue_filters",
        "document_filters",
        "article_numbers",
        "inferred_article_numbers",
        "force_reference_article_numbers",
        "clause_refs",
        "point_refs",
        "query_expansions",
        "query_types",
        "matched_direct_reference_rules",
        "forced_references",
    )
    payload: dict[str, Any] = {}
    for field in fields:
        if hasattr(intent, field):
            payload[field] = _jsonable(getattr(intent, field))
    return payload


def _sanitize_retrieved_hits(hits: Iterable[Any] | None) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for hit in list(hits or ())[:MAX_RETRIEVED_HITS]:
        payload = _object_payload(hit)
        sanitized.append(
            {
                "chunkId": _clean_string(getattr(hit, "chunk_id", payload.get("chunk_id", ""))),
                "score": _float_or_none(getattr(hit, "score", None)),
                "citationText": _clean_string(
                    getattr(hit, "citation_text", payload.get("citation_text", ""))
                ),
                "documentId": _clean_string(payload.get("document_id", "")),
                "articleNumber": _clean_string(payload.get("article_number", "")),
                "clauseRef": _clean_string(payload.get("clause_ref", "")),
                "pointRefs": _string_list(
                    payload.get("point_refs") or payload.get("point_ref") or []
                ),
                "issueType": _string_list(
                    payload.get("issue_type") or payload.get("issue_types") or []
                ),
                "topic": _string_list(payload.get("topic") or payload.get("topics") or []),
                "forcedReference": bool(payload.get("retrieval_forced_reference")),
            }
        )
    return sanitized


def _sanitize_selected_contexts(contexts: Iterable[Any] | None) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for context in list(contexts or ())[:MAX_SELECTED_CONTEXTS]:
        payload = _object_payload(context)
        sanitized.append(
            {
                "chunkId": _clean_string(getattr(context, "chunk_id", payload.get("chunk_id", ""))),
                "citationText": _clean_string(
                    getattr(context, "citation_text", payload.get("citation_text", ""))
                ),
                "score": _float_or_none(getattr(context, "score", None)),
                "matchedChunkIds": _string_list(getattr(context, "matched_chunk_ids", ())),
                "matchedCitations": _string_list(getattr(context, "matched_citations", ())),
                "textPreview": _truncate(str(getattr(context, "text", "") or ""), TEXT_PREVIEW_LIMIT),
            }
        )
    return sanitized


def _sanitize_citations(citations: Any | None) -> dict[str, Any]:
    if not isinstance(citations, dict):
        return {"legal_basis": [], "evidence_quotes": []}

    evidence_quotes: list[dict[str, str]] = []
    for item in citations.get("evidence_quotes") or []:
        if not isinstance(item, dict):
            continue
        evidence_quotes.append(
            {
                "citation": _truncate(str(item.get("citation") or ""), 500),
                "quote": _truncate(str(item.get("quote") or ""), TEXT_PREVIEW_LIMIT),
            }
        )
    return {
        "legal_basis": [
            _truncate(str(value), 500)
            for value in citations.get("legal_basis") or []
            if str(value or "").strip()
        ],
        "evidence_quotes": evidence_quotes,
    }


def _object_payload(value: Any) -> dict[str, Any]:
    payload = getattr(value, "payload", {})
    return payload if isinstance(payload, dict) else {}


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = []
    return [_truncate(str(item), 250) for item in values if str(item or "").strip()]


def _clean_string(value: Any) -> str:
    return str(value or "").strip()


def _clean_optional_string(value: Any, limit: int) -> str | None:
    cleaned = _truncate(str(value or "").strip(), limit)
    return cleaned or None


def _truncate(value: str, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
