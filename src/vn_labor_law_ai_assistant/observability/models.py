from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatTraceCreate:
    user_id: str
    question: str
    request_id: str | None = None
    conversation_id: str | None = None
    message_id: str | None = None
    provider: str | None = None
    model: str | None = None
    retrieve_only: bool = False
    insufficient_context: bool = False
    latency_ms: int | None = None
    retrieval_latency_ms: int | None = None
    generation_latency_ms: int | None = None
    intent: dict[str, Any] | None = None
    retrieved_hits: list[dict[str, Any]] | None = None
    selected_contexts: list[dict[str, Any]] | None = None
    citations: dict[str, Any] | None = None
    error: str | None = None
