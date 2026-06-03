from __future__ import annotations

import logging
import time
from typing import Any
import uuid

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from ...answering import format_answer_for_user, generate_grounded_answer
from ...auth_store import AuthUser
from ...llm import DEFAULT_PROVIDER
from ...observability import ChatTraceService
from ...retriever import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_CONTEXT_TOKENS,
    format_context_for_prompt,
    select_contexts_for_prompt,
)
from ..deps import get_auth_store, get_retriever, require_current_user


router = APIRouter()
LOGGER = logging.getLogger(__name__)


def extract_last_user_message(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ""

    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "").strip()
        if content:
            return content
    return ""


def format_plain_answer(
    answer_payload: Any,
    *,
    question: str = "",
    include_citations: bool = True,
) -> str:
    return format_answer_for_user(
        answer_payload,
        question=question,
        include_citations=include_citations,
    )


def request_id_from_request(request: Request) -> str:
    return request.headers.get("X-Request-Id", "").strip() or str(uuid.uuid4())


def response_headers(*, request_id: str, conversation_id: str | None = None) -> dict[str, str]:
    headers = {"X-Request-Id": request_id}
    if conversation_id:
        headers["X-Conversation-Id"] = conversation_id
    return headers


def elapsed_ms(start: float, end: float | None = None) -> int:
    return int(round(((end or time.perf_counter()) - start) * 1000))


def trace_citations_from_parsed(parsed: Any | None) -> dict[str, Any]:
    if parsed is None:
        return {"legal_basis": [], "evidence_quotes": []}
    return {
        "legal_basis": list(parsed.legal_basis),
        "evidence_quotes": [
            {"citation": quote.citation, "quote": quote.quote}
            for quote in parsed.evidence_quotes
        ],
    }


def record_chat_trace_best_effort(
    *,
    store: Any,
    user_id: str,
    question: str,
    request_id: str,
    conversation_id: str | None,
    message_id: str | None,
    provider: str | None,
    model: str | None,
    retrieve_only: bool,
    insufficient_context: bool,
    total_start: float,
    retrieval_latency_ms: int | None = None,
    generation_latency_ms: int | None = None,
    intent: Any | None = None,
    retrieved_hits: Any | None = None,
    selected_contexts: Any | None = None,
    citations: Any | None = None,
    error: str | None = None,
) -> None:
    try:
        ChatTraceService(store.database).record_chat_trace(
            user_id=user_id,
            question=question,
            request_id=request_id,
            conversation_id=conversation_id,
            message_id=message_id,
            provider=provider,
            model=model,
            retrieve_only=retrieve_only,
            insufficient_context=insufficient_context,
            latency_ms=elapsed_ms(total_start),
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            intent=intent,
            retrieved_hits=retrieved_hits,
            selected_contexts=selected_contexts,
            citations=citations,
            error=error,
        )
    except Exception as exc:
        LOGGER.warning("Failed to record chat trace: %s", exc)


@router.post("/chat")
async def chat(
    request: Request,
    current_user: AuthUser = Depends(require_current_user),
):
    request_id = request_id_from_request(request)
    total_start = time.perf_counter()
    payload = await request.json()
    if not isinstance(payload, dict):
        return JSONResponse(
            {"error": "Request body must be a JSON object."},
            status_code=400,
            headers=response_headers(request_id=request_id),
        )

    question = extract_last_user_message(payload)
    if not question:
        return PlainTextResponse(
            "Cau hoi khong duoc de trong.",
            status_code=400,
            headers=response_headers(request_id=request_id),
        )

    store = get_auth_store()
    try:
        conversation = store.ensure_conversation_for_question(
            user_id=current_user.id,
            conversation_id=str(payload.get("conversationId") or "").strip() or None,
            question=question,
        )
    except PermissionError:
        return JSONResponse(
            {"error": "Conversation not found."},
            status_code=404,
            headers=response_headers(request_id=request_id),
        )

    conversation_id = str(conversation["id"])
    provider = str(payload.get("provider") or DEFAULT_PROVIDER)
    model = str(payload.get("model") or "")
    retrieve_only = bool(payload.get("retrieveOnly"))

    retrieval_start = time.perf_counter()
    try:
        retriever = get_retriever()
        result = retriever.retrieve(
            question,
            top_k=max(1, int(payload.get("topK") or 8)),
            prefetch_limit=max(1, int(payload.get("prefetchLimit") or 24)),
        )
        retrieval_latency_ms = elapsed_ms(retrieval_start)
        contexts = select_contexts_for_prompt(
            result.contexts,
            max_contexts=max(1, int(payload.get("maxContexts") or 6)),
            max_chars=max(1, int(payload.get("maxContextChars") or DEFAULT_MAX_CONTEXT_CHARS)),
            max_tokens=max(1, int(payload.get("maxContextTokens") or DEFAULT_MAX_CONTEXT_TOKENS)),
        )
    except Exception as exc:
        record_chat_trace_best_effort(
            store=store,
            user_id=current_user.id,
            question=question,
            request_id=request_id,
            conversation_id=conversation_id,
            message_id=None,
            provider=provider,
            model=model,
            retrieve_only=retrieve_only,
            insufficient_context=False,
            total_start=total_start,
            retrieval_latency_ms=elapsed_ms(retrieval_start),
            error=str(exc),
        )
        raise

    if not contexts:
        store.append_message(
            conversation_id=conversation_id,
            role="user",
            content=question,
        )
        answer = "Không tìm thấy ngữ cảnh phù hợp trong index."
        assistant_message = store.append_message(
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
        )
        record_chat_trace_best_effort(
            store=store,
            user_id=current_user.id,
            question=question,
            request_id=request_id,
            conversation_id=conversation_id,
            message_id=str(assistant_message["id"]),
            provider=provider,
            model=model,
            retrieve_only=retrieve_only,
            insufficient_context=True,
            total_start=total_start,
            retrieval_latency_ms=retrieval_latency_ms,
            intent=result.intent,
            retrieved_hits=result.hits,
            selected_contexts=(),
            citations={"legal_basis": [], "evidence_quotes": []},
        )
        return PlainTextResponse(
            answer,
            headers=response_headers(request_id=request_id, conversation_id=conversation_id),
            media_type="text/plain; charset=utf-8",
        )

    if retrieve_only:
        record_chat_trace_best_effort(
            store=store,
            user_id=current_user.id,
            question=question,
            request_id=request_id,
            conversation_id=conversation_id,
            message_id=None,
            provider=provider,
            model=model,
            retrieve_only=True,
            insufficient_context=False,
            total_start=total_start,
            retrieval_latency_ms=retrieval_latency_ms,
            intent=result.intent,
            retrieved_hits=result.hits,
            selected_contexts=contexts,
            citations={"legal_basis": [], "evidence_quotes": []},
        )
        return PlainTextResponse(
            format_context_for_prompt(contexts),
            headers=response_headers(request_id=request_id, conversation_id=conversation_id),
            media_type="text/plain; charset=utf-8",
        )

    store.append_message(
        conversation_id=conversation_id,
        role="user",
        content=question,
    )
    generation_start = time.perf_counter()
    try:
        answer_result = generate_grounded_answer(
            question,
            contexts,
            provider=provider,
            model=model,
            temperature=float(payload.get("temperature") or 0),
        )
        parsed = answer_result.parsed
        generation_latency_ms = elapsed_ms(generation_start)
    except Exception as exc:
        record_chat_trace_best_effort(
            store=store,
            user_id=current_user.id,
            question=question,
            request_id=request_id,
            conversation_id=conversation_id,
            message_id=None,
            provider=provider,
            model=model,
            retrieve_only=False,
            insufficient_context=False,
            total_start=total_start,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=elapsed_ms(generation_start),
            intent=result.intent,
            retrieved_hits=result.hits,
            selected_contexts=contexts,
            error=str(exc),
        )
        raise

    answer = format_plain_answer(
        parsed,
        question=question,
        include_citations=bool(payload.get("includeCitations", True)),
    )
    citations = trace_citations_from_parsed(parsed)
    assistant_message = store.append_message(
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        citations=citations,
    )
    record_chat_trace_best_effort(
        store=store,
        user_id=current_user.id,
        question=question,
        request_id=request_id,
        conversation_id=conversation_id,
        message_id=str(assistant_message["id"]),
        provider=answer_result.provider,
        model=answer_result.model,
        retrieve_only=False,
        insufficient_context=bool(parsed.insufficient_context),
        total_start=total_start,
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
        intent=result.intent,
        retrieved_hits=result.hits,
        selected_contexts=contexts,
        citations=citations,
    )
    return PlainTextResponse(
        answer,
        headers=response_headers(request_id=request_id, conversation_id=conversation_id),
        media_type="text/plain; charset=utf-8",
    )
