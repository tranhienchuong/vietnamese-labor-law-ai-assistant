from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from ...answering import build_messages, format_answer_for_user, parse_answer_payload
from ...auth_store import AuthUser
from ...llm import DEFAULT_PROVIDER, chat_completion
from ...retriever import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_CONTEXT_TOKENS,
    format_context_for_prompt,
    select_contexts_for_prompt,
)
from ..deps import get_auth_store, get_retriever, require_current_user


router = APIRouter()


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


@router.post("/chat")
async def chat(
    request: Request,
    current_user: AuthUser = Depends(require_current_user),
):
    payload = await request.json()
    if not isinstance(payload, dict):
        return JSONResponse({"error": "Request body must be a JSON object."}, status_code=400)

    question = extract_last_user_message(payload)
    if not question:
        return PlainTextResponse("Cau hoi khong duoc de trong.", status_code=400)

    store = get_auth_store()
    try:
        conversation = store.ensure_conversation_for_question(
            user_id=current_user.id,
            conversation_id=str(payload.get("conversationId") or "").strip() or None,
            question=question,
        )
    except PermissionError:
        return JSONResponse({"error": "Conversation not found."}, status_code=404)

    conversation_id = str(conversation["id"])

    retriever = get_retriever()
    result = retriever.retrieve(
        question,
        top_k=max(1, int(payload.get("topK") or 8)),
        prefetch_limit=max(1, int(payload.get("prefetchLimit") or 24)),
    )
    contexts = select_contexts_for_prompt(
        result.contexts,
        max_contexts=max(1, int(payload.get("maxContexts") or 6)),
        max_chars=max(1, int(payload.get("maxContextChars") or DEFAULT_MAX_CONTEXT_CHARS)),
        max_tokens=max(1, int(payload.get("maxContextTokens") or DEFAULT_MAX_CONTEXT_TOKENS)),
    )
    if not contexts:
        store.append_message(
            conversation_id=conversation_id,
            role="user",
            content=question,
        )
        answer = "KhÃ´ng tÃ¬m tháº¥y ngá»¯ cáº£nh phÃ¹ há»£p trong index."
        store.append_message(
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
        )
        return PlainTextResponse(
            answer,
            headers={"X-Conversation-Id": conversation_id},
            media_type="text/plain; charset=utf-8",
        )

    if payload.get("retrieveOnly"):
        return PlainTextResponse(
            format_context_for_prompt(contexts),
            headers={"X-Conversation-Id": conversation_id},
            media_type="text/plain; charset=utf-8",
        )

    store.append_message(
        conversation_id=conversation_id,
        role="user",
        content=question,
    )
    response = chat_completion(
        provider=str(payload.get("provider") or DEFAULT_PROVIDER),
        model=str(payload.get("model") or ""),
        messages=build_messages(question, contexts),
        temperature=float(payload.get("temperature") or 0),
    )
    parsed = parse_answer_payload(response.content, contexts, question=question)
    answer = format_plain_answer(
        parsed,
        question=question,
        include_citations=bool(payload.get("includeCitations", True)),
    )
    store.append_message(
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        citations={
            "legal_basis": list(parsed.legal_basis),
            "evidence_quotes": [
                {"citation": quote.citation, "quote": quote.quote}
                for quote in parsed.evidence_quotes
            ],
        },
    )
    return PlainTextResponse(
        answer,
        headers={"X-Conversation-Id": conversation_id},
        media_type="text/plain; charset=utf-8",
    )
