from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from .answering import build_messages, format_answer_for_user, parse_answer_payload
from .auth_store import AuthStore, AuthUser, user_payload
from .config import load_repo_env
from .llm import DEFAULT_PROVIDER, chat_completion
from .retriever import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_RERANKER_TOP_N,
    HybridRetriever,
    format_context_for_prompt,
    select_contexts_for_prompt,
)


load_repo_env()

app = FastAPI(title="Vietnamese Labor Law AI Assistant")

allow_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_retriever: HybridRetriever | None = None
_retriever_lock = Lock()
_auth_store: AuthStore | None = None
_auth_store_lock = Lock()


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is not None:
        return _retriever

    with _retriever_lock:
        if _retriever is None:
            _retriever = HybridRetriever(
                index_path=Path(os.getenv("INDEX_PATH", "artifacts/index")),
                reranker_model=os.getenv("RERANKER_MODEL", "").strip(),
                reranker_top_n=max(
                    1,
                    int(os.getenv("RERANKER_TOP_N", str(DEFAULT_RERANKER_TOP_N))),
                ),
            )
    return _retriever


def get_auth_store() -> AuthStore:
    global _auth_store
    if _auth_store is not None:
        return _auth_store

    with _auth_store_lock:
        if _auth_store is None:
            _auth_store = AuthStore()
    return _auth_store


def extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        return ""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return ""
    return token.strip()


def require_current_user(authorization: str | None = Header(default=None)) -> AuthUser:
    token = extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required.")

    user = get_auth_store().get_user_by_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")
    return user


def require_admin_user(current_user: AuthUser = Depends(require_current_user)) -> AuthUser:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required.")
    return current_user


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
    answer_payload,
    *,
    question: str = "",
    include_citations: bool = True,
) -> str:
    return format_answer_for_user(
        answer_payload,
        question=question,
        include_citations=include_citations,
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/auth/login")
async def login(request: Request):
    payload = await request.json()
    if not isinstance(payload, dict):
        return JSONResponse({"error": "Request body must be a JSON object."}, status_code=400)

    email = str(payload.get("email") or "").strip().lower()
    password = str(payload.get("password") or "")
    if not email or not password:
        return JSONResponse({"error": "Email and password are required."}, status_code=400)

    store = get_auth_store()
    user = store.authenticate_user(email, password)
    if user is None:
        return JSONResponse({"error": "Invalid email or password."}, status_code=401)

    access_token, expires_at = store.create_session(user)
    return {
        "accessToken": access_token,
        "expiresAt": expires_at,
        "user": user_payload(user),
    }


@app.post("/auth/logout")
def logout(
    authorization: str | None = Header(default=None),
    current_user: AuthUser = Depends(require_current_user),
) -> dict[str, bool]:
    token = extract_bearer_token(authorization)
    if token:
        get_auth_store().revoke_session(token)
    return {"ok": True}


@app.get("/auth/me")
def me(current_user: AuthUser = Depends(require_current_user)) -> dict[str, Any]:
    return {"user": user_payload(current_user)}


@app.get("/conversations")
def list_conversations(current_user: AuthUser = Depends(require_current_user)):
    return {
        "conversations": get_auth_store().list_conversations(user_id=current_user.id)
    }


@app.post("/conversations")
async def create_conversation(
    request: Request,
    current_user: AuthUser = Depends(require_current_user),
):
    payload = await request.json()
    title = str(payload.get("title") or "Cuộc trò chuyện mới")
    conversation = get_auth_store().create_conversation(
        user_id=current_user.id,
        title=title,
    )
    return {"conversation": conversation}


@app.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str,
    current_user: AuthUser = Depends(require_current_user),
):
    store = get_auth_store()
    conversation = store.get_conversation(
        user_id=current_user.id,
        conversation_id=conversation_id,
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    messages = store.list_messages(
        user_id=current_user.id,
        conversation_id=conversation_id,
    )
    return {"conversation": conversation, "messages": messages or []}


@app.get("/conversations/{conversation_id}/messages")
def list_messages(
    conversation_id: str,
    current_user: AuthUser = Depends(require_current_user),
):
    messages = get_auth_store().list_messages(
        user_id=current_user.id,
        conversation_id=conversation_id,
    )
    if messages is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"messages": messages}


@app.get("/admin/stats")
def admin_stats(current_user: AuthUser = Depends(require_admin_user)):
    store = get_auth_store()
    return {
        "user": user_payload(current_user),
        "conversations": len(store.list_conversations(user_id=current_user.id)),
    }


@app.post("/chat")
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
        answer = "Không tìm thấy ngữ cảnh phù hợp trong index."
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


@app.on_event("shutdown")
def shutdown() -> None:
    if _retriever is not None:
        _retriever.close()
