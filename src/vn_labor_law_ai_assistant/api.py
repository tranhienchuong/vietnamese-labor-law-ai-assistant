from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from .answering import build_messages, parse_answer_payload
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


def format_plain_answer(answer_payload, *, include_citations: bool = True) -> str:
    parts = [answer_payload.answer or "Khong co noi dung tra loi."]
    if include_citations:
        parts.append("")
        parts.append("Co so phap ly:")
        if answer_payload.legal_basis:
            parts.extend(f"- {citation}" for citation in answer_payload.legal_basis)
        else:
            parts.append("- Khong co co so phap ly hop le duoc xac nhan.")
    if answer_payload.notes:
        parts.append("")
        parts.append(f"Ghi chu: {answer_payload.notes}")
    return "\n".join(parts).strip()


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: Request):
    payload = await request.json()
    if not isinstance(payload, dict):
        return JSONResponse({"error": "Request body must be a JSON object."}, status_code=400)

    question = extract_last_user_message(payload)
    if not question:
        return PlainTextResponse("Cau hoi khong duoc de trong.", status_code=400)

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
        return PlainTextResponse(
            "Khong tim thay ngu canh phu hop trong index.",
            media_type="text/plain; charset=utf-8",
        )

    if payload.get("retrieveOnly"):
        return PlainTextResponse(
            format_context_for_prompt(contexts),
            media_type="text/plain; charset=utf-8",
        )

    response = chat_completion(
        provider=str(payload.get("provider") or DEFAULT_PROVIDER),
        model=str(payload.get("model") or ""),
        messages=build_messages(question, contexts),
        temperature=float(payload.get("temperature") or 0),
    )
    parsed = parse_answer_payload(response.content, contexts, question=question)
    return PlainTextResponse(
        format_plain_answer(
            parsed,
            include_citations=bool(payload.get("includeCitations", True)),
        ),
        media_type="text/plain; charset=utf-8",
    )


@app.on_event("shutdown")
def shutdown() -> None:
    if _retriever is not None:
        _retriever.close()
