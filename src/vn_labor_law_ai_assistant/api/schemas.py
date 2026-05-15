from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LoginRequest(BaseModel):
    email: str
    password: str


class ConversationCreateRequest(BaseModel):
    title: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    messages: list[ChatMessage]
    conversation_id: str | None = Field(default=None, alias="conversationId")
    provider: str | None = None
    model: str | None = None
    top_k: int | None = Field(default=None, alias="topK")
    prefetch_limit: int | None = Field(default=None, alias="prefetchLimit")
    max_contexts: int | None = Field(default=None, alias="maxContexts")
    max_context_chars: int | None = Field(default=None, alias="maxContextChars")
    max_context_tokens: int | None = Field(default=None, alias="maxContextTokens")
    temperature: float | None = None
    include_citations: bool = Field(default=True, alias="includeCitations")
    retrieve_only: bool = Field(default=False, alias="retrieveOnly")
