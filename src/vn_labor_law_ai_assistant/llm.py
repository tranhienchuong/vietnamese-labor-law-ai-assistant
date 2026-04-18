from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping, Sequence

from .answering import ANSWER_JSON_SCHEMA
from .config import load_repo_env


load_repo_env()

SUPPORTED_PROVIDERS = ("ollama", "groq")
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_STRICT_JSON_MODELS = frozenset(
    {
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    }
)


@dataclass(frozen=True)
class LLMResponse:
    provider: str
    model: str
    content: str


def require_ollama():
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError("ollama is required for the Ollama provider.") from exc
    return ollama


def require_groq_client_class():
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("groq is required for the Groq provider.") from exc
    return Groq


def normalize_provider(provider: str | None) -> str:
    normalized = str(provider or DEFAULT_PROVIDER).strip().lower()
    if normalized not in SUPPORTED_PROVIDERS:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(f"Unsupported provider '{provider}'. Expected one of: {supported}.")
    return normalized


def default_model_for_provider(provider: str) -> str:
    provider_name = normalize_provider(provider)
    if provider_name == "ollama":
        return DEFAULT_OLLAMA_MODEL
    return DEFAULT_GROQ_MODEL


def resolve_model_name(provider: str, model: str | None = None) -> str:
    cleaned = str(model or "").strip()
    if cleaned:
        return cleaned
    return default_model_for_provider(provider)


def provider_model_label(provider: str, model: str) -> str:
    return f"{normalize_provider(provider)}:{model}"


def groq_response_format_for_model(model: str) -> dict[str, object]:
    if model in GROQ_STRICT_JSON_MODELS:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "legal_answer",
                "strict": True,
                "schema": ANSWER_JSON_SCHEMA,
            },
        }
    return {"type": "json_object"}


def build_groq_client():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    Groq = require_groq_client_class()
    return Groq(api_key=api_key)


def chat_completion(
    *,
    provider: str,
    model: str | None,
    messages: Sequence[Mapping[str, str]],
    temperature: float = 0,
) -> LLMResponse:
    provider_name = normalize_provider(provider)
    model_name = resolve_model_name(provider_name, model)
    payload_messages = [dict(message) for message in messages]

    if provider_name == "ollama":
        ollama = require_ollama()
        response = ollama.chat(
            model=model_name,
            format="json",
            options={"temperature": temperature},
            messages=payload_messages,
        )
        content = response["message"]["content"]
    else:
        client = build_groq_client()
        response = client.chat.completions.create(
            model=model_name,
            messages=payload_messages,
            temperature=temperature,
            response_format=groq_response_format_for_model(model_name),
        )
        content = response.choices[0].message.content or ""

    return LLMResponse(
        provider=provider_name,
        model=model_name,
        content=content,
    )


__all__ = [
    "DEFAULT_GROQ_MODEL",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_PROVIDER",
    "GROQ_STRICT_JSON_MODELS",
    "LLMResponse",
    "SUPPORTED_PROVIDERS",
    "build_groq_client",
    "chat_completion",
    "default_model_for_provider",
    "groq_response_format_for_model",
    "normalize_provider",
    "provider_model_label",
    "resolve_model_name",
]
