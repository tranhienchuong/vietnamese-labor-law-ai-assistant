from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Mapping, Sequence

from .answering import ANSWER_JSON_SCHEMA
from .core.config import load_repo_env, load_settings


load_repo_env()
_SETTINGS = load_settings()

SUPPORTED_PROVIDERS = ("groq",)
DEFAULT_PROVIDER = "groq"
DEFAULT_GROQ_MODEL = _SETTINGS.groq_model
DEFAULT_GROQ_BENCHMARK_JUDGE_MODEL = "openai/gpt-oss-120b"
GROQ_STRICT_JSON_MODELS = frozenset(
    {
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    }
)
DEFAULT_GROQ_RATE_LIMIT_RETRIES = max(0, int(_SETTINGS.groq_rate_limit_retries))
DEFAULT_GROQ_RATE_LIMIT_BACKOFF_SECONDS = max(
    0.0,
    float(_SETTINGS.groq_rate_limit_backoff_seconds),
)
DEFAULT_GROQ_RATE_LIMIT_MAX_SLEEP_SECONDS = max(
    DEFAULT_GROQ_RATE_LIMIT_BACKOFF_SECONDS,
    float(_SETTINGS.groq_rate_limit_max_sleep_seconds),
)
RATE_LIMIT_RETRY_AFTER_RE = re.compile(
    r"try again in\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|s|sec|second|seconds)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LLMResponse:
    provider: str
    model: str
    content: str


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
    if provider_name == "groq":
        return DEFAULT_GROQ_MODEL
    raise ValueError(f"Unsupported provider '{provider}'.")


def default_benchmark_judge_provider() -> str:
    configured = load_settings().benchmark_judge_provider
    return normalize_provider(configured)


def default_benchmark_judge_model(provider: str | None = None) -> str:
    configured = load_settings().benchmark_judge_model.strip()
    provider_name = normalize_provider(provider or default_benchmark_judge_provider())
    if configured:
        return configured
    if provider_name == "groq":
        return DEFAULT_GROQ_BENCHMARK_JUDGE_MODEL
    return default_model_for_provider(provider_name)


def resolve_model_name(provider: str, model: str | None = None) -> str:
    cleaned = str(model or "").strip()
    if cleaned:
        return cleaned
    return default_model_for_provider(provider)


def provider_model_label(provider: str, model: str) -> str:
    return f"{normalize_provider(provider)}:{model}"


def groq_response_format_for_model(
    model: str,
    *,
    json_schema: Mapping[str, object] | None = None,
    schema_name: str = "legal_answer",
) -> dict[str, object]:
    if model in GROQ_STRICT_JSON_MODELS:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": dict(json_schema or ANSWER_JSON_SCHEMA),
            },
        }
    return {"type": "json_object"}


def build_groq_client():
    settings = load_settings()
    api_key = settings.optional_secret_value(settings.groq_api_key)
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    Groq = require_groq_client_class()
    return Groq(api_key=api_key)


def is_groq_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True

    name = exc.__class__.__name__.lower()
    if "ratelimit" in name or "rate_limit" in name:
        return True

    message = str(exc).lower()
    return "rate limit" in message and "429" in message


def is_groq_json_validation_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    message = str(exc).lower()
    if status_code != 400:
        return False
    return "json_validate_failed" in message or "failed to validate json" in message


def groq_rate_limit_sleep_seconds(exc: Exception, attempt: int) -> float:
    message = str(exc)
    match = RATE_LIMIT_RETRY_AFTER_RE.search(message)
    if match:
        value = float(match.group("value"))
        unit = match.group("unit").lower()
        seconds = value / 1000.0 if unit == "ms" else value
    else:
        seconds = DEFAULT_GROQ_RATE_LIMIT_BACKOFF_SECONDS * (2**attempt)

    return min(max(seconds, 0.0), DEFAULT_GROQ_RATE_LIMIT_MAX_SLEEP_SECONDS)


def create_groq_chat_completion_with_retries(
    *,
    client,
    model: str,
    messages: Sequence[Mapping[str, str]],
    temperature: float,
    response_format: Mapping[str, object] | None,
):
    for attempt in range(DEFAULT_GROQ_RATE_LIMIT_RETRIES + 1):
        try:
            request_kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if response_format is not None:
                request_kwargs["response_format"] = response_format
            return client.chat.completions.create(
                **request_kwargs,
            )
        except Exception as exc:
            if not is_groq_rate_limit_error(exc) or attempt >= DEFAULT_GROQ_RATE_LIMIT_RETRIES:
                raise
            time.sleep(groq_rate_limit_sleep_seconds(exc, attempt))


def groq_response_format_fallbacks(
    model: str,
    *,
    json_schema: Mapping[str, object] | None = None,
    schema_name: str = "legal_answer",
) -> tuple[Mapping[str, object] | None, ...]:
    primary = groq_response_format_for_model(
        model,
        json_schema=json_schema,
        schema_name=schema_name,
    )
    if primary.get("type") == "json_schema":
        return (primary, {"type": "json_object"}, None)
    return (primary, None)


def groq_chat_completion(
    *,
    model: str,
    messages: Sequence[Mapping[str, str]],
    temperature: float,
    json_schema: Mapping[str, object] | None,
    json_schema_name: str,
) -> LLMResponse:
    client = build_groq_client()
    response = None
    response_formats = groq_response_format_fallbacks(
        model,
        json_schema=json_schema,
        schema_name=json_schema_name,
    )
    for index, response_format in enumerate(response_formats):
        try:
            response = create_groq_chat_completion_with_retries(
                client=client,
                model=model,
                messages=messages,
                temperature=temperature,
                response_format=response_format,
            )
            break
        except Exception as exc:
            is_last_fallback = index == len(response_formats) - 1
            if not is_groq_json_validation_error(exc) or is_last_fallback:
                raise

    if response is None:
        raise RuntimeError("Groq response unexpectedly missing after fallback attempts.")
    content = response.choices[0].message.content or ""

    return LLMResponse(
        provider="groq",
        model=model,
        content=content,
    )


def chat_completion(
    *,
    provider: str,
    model: str | None,
    messages: Sequence[Mapping[str, str]],
    temperature: float = 0,
    json_schema: Mapping[str, object] | None = None,
    json_schema_name: str = "legal_answer",
) -> LLMResponse:
    provider_name = normalize_provider(provider)
    model_name = resolve_model_name(provider_name, model)
    payload_messages = [dict(message) for message in messages]

    if provider_name == "groq":
        return groq_chat_completion(
            model=model_name,
            messages=payload_messages,
            temperature=temperature,
            json_schema=json_schema,
            json_schema_name=json_schema_name,
        )
    raise ValueError(f"Unsupported provider '{provider}'.")


__all__ = [
    "DEFAULT_GROQ_MODEL",
    "DEFAULT_GROQ_BENCHMARK_JUDGE_MODEL",
    "DEFAULT_GROQ_RATE_LIMIT_BACKOFF_SECONDS",
    "DEFAULT_GROQ_RATE_LIMIT_MAX_SLEEP_SECONDS",
    "DEFAULT_GROQ_RATE_LIMIT_RETRIES",
    "DEFAULT_PROVIDER",
    "GROQ_STRICT_JSON_MODELS",
    "LLMResponse",
    "SUPPORTED_PROVIDERS",
    "build_groq_client",
    "chat_completion",
    "create_groq_chat_completion_with_retries",
    "default_benchmark_judge_model",
    "default_benchmark_judge_provider",
    "default_model_for_provider",
    "groq_chat_completion",
    "groq_rate_limit_sleep_seconds",
    "groq_response_format_fallbacks",
    "groq_response_format_for_model",
    "is_groq_json_validation_error",
    "is_groq_rate_limit_error",
    "normalize_provider",
    "provider_model_label",
    "resolve_model_name",
]
