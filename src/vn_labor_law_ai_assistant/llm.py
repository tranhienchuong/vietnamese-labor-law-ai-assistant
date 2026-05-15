from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import time
from typing import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from .answering import ANSWER_JSON_SCHEMA
from .config import load_repo_env


load_repo_env()

SUPPORTED_PROVIDERS = ("groq", "azure_openai")
DEFAULT_PROVIDER = "groq"
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
DEFAULT_GROQ_BENCHMARK_JUDGE_MODEL = "openai/gpt-oss-120b"
DEFAULT_AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "GPT-5.4-MINI")
GROQ_STRICT_JSON_MODELS = frozenset(
    {
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    }
)
DEFAULT_AZURE_OPENAI_TIMEOUT_SECONDS = max(
    1.0,
    float(os.getenv("AZURE_OPENAI_TIMEOUT_SECONDS", "120.0")),
)
DEFAULT_GROQ_RATE_LIMIT_RETRIES = max(0, int(os.getenv("GROQ_RATE_LIMIT_RETRIES", "6")))
DEFAULT_GROQ_RATE_LIMIT_BACKOFF_SECONDS = max(
    0.0,
    float(os.getenv("GROQ_RATE_LIMIT_BACKOFF_SECONDS", "1.0")),
)
DEFAULT_GROQ_RATE_LIMIT_MAX_SLEEP_SECONDS = max(
    DEFAULT_GROQ_RATE_LIMIT_BACKOFF_SECONDS,
    float(os.getenv("GROQ_RATE_LIMIT_MAX_SLEEP_SECONDS", "10.0")),
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


class LLMProviderHTTPError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


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
    if provider_name == "azure_openai":
        return DEFAULT_AZURE_OPENAI_MODEL
    raise ValueError(f"Unsupported provider '{provider}'.")


def default_benchmark_judge_provider() -> str:
    configured = os.getenv("BENCHMARK_JUDGE_PROVIDER", "groq")
    return normalize_provider(configured)


def default_benchmark_judge_model(provider: str | None = None) -> str:
    configured = os.getenv("BENCHMARK_JUDGE_MODEL", "").strip()
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
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    Groq = require_groq_client_class()
    return Groq(api_key=api_key)


def build_azure_openai_responses_endpoint() -> str:
    endpoint = os.getenv("AZURE_OPENAI_RESPONSES_ENDPOINT", "").strip()
    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_RESPONSES_ENDPOINT is not set.")

    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError("AZURE_OPENAI_RESPONSES_ENDPOINT must be an absolute URL.")

    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
    if api_version and "api-version=" not in endpoint.lower():
        separator = "&" if parsed.query else "?"
        endpoint = f"{endpoint}{separator}api-version={quote(api_version)}"
    return endpoint


def require_azure_openai_api_key() -> str:
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("AZURE_OPENAI_API_KEY is not set.")
    return api_key


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


def is_azure_openai_rate_limit_error(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 429


def is_azure_openai_text_format_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code != 400:
        return False
    message = f"{exc} {getattr(exc, 'response_body', '')}".lower()
    return any(
        token in message
        for token in (
            "json_schema",
            "json schema",
            "text.format",
            "response_format",
            "schema",
            "structured",
        )
    )


def is_azure_openai_content_filter_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code != 400:
        return False
    message = f"{exc} {getattr(exc, 'response_body', '')}".lower()
    return (
        "content_filter" in message
        or "content management policy" in message
        or "content filtering" in message
    )


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


def azure_openai_rate_limit_sleep_seconds(exc: Exception, attempt: int) -> float:
    return groq_rate_limit_sleep_seconds(exc, attempt)


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


def azure_openai_text_format_for_response(
    *,
    json_schema: Mapping[str, object] | None = None,
    schema_name: str = "legal_answer",
) -> dict[str, object]:
    return {
        "type": "json_schema",
        "name": schema_name,
        "strict": True,
        "schema": dict(json_schema or ANSWER_JSON_SCHEMA),
    }


def azure_openai_text_format_fallbacks(
    *,
    json_schema: Mapping[str, object] | None = None,
    schema_name: str = "legal_answer",
) -> tuple[Mapping[str, object] | None, ...]:
    return (
        azure_openai_text_format_for_response(
            json_schema=json_schema,
            schema_name=schema_name,
        ),
        {"type": "json_object"},
        None,
    )


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


def build_azure_openai_payload(
    *,
    model: str,
    messages: Sequence[Mapping[str, str]],
    temperature: float,
    text_format: Mapping[str, object] | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": model,
        "input": [dict(message) for message in messages],
        "temperature": temperature,
    }
    if text_format is not None:
        payload["text"] = {"format": dict(text_format)}
    return payload


def post_azure_openai_responses_payload(
    *,
    endpoint: str,
    api_key: str,
    payload: Mapping[str, object],
    timeout: float = DEFAULT_AZURE_OPENAI_TIMEOUT_SECONDS,
) -> dict[str, object]:
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "api-key": api_key,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        raise LLMProviderHTTPError(
            f"Azure OpenAI request failed with status {exc.code}: {response_body}",
            status_code=exc.code,
            response_body=response_body,
        ) from exc
    except URLError as exc:
        raise LLMProviderHTTPError(f"Azure OpenAI request failed: {exc}") from exc

    try:
        decoded = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise LLMProviderHTTPError(
            f"Azure OpenAI response was not valid JSON: {response_body[:500]}"
        ) from exc

    if not isinstance(decoded, dict):
        raise LLMProviderHTTPError("Azure OpenAI response JSON was not an object.")
    return decoded


def extract_azure_openai_response_content(payload: Mapping[str, object]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text

    parts: list[str] = []
    output_items = payload.get("output")
    if isinstance(output_items, list):
        for output_item in output_items:
            if not isinstance(output_item, dict):
                continue
            item_text = output_item.get("text")
            if isinstance(item_text, str):
                parts.append(item_text)
            content_items = output_item.get("content")
            if not isinstance(content_items, list):
                continue
            for content_item in content_items:
                if isinstance(content_item, str):
                    parts.append(content_item)
                elif isinstance(content_item, dict):
                    content_text = content_item.get("text")
                    if isinstance(content_text, str):
                        parts.append(content_text)

    if parts:
        return "\n".join(part for part in parts if part).strip()

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return str(message["content"])
            if isinstance(first_choice.get("text"), str):
                return str(first_choice["text"])

    raise LLMProviderHTTPError("Azure OpenAI response did not contain output text.")


def create_azure_openai_response_with_retries(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    messages: Sequence[Mapping[str, str]],
    temperature: float,
    text_format: Mapping[str, object] | None,
) -> dict[str, object]:
    payload = build_azure_openai_payload(
        model=model,
        messages=messages,
        temperature=temperature,
        text_format=text_format,
    )
    for attempt in range(DEFAULT_GROQ_RATE_LIMIT_RETRIES + 1):
        try:
            return post_azure_openai_responses_payload(
                endpoint=endpoint,
                api_key=api_key,
                payload=payload,
            )
        except Exception as exc:
            if (
                not is_azure_openai_rate_limit_error(exc)
                or attempt >= DEFAULT_GROQ_RATE_LIMIT_RETRIES
            ):
                raise
            time.sleep(azure_openai_rate_limit_sleep_seconds(exc, attempt))
    raise RuntimeError("Azure OpenAI response unexpectedly missing after retry attempts.")


def azure_openai_response_completion(
    *,
    model: str,
    messages: Sequence[Mapping[str, str]],
    temperature: float,
    json_schema: Mapping[str, object] | None,
    json_schema_name: str,
) -> LLMResponse:
    endpoint = build_azure_openai_responses_endpoint()
    api_key = require_azure_openai_api_key()
    response_payload = None
    text_formats = azure_openai_text_format_fallbacks(
        json_schema=json_schema,
        schema_name=json_schema_name,
    )
    for index, text_format in enumerate(text_formats):
        try:
            response_payload = create_azure_openai_response_with_retries(
                endpoint=endpoint,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                text_format=text_format,
            )
            break
        except Exception as exc:
            is_last_fallback = index == len(text_formats) - 1
            if not is_azure_openai_text_format_error(exc) or is_last_fallback:
                raise

    if response_payload is None:
        raise RuntimeError("Azure OpenAI response unexpectedly missing after fallback attempts.")

    return LLMResponse(
        provider="azure_openai",
        model=model,
        content=extract_azure_openai_response_content(response_payload),
    )


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
    if provider_name == "azure_openai":
        return azure_openai_response_completion(
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
    "DEFAULT_AZURE_OPENAI_MODEL",
    "DEFAULT_AZURE_OPENAI_TIMEOUT_SECONDS",
    "DEFAULT_PROVIDER",
    "GROQ_STRICT_JSON_MODELS",
    "LLMProviderHTTPError",
    "LLMResponse",
    "SUPPORTED_PROVIDERS",
    "azure_openai_response_completion",
    "azure_openai_rate_limit_sleep_seconds",
    "azure_openai_text_format_fallbacks",
    "azure_openai_text_format_for_response",
    "build_azure_openai_payload",
    "build_azure_openai_responses_endpoint",
    "build_groq_client",
    "chat_completion",
    "create_azure_openai_response_with_retries",
    "create_groq_chat_completion_with_retries",
    "default_benchmark_judge_model",
    "default_benchmark_judge_provider",
    "default_model_for_provider",
    "extract_azure_openai_response_content",
    "groq_chat_completion",
    "groq_rate_limit_sleep_seconds",
    "groq_response_format_fallbacks",
    "groq_response_format_for_model",
    "is_azure_openai_rate_limit_error",
    "is_azure_openai_content_filter_error",
    "is_azure_openai_text_format_error",
    "is_groq_json_validation_error",
    "is_groq_rate_limit_error",
    "normalize_provider",
    "post_azure_openai_responses_payload",
    "require_azure_openai_api_key",
    "provider_model_label",
    "resolve_model_name",
]
