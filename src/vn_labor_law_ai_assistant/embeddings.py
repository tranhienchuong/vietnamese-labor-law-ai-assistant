from __future__ import annotations

import json
import os
from typing import Sequence
from urllib import error, request

from .config import load_repo_env


load_repo_env()

LOCAL_EMBEDDING_PROVIDER = "sentence_transformers"
CUSTOM_HTTP_EMBEDDING_PROVIDER = "custom_http"
EMBEDDING_API_TIMEOUT_SECONDS = 60.0


def embedding_provider() -> str:
    provider = os.getenv("EMBEDDING_PROVIDER", LOCAL_EMBEDDING_PROVIDER).strip().lower()
    if provider in {"", "local"}:
        return LOCAL_EMBEDDING_PROVIDER
    if provider in {LOCAL_EMBEDDING_PROVIDER, CUSTOM_HTTP_EMBEDDING_PROVIDER}:
        return provider
    raise RuntimeError(
        "Unsupported EMBEDDING_PROVIDER. Expected 'sentence_transformers' or 'custom_http'."
    )


def is_custom_http_embedding_provider() -> bool:
    return embedding_provider() == CUSTOM_HTTP_EMBEDDING_PROVIDER


def embedding_api_url() -> str:
    api_url = os.getenv("EMBEDDING_API_URL", "").strip()
    if not api_url:
        raise RuntimeError("EMBEDDING_API_URL is required when EMBEDDING_PROVIDER=custom_http.")
    return api_url


def embedding_api_timeout_seconds() -> float:
    raw_value = os.getenv("EMBEDDING_API_TIMEOUT_SECONDS", "").strip()
    if not raw_value:
        return EMBEDDING_API_TIMEOUT_SECONDS

    try:
        timeout_seconds = float(raw_value)
    except ValueError as exc:
        raise RuntimeError("EMBEDDING_API_TIMEOUT_SECONDS must be a number.") from exc

    if timeout_seconds <= 0:
        raise RuntimeError("EMBEDDING_API_TIMEOUT_SECONDS must be greater than 0.")
    return timeout_seconds


def embed_texts_via_http(
    texts: Sequence[str],
    *,
    api_url: str | None = None,
    timeout_seconds: float | None = None,
    batch_size: int | None = None,
) -> list[list[float]]:
    text_list = [str(text) for text in texts]
    if not text_list:
        return []

    resolved_api_url = api_url or embedding_api_url()
    resolved_timeout = timeout_seconds or embedding_api_timeout_seconds()
    resolved_batch_size = max(1, int(batch_size or len(text_list)))

    vectors: list[list[float]] = []
    for start in range(0, len(text_list), resolved_batch_size):
        batch = text_list[start : start + resolved_batch_size]
        vectors.extend(
            _request_embedding_batch(
                batch,
                api_url=resolved_api_url,
                timeout_seconds=resolved_timeout,
            )
        )
    return vectors


def embed_query_via_http(query: str) -> list[float]:
    vectors = embed_texts_via_http([query], batch_size=1)
    if not vectors:
        raise RuntimeError("Embedding API returned no vector for the query.")
    return vectors[0]


def _request_embedding_batch(
    texts: Sequence[str],
    *,
    api_url: str,
    timeout_seconds: float,
) -> list[list[float]]:
    payload = json.dumps({"input": list(texts)}, ensure_ascii=False).encode("utf-8")
    http_request = request.Request(
        api_url,
        data=payload,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            response_body = response.read()
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        details = details[:500].strip()
        message = f"Embedding API request failed with HTTP {exc.code}."
        if details:
            message = f"{message} Response: {details}"
        raise RuntimeError(message) from exc
    except error.URLError as exc:
        raise RuntimeError(f"Embedding API request failed: {exc.reason}") from exc

    try:
        response_payload = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Embedding API returned invalid JSON.") from exc

    return _extract_embedding_vectors(response_payload, expected_count=len(texts))


def _extract_embedding_vectors(payload: object, *, expected_count: int) -> list[list[float]]:
    if not isinstance(payload, dict):
        raise RuntimeError("Embedding API response must be a JSON object.")

    embedding = payload.get("embedding")
    if embedding is None and isinstance(payload.get("data"), list):
        embedding = [item.get("embedding") for item in payload["data"] if isinstance(item, dict)]

    if embedding is None:
        raise RuntimeError("Embedding API response is missing the 'embedding' field.")

    if expected_count == 1 and _is_vector(embedding):
        return [_coerce_vector(embedding)]

    if isinstance(embedding, list) and all(_is_vector(item) for item in embedding):
        vectors = [_coerce_vector(item) for item in embedding]
        if len(vectors) != expected_count:
            raise RuntimeError(
                f"Embedding API returned {len(vectors)} vectors for {expected_count} inputs."
            )
        return vectors

    raise RuntimeError("Embedding API response has an invalid embedding shape.")


def _is_vector(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
    )


def _coerce_vector(value: object) -> list[float]:
    if not _is_vector(value):
        raise RuntimeError("Embedding vector must be a list of numbers.")
    return [float(item) for item in value]
