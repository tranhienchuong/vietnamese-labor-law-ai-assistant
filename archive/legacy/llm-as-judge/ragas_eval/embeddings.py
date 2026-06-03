from __future__ import annotations

import os
from typing import Sequence

from .config import load_env_files


try:
    from langchain_core.embeddings import Embeddings
except ImportError:  # pragma: no cover - dependency is provided by langchain-openai.

    class Embeddings:  # type: ignore[no-redef]
        pass


CUSTOM_HTTP_PROVIDER = "custom_http"
SENTENCE_TRANSFORMERS_PROVIDER = "sentence_transformers"
DEFAULT_EMBEDDING_DEVICE = "cpu"
DEFAULT_EMBEDDING_BATCH_SIZE = 32


class ProjectRagasEmbeddings(Embeddings):
    """LangChain-compatible embeddings backed by the project's embedding config."""

    def __init__(
        self,
        *,
        provider: str,
        model_name: str = "",
        device: str = DEFAULT_EMBEDDING_DEVICE,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self._model = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embed_texts([text])
        if not vectors:
            raise RuntimeError("RAGAS embedding model returned no vector for query.")
        return vectors[0]

    def _embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        text_list = [str(text) for text in texts]
        if not text_list:
            return []
        if self.provider == CUSTOM_HTTP_PROVIDER:
            return _embed_texts_via_project_http(text_list, batch_size=self.batch_size)
        if self.provider == SENTENCE_TRANSFORMERS_PROVIDER:
            return self._embed_texts_via_sentence_transformers(text_list)
        raise RuntimeError(
            "Unsupported RAGAS embedding provider. Expected 'sentence_transformers' "
            "or 'custom_http'."
        )

    def _embed_texts_via_sentence_transformers(self, texts: Sequence[str]) -> list[list[float]]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required when EMBEDDING_PROVIDER=sentence_transformers."
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)

        vectors = self._model.encode(list(texts), batch_size=self.batch_size)
        return [_vector_to_list(vector) for vector in vectors]


def build_ragas_embeddings() -> ProjectRagasEmbeddings:
    load_env_files()
    provider, model_name = _project_embedding_config()
    return ProjectRagasEmbeddings(
        provider=provider,
        model_name=os.getenv("RAGAS_EMBEDDING_MODEL", "").strip() or model_name,
        device=os.getenv("RAGAS_EMBEDDING_DEVICE", DEFAULT_EMBEDDING_DEVICE).strip()
        or DEFAULT_EMBEDDING_DEVICE,
        batch_size=_env_int("RAGAS_EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE),
    )


def _project_embedding_config() -> tuple[str, str]:
    try:
        from vn_labor_law_ai_assistant.core.config import load_settings
        from vn_labor_law_ai_assistant.embeddings import embedding_provider
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import project embedding config. Run `python -m pip install -e .` "
            "from the repository root before running RAGAS evaluation."
        ) from exc

    settings = load_settings()
    return embedding_provider(), settings.dense_model


def _embed_texts_via_project_http(texts: Sequence[str], *, batch_size: int) -> list[list[float]]:
    try:
        from vn_labor_law_ai_assistant.embeddings import embed_texts_via_http
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import project HTTP embedding client. Run `python -m pip install -e .` "
            "from the repository root before running RAGAS evaluation."
        ) from exc

    return embed_texts_via_http(texts, batch_size=batch_size)


def _vector_to_list(vector: object) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    if not isinstance(vector, list):
        vector = list(vector)  # type: ignore[arg-type]
    return [float(value) for value in vector]


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
