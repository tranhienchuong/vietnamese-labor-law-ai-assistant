from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = REPO_ROOT / ".env"
DEV_ONLY_AUTH_SECRET = "dev-only-change-me-vietnamese-labor-law-ai"
DEFAULT_USER_PASSWORD_VALUE = "user12345"
DEFAULT_ADMIN_PASSWORD_VALUE = "admin12345"
TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def _strip_optional_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_repo_env(*, override: bool = False) -> Path | None:
    if not ENV_FILE.exists():
        return None

    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        if not override and key in os.environ:
            continue

        os.environ[key] = _strip_optional_quotes(value.strip())

    return ENV_FILE


def _bool_with_default(value: Any, default: bool) -> Any:
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if not cleaned:
            return default
        if cleaned in TRUE_VALUES:
            return True
        if cleaned in FALSE_VALUES:
            return False
        return default
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_env: str = Field(default="development", alias="APP_ENV")
    environment: str = Field(default="", alias="ENVIRONMENT")

    cors_allow_origins: str = Field(default="*", alias="CORS_ALLOW_ORIGINS")

    app_db_path: Path = Field(default=Path("artifacts/app.db"), alias="APP_DB_PATH")

    auth_secret: SecretStr | None = Field(default=None, alias="AUTH_SECRET")
    auth_seed_default_users: bool = Field(default=True, alias="AUTH_SEED_DEFAULT_USERS")
    default_user_name: str = Field(default="Nguoi dung", alias="DEFAULT_USER_NAME")
    default_user_email: str = Field(default="user@example.com", alias="DEFAULT_USER_EMAIL")
    default_user_password: SecretStr = Field(
        default=SecretStr(DEFAULT_USER_PASSWORD_VALUE),
        alias="DEFAULT_USER_PASSWORD",
    )
    default_admin_name: str = Field(default="Quan tri vien", alias="DEFAULT_ADMIN_NAME")
    default_admin_email: str = Field(default="admin@example.com", alias="DEFAULT_ADMIN_EMAIL")
    default_admin_password: SecretStr = Field(
        default=SecretStr(DEFAULT_ADMIN_PASSWORD_VALUE),
        alias="DEFAULT_ADMIN_PASSWORD",
    )

    groq_api_key: SecretStr | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="qwen/qwen3-32b", alias="GROQ_MODEL")
    llm_provider: str = Field(default="groq", alias="LLM_PROVIDER")

    # Legacy exploratory judge settings. Official thesis metrics use deterministic checks.
    benchmark_judge_provider: str = Field(default="groq", alias="BENCHMARK_JUDGE_PROVIDER")
    benchmark_judge_model: str = Field(default="", alias="BENCHMARK_JUDGE_MODEL")
    benchmark_path: Path = Field(
        default=Path("artifacts/evaluation/golden_benchmark_100_extended.jsonl"),
        alias="BENCHMARK_PATH",
    )
    benchmark_metric_mode: str = Field(
        default="deterministic_100_split",
        alias="BENCHMARK_METRIC_MODE",
    )
    citation_validation_mode: str = Field(default="deterministic", alias="CITATION_VALIDATION_MODE")
    eval_citation_match_mode: str = Field(default="containment", alias="EVAL_CITATION_MATCH_MODE")

    qdrant_url: str = Field(default="", alias="QDRANT_URL")
    qdrant_api_key: SecretStr | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(
        default="vietnamese_labor_law_chunks",
        alias="QDRANT_COLLECTION",
    )
    qdrant_timeout: float = Field(default=120.0, alias="QDRANT_TIMEOUT")

    retriever_record_source: str = Field(default="qdrant_payload", alias="RETRIEVER_RECORD_SOURCE")
    index_path: Path = Field(default=Path("artifacts/index"), alias="INDEX_PATH")
    reranker_model: str = Field(default="", alias="RERANKER_MODEL")
    reranker_top_n: int = Field(default=24, alias="RERANKER_TOP_N")
    enable_article_sibling_contexts: bool = Field(
        default=True,
        alias="ENABLE_ARTICLE_SIBLING_CONTEXTS",
    )
    sibling_context_limit: int = Field(default=8, alias="SIBLING_CONTEXT_LIMIT")

    embedding_provider: str = Field(
        default="sentence_transformers",
        alias="EMBEDDING_PROVIDER",
    )
    embedding_api_url: str = Field(default="", alias="EMBEDDING_API_URL")
    embedding_api_token: SecretStr | None = Field(default=None, alias="EMBEDDING_API_TOKEN")
    hf_token: SecretStr | None = Field(default=None, alias="HF_TOKEN")
    huggingface_hub_token: SecretStr | None = Field(default=None, alias="HUGGINGFACE_HUB_TOKEN")
    embedding_api_timeout_seconds: float = Field(
        default=60.0,
        alias="EMBEDDING_API_TIMEOUT_SECONDS",
    )

    query_router_enabled: bool = Field(default=True, alias="QUERY_ROUTER_ENABLED")
    query_router_provider: str = Field(default="", alias="QUERY_ROUTER_PROVIDER")
    query_router_model: str = Field(default="", alias="QUERY_ROUTER_MODEL")
    query_router_fallback_to_heuristic: bool = Field(
        default=True,
        alias="QUERY_ROUTER_FALLBACK_TO_HEURISTIC",
    )

    legal_graph_enabled: bool = Field(default=False, alias="LEGAL_GRAPH_ENABLED")
    legal_graph_backend: str = Field(default="neo4j", alias="LEGAL_GRAPH_BACKEND")
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: SecretStr = Field(default=SecretStr("password"), alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")
    legal_graph_expansion_depth: int = Field(default=2, alias="LEGAL_GRAPH_EXPANSION_DEPTH")
    legal_graph_max_expanded_chunks: int = Field(
        default=12,
        alias="LEGAL_GRAPH_MAX_EXPANDED_CHUNKS",
    )
    legal_graph_min_confidence: float = Field(
        default=0.60,
        alias="LEGAL_GRAPH_MIN_CONFIDENCE",
    )
    legal_graph_complex_query_only: bool = Field(
        default=True,
        alias="LEGAL_GRAPH_COMPLEX_QUERY_ONLY",
    )
    legal_graph_trace: bool = Field(default=False, alias="LEGAL_GRAPH_TRACE")

    groq_rate_limit_retries: int = Field(default=6, alias="GROQ_RATE_LIMIT_RETRIES")
    groq_rate_limit_backoff_seconds: float = Field(
        default=1.0,
        alias="GROQ_RATE_LIMIT_BACKOFF_SECONDS",
    )
    groq_rate_limit_max_sleep_seconds: float = Field(
        default=10.0,
        alias="GROQ_RATE_LIMIT_MAX_SLEEP_SECONDS",
    )

    dense_model: str = Field(
        default="keepitreal/vietnamese-sbert",
        alias="DENSE_MODEL",
    )

    @field_validator("auth_seed_default_users", mode="before")
    @classmethod
    def _validate_auth_seed_default_users(cls, value: Any) -> Any:
        return _bool_with_default(value, True)

    @field_validator("query_router_enabled", mode="before")
    @classmethod
    def _validate_query_router_enabled(cls, value: Any) -> Any:
        return _bool_with_default(value, True)

    @field_validator("query_router_fallback_to_heuristic", mode="before")
    @classmethod
    def _validate_query_router_fallback_to_heuristic(cls, value: Any) -> Any:
        return _bool_with_default(value, True)

    @field_validator("enable_article_sibling_contexts", mode="before")
    @classmethod
    def _validate_enable_article_sibling_contexts(cls, value: Any) -> Any:
        return _bool_with_default(value, True)

    @field_validator("legal_graph_enabled", mode="before")
    @classmethod
    def _validate_legal_graph_enabled(cls, value: Any) -> Any:
        return _bool_with_default(value, False)

    @field_validator("legal_graph_complex_query_only", mode="before")
    @classmethod
    def _validate_legal_graph_complex_query_only(cls, value: Any) -> Any:
        return _bool_with_default(value, True)

    @field_validator("legal_graph_trace", mode="before")
    @classmethod
    def _validate_legal_graph_trace(cls, value: Any) -> Any:
        return _bool_with_default(value, False)

    @property
    def is_production(self) -> bool:
        return (
            self.app_env.strip().lower() == "production"
            or self.environment.strip().lower() == "production"
        )

    def require_auth_secret(self) -> str:
        secret = self.optional_secret_value(self.auth_secret)
        if not secret:
            if self.is_production:
                raise RuntimeError("AUTH_SECRET is required in production.")
            return DEV_ONLY_AUTH_SECRET
        return secret

    def validate_auth_seed_configuration(self) -> None:
        if not self.is_production or not self.auth_seed_default_users:
            return

        default_passwords = {
            DEFAULT_USER_PASSWORD_VALUE,
            DEFAULT_ADMIN_PASSWORD_VALUE,
        }
        seeded_passwords = {
            self.default_user_password.get_secret_value(),
            self.default_admin_password.get_secret_value(),
        }
        if default_passwords.intersection(seeded_passwords):
            raise RuntimeError(
                "Default seeded users cannot use default passwords in production."
            )

    def cors_origins_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_allow_origins.split(",")
            if origin.strip()
        ] or ["*"]

    def optional_secret_value(self, value: SecretStr | None) -> str:
        return value.get_secret_value().strip() if value is not None else ""

    def embedding_api_token_value(self) -> str:
        for token in (
            self.embedding_api_token,
            self.hf_token,
            self.huggingface_hub_token,
        ):
            value = self.optional_secret_value(token)
            if value:
                return value
        return ""

    def qdrant_collection_was_configured(self) -> bool:
        return "qdrant_collection" in self.model_fields_set

    def field_was_configured(self, field_name: str) -> bool:
        return field_name in self.model_fields_set

    def public_runtime_config(
        self,
        *,
        qdrant_collection: str | None = None,
        retriever_record_source: str | None = None,
    ) -> dict[str, object]:
        return {
            "appEnv": self.app_env,
            "databasePath": str(self.app_db_path),
            "qdrantCollection": qdrant_collection or self.qdrant_collection,
            "retrieverRecordSource": retriever_record_source or self.retriever_record_source,
            "indexPath": str(self.index_path),
            "rerankerEnabled": bool(self.reranker_model.strip()),
            "queryRouterEnabled": self.query_router_enabled,
            "llmProvider": self.llm_provider,
            "groqModel": self.groq_model,
            "benchmarkPath": str(self.benchmark_path),
            "benchmarkMetricMode": self.benchmark_metric_mode,
            "citationValidationMode": self.citation_validation_mode,
        }

    def public_retrieval_config(
        self,
        *,
        qdrant_collection: str | None = None,
        retriever_record_source: str | None = None,
        dense_model: str | None = None,
    ) -> dict[str, object]:
        return {
            "qdrantCollection": qdrant_collection or self.qdrant_collection,
            "retrieverRecordSource": retriever_record_source or self.retriever_record_source,
            "indexPath": str(self.index_path),
            "rerankerModel": self.reranker_model,
            "rerankerEnabled": bool(self.reranker_model.strip()),
            "rerankerTopN": self.reranker_top_n,
            "articleSiblingContextsEnabled": self.enable_article_sibling_contexts,
            "siblingContextLimit": self.sibling_context_limit,
            "qdrantTimeout": self.qdrant_timeout,
            "queryRouterEnabled": self.query_router_enabled,
            "queryRouterProvider": self.query_router_provider,
            "queryRouterModel": self.query_router_model,
            "queryRouterFallbackToHeuristic": self.query_router_fallback_to_heuristic,
            "embeddingProvider": self.embedding_provider,
            "denseModel": dense_model or self.dense_model,
            "benchmarkPath": str(self.benchmark_path),
            "benchmarkMetricMode": self.benchmark_metric_mode,
            "citationValidationMode": self.citation_validation_mode,
            "legalGraphEnabled": self.legal_graph_enabled,
            "legalGraphBackend": self.legal_graph_backend,
            "legalGraphExpansionDepth": self.legal_graph_expansion_depth,
            "legalGraphMaxExpandedChunks": self.legal_graph_max_expanded_chunks,
        }


def load_settings() -> Settings:
    return Settings(_env_file=None)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


__all__ = [
    "DEV_ONLY_AUTH_SECRET",
    "DEFAULT_ADMIN_PASSWORD_VALUE",
    "DEFAULT_USER_PASSWORD_VALUE",
    "ENV_FILE",
    "REPO_ROOT",
    "Settings",
    "get_settings",
    "load_repo_env",
    "load_settings",
]
