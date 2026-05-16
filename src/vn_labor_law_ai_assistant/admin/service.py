from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..auth.models import AuthUser
from ..auth.repository import AuthRepository
from ..auth.service import user_payload
from ..auth_store import AuthStore
from ..conversations.repository import ConversationRepository
from ..core.config import Settings, get_settings
from ..observability import ChatTraceService


class AdminService:
    def __init__(self, store: AuthStore, settings: Settings | None = None) -> None:
        self.store = store
        self.settings = settings or get_settings()
        self.auth_repository = AuthRepository(store.database)
        self.conversation_repository = ConversationRepository(store.database)
        self.trace_service = ChatTraceService(store.database)

    def get_stats(self, current_user: AuthUser) -> dict[str, Any]:
        return {
            "user": user_payload(current_user),
            "stats": {
                "totalUsers": self.auth_repository.count_users(),
                "activeUsers": self.auth_repository.count_active_users(),
                "adminUsers": self.auth_repository.count_admin_users(),
                "totalConversations": self.conversation_repository.count_conversations(),
                "totalMessages": self.conversation_repository.count_messages(),
                "activeSessions": self.auth_repository.count_active_sessions(),
                "totalTraces": self.trace_service.count_traces(),
                "tracesWithErrors": self.trace_service.count_traces_with_errors(),
                "insufficientContextTraces": self.trace_service.count_insufficient_context_traces(),
            },
            "runtime": self.settings.public_runtime_config(
                qdrant_collection=self._effective_qdrant_collection(),
                retriever_record_source=self._effective_retriever_record_source(),
            ),
        }

    def get_health(self) -> dict[str, Any]:
        checks = {
            "database": self._database_check(),
            "settings": self._settings_check(),
            "index": self._index_check(),
            "qdrantConfig": self._qdrant_config_check(),
            "llmConfig": self._llm_config_check(),
        }
        status = "ok" if all(self._check_is_healthy(check) for check in checks.values()) else "degraded"
        return {"status": status, "checks": checks}

    def get_retrieval_config(self) -> dict[str, Any]:
        return self.settings.public_retrieval_config(
            qdrant_collection=self._effective_qdrant_collection(),
            retriever_record_source=self._effective_retriever_record_source(),
            dense_model=self._effective_dense_model(),
        )

    def list_recent_traces(
        self,
        *,
        limit: int = 50,
        user_id: str | None = None,
        conversation_id: str | None = None,
        insufficient_only: bool = False,
        error_only: bool = False,
    ) -> dict[str, Any]:
        return {
            "traces": self.trace_service.list_recent_traces(
                limit=limit,
                user_id=user_id,
                conversation_id=conversation_id,
                insufficient_only=insufficient_only,
                error_only=error_only,
            )
        }

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        trace = self.trace_service.get_trace(trace_id)
        return {"trace": trace} if trace is not None else None

    def _database_check(self) -> dict[str, str]:
        try:
            with self.store.connect() as connection:
                connection.execute("SELECT 1").fetchone()
            return {"status": "ok", "message": "Database is reachable."}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def _settings_check(self) -> dict[str, str]:
        try:
            self.settings.require_auth_secret()
            return {"status": "ok", "message": "Settings loaded."}
        except Exception:
            return {"status": "error", "message": "Required settings are missing."}

    def _index_check(self) -> dict[str, str]:
        path = self.settings.index_path
        try:
            if path.exists():
                return {
                    "status": "ok",
                    "message": "Index path exists.",
                    "path": str(path),
                }
            return {
                "status": "missing",
                "message": "Index path does not exist.",
                "path": str(path),
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc), "path": str(path)}

    def _qdrant_config_check(self) -> dict[str, object]:
        uses_cloud = bool(self.settings.qdrant_url.strip())
        collection = self._effective_qdrant_collection()
        if uses_cloud:
            status = "configured" if collection else "missing"
        else:
            status = "local"
        return {
            "status": status,
            "collection": collection,
            "usesCloud": uses_cloud,
        }

    def _llm_config_check(self) -> dict[str, str]:
        provider = self.settings.llm_provider.strip().lower() or "groq"
        model = self._llm_model(provider)
        configured = self._llm_provider_is_configured(provider)
        return {
            "status": "configured" if configured else "missing",
            "provider": provider,
            "model": model,
        }

    def _llm_model(self, provider: str) -> str:
        if provider == "azure_openai":
            return self.settings.azure_openai_model
        return self.settings.groq_model

    def _llm_provider_is_configured(self, provider: str) -> bool:
        if provider == "azure_openai":
            return bool(
                self.settings.azure_openai_responses_endpoint.strip()
                and self.settings.optional_secret_value(self.settings.azure_openai_api_key)
            )
        return bool(self.settings.optional_secret_value(self.settings.groq_api_key))

    def _effective_qdrant_collection(self) -> str:
        if self.settings.qdrant_collection_was_configured():
            return self.settings.qdrant_collection
        return str(self._index_manifest().get("collection_name") or self.settings.qdrant_collection)

    def _effective_retriever_record_source(self) -> str:
        if self.settings.retriever_record_source.strip():
            return self.settings.retriever_record_source.strip()
        return str(self._index_manifest().get("record_source") or "sqlite")

    def _effective_dense_model(self) -> str:
        return str(self._index_manifest().get("dense_model_name") or self.settings.dense_model)

    def _index_manifest(self) -> dict[str, Any]:
        manifest_path = self._manifest_path(self.settings.index_path)
        if not manifest_path.exists():
            return {}
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _manifest_path(self, index_path: Path) -> Path:
        if index_path.is_dir():
            return index_path / "current.json"
        return index_path

    def _check_is_healthy(self, check: dict[str, Any]) -> bool:
        return str(check.get("status") or "") in {"ok", "configured", "local"}
