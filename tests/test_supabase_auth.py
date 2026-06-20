from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from urllib.error import HTTPError

from fastapi.testclient import TestClient

import vn_labor_law_ai_assistant.api.deps as api_deps
from helpers import create_test_auth_store
from vn_labor_law_ai_assistant.api import app
from vn_labor_law_ai_assistant.auth.models import AuthUser
from vn_labor_law_ai_assistant.core.config import get_settings


class _FakeSupabaseResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeSupabaseResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class _FakeSupabaseStore:
    def __init__(self) -> None:
        self.users: dict[str, AuthUser] = {}
        self.messages: list[dict[str, object]] = []
        self.traces: list[dict[str, object]] = []

    def upsert_external_user(
        self,
        *,
        user_id: str,
        name: str,
        email: str,
        auth_provider: str,
        provider_id: str,
        role: str,
        avatar_url: str | None = None,
    ) -> AuthUser:
        user = AuthUser(
            id=user_id,
            name=name or email,
            email=email,
            role=role,  # type: ignore[arg-type]
            avatar_url=avatar_url,
            is_active=True,
        )
        self.users[user_id] = user
        return user

    def ensure_conversation_for_question(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        question: str,
    ) -> dict[str, object]:
        return {
            "id": conversation_id or "conversation-1",
            "user_id": user_id,
            "title": question,
        }

    def append_message(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
        citations: object | None = None,
        metadata: object | None = None,
    ) -> dict[str, object]:
        message = {
            "id": f"message-{len(self.messages) + 1}",
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "citations": citations,
            "metadata": metadata,
        }
        self.messages.append(message)
        return message

    def record_chat_trace(self, **kwargs: object) -> dict[str, object]:
        trace = {"id": f"trace-{len(self.traces) + 1}", **kwargs}
        self.traces.append(trace)
        return trace

    def count_users(self) -> int:
        return 4

    def count_active_users(self) -> int:
        return 4

    def count_admin_users(self) -> int:
        return 1

    def count_conversations(self) -> int:
        return 7

    def count_messages(self) -> int:
        return 12

    def count_active_sessions(self) -> int:
        return 0

    def count_traces(self) -> int:
        return 5

    def count_traces_with_errors(self) -> int:
        return 2

    def count_insufficient_context_traces(self) -> int:
        return 1

    def list_recent_traces(self, **kwargs: object) -> list[dict[str, object]]:
        return self.traces

    def get_trace(self, trace_id: str) -> dict[str, object] | None:
        return next((trace for trace in self.traces if trace["id"] == trace_id), None)


def _supabase_env(admin_emails: str = "", *, app_data_backend: str = "sqlite") -> dict[str, str]:
    return {
        "AUTH_PROVIDER": "supabase",
        "APP_DATA_BACKEND": app_data_backend,
        "SUPABASE_URL": "https://project-ref.supabase.co",
        "SUPABASE_ANON_KEY": "test-anon-key",
        "SUPABASE_DB_URL": "postgresql://postgres:password@localhost:5432/postgres",
        "ADMIN_EMAILS": admin_emails,
        "AUTH_SEED_DEFAULT_USERS": "0",
    }


class SupabaseAuthTest(TestCase):
    def setUp(self) -> None:
        self.tmpdir = TemporaryDirectory()
        self.store = create_test_auth_store(Path(self.tmpdir.name) / "app.db")
        self.previous_auth_store = api_deps._auth_store
        self.previous_app_store = api_deps._app_store
        self.previous_app_store_backend = api_deps._app_store_backend
        api_deps._auth_store = self.store
        api_deps._app_store = None
        api_deps._app_store_backend = None
        self.client = TestClient(app)

    def tearDown(self) -> None:
        api_deps._auth_store = self.previous_auth_store
        api_deps._app_store = self.previous_app_store
        api_deps._app_store_backend = self.previous_app_store_backend
        get_settings.cache_clear()
        self.tmpdir.cleanup()

    def test_missing_authorization_returns_401_in_supabase_mode(self) -> None:
        with patch.dict(os.environ, _supabase_env(), clear=False):
            get_settings.cache_clear()

            response = self.client.get("/auth/me")

        self.assertEqual(response.status_code, 401)

    def test_invalid_supabase_token_returns_401(self) -> None:
        error = HTTPError(
            url="https://project-ref.supabase.co/auth/v1/user",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=None,
        )
        with (
            patch.dict(os.environ, _supabase_env(), clear=False),
            patch("vn_labor_law_ai_assistant.auth.supabase.urlopen", side_effect=error),
        ):
            get_settings.cache_clear()

            response = self.client.get(
                "/auth/me",
                headers={"Authorization": "Bearer invalid-token"},
            )

        self.assertEqual(response.status_code, 401)

    def test_supabase_user_verification_success_upserts_user(self) -> None:
        payload = {
            "id": "supabase-user-id",
            "email": "researcher@example.com",
            "user_metadata": {
                "full_name": "Legal Researcher",
                "avatar_url": "https://example.com/avatar.png",
            },
        }
        with (
            patch.dict(os.environ, _supabase_env(), clear=False),
            patch(
                "vn_labor_law_ai_assistant.auth.supabase.urlopen",
                return_value=_FakeSupabaseResponse(payload),
            ) as mocked_urlopen,
        ):
            get_settings.cache_clear()

            response = self.client.get(
                "/auth/me",
                headers={"Authorization": "Bearer valid-token"},
            )

        self.assertEqual(response.status_code, 200)
        user = response.json()["user"]
        self.assertEqual(user["id"], "supabase-user-id")
        self.assertEqual(user["email"], "researcher@example.com")
        self.assertEqual(user["name"], "Legal Researcher")
        self.assertEqual(user["role"], "user")
        self.assertEqual(user["avatarUrl"], "https://example.com/avatar.png")
        self.assertIsNotNone(self.store.get_user_by_id("supabase-user-id"))

        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://project-ref.supabase.co/auth/v1/user")
        self.assertEqual(request.get_header("Authorization"), "Bearer valid-token")
        self.assertEqual(request.get_header("Apikey"), "test-anon-key")

    def test_supabase_admin_email_mapping(self) -> None:
        payload = {
            "id": "supabase-admin-id",
            "email": "owner@example.com",
            "user_metadata": {"name": "Admin User"},
        }
        with (
            patch.dict(os.environ, _supabase_env("owner@example.com"), clear=False),
            patch(
                "vn_labor_law_ai_assistant.auth.supabase.urlopen",
                return_value=_FakeSupabaseResponse(payload),
            ),
        ):
            get_settings.cache_clear()

            response = self.client.get(
                "/auth/me",
                headers={"Authorization": "Bearer admin-token"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user"]["role"], "admin")

    def test_supabase_app_data_mode_maps_non_admin_to_403(self) -> None:
        payload = {
            "id": "supabase-user-id",
            "email": "researcher@example.com",
            "user_metadata": {"name": "Researcher"},
        }
        fake_store = _FakeSupabaseStore()
        with (
            patch.dict(os.environ, _supabase_env(app_data_backend="supabase"), clear=False),
            patch(
                "vn_labor_law_ai_assistant.auth.supabase.urlopen",
                return_value=_FakeSupabaseResponse(payload),
            ),
        ):
            get_settings.cache_clear()
            api_deps._app_store = fake_store  # type: ignore[assignment]
            api_deps._app_store_backend = "supabase"

            response = self.client.get(
                "/admin/stats",
                headers={"Authorization": "Bearer user-token"},
            )

        self.assertEqual(response.status_code, 403)
        self.assertIn("supabase-user-id", fake_store.users)

    def test_supabase_app_data_mode_admin_email_can_read_stats(self) -> None:
        payload = {
            "id": "supabase-admin-id",
            "email": "owner@example.com",
            "user_metadata": {"name": "Owner"},
        }
        fake_store = _FakeSupabaseStore()
        with (
            patch.dict(
                os.environ,
                _supabase_env("owner@example.com", app_data_backend="supabase"),
                clear=False,
            ),
            patch(
                "vn_labor_law_ai_assistant.auth.supabase.urlopen",
                return_value=_FakeSupabaseResponse(payload),
            ),
        ):
            get_settings.cache_clear()
            api_deps._app_store = fake_store  # type: ignore[assignment]
            api_deps._app_store_backend = "supabase"

            response = self.client.get(
                "/admin/stats",
                headers={"Authorization": "Bearer admin-token"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["stats"]["totalUsers"], 4)
        self.assertEqual(response.json()["stats"]["activeSessions"], 0)

    def test_chat_accepts_valid_supabase_token_with_supabase_app_store(self) -> None:
        payload = {
            "id": "supabase-user-id",
            "email": "researcher@example.com",
            "user_metadata": {"name": "Researcher"},
        }
        fake_store = _FakeSupabaseStore()
        fake_retriever = SimpleNamespace(
            retrieve=lambda *args, **kwargs: SimpleNamespace(contexts=[], hits=[], intent={})
        )
        with (
            patch.dict(os.environ, _supabase_env(app_data_backend="supabase"), clear=False),
            patch(
                "vn_labor_law_ai_assistant.auth.supabase.urlopen",
                return_value=_FakeSupabaseResponse(payload),
            ),
            patch(
                "vn_labor_law_ai_assistant.api.routes.chat.get_retriever",
                return_value=fake_retriever,
            ),
        ):
            get_settings.cache_clear()
            api_deps._app_store = fake_store  # type: ignore[assignment]
            api_deps._app_store_backend = "supabase"

            response = self.client.post(
                "/chat",
                headers={"Authorization": "Bearer valid-token"},
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": "Can an employee terminate without notice?",
                        }
                    ],
                    "includeCitations": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("X-Conversation-Id"), "conversation-1")
        self.assertEqual(len(fake_store.messages), 2)
        self.assertEqual(len(fake_store.traces), 1)
