from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient

import vn_labor_law_ai_assistant.api.deps as api_deps
from vn_labor_law_ai_assistant.admin.service import AdminService
from vn_labor_law_ai_assistant.api import app
from vn_labor_law_ai_assistant.core.config import Settings, get_settings
from vn_labor_law_ai_assistant.observability import ChatTraceService
from helpers import create_test_auth_store


class AdminApiTest(TestCase):
    def setUp(self) -> None:
        self.tmpdir = TemporaryDirectory()
        self.store = create_test_auth_store(Path(self.tmpdir.name) / "app.db")
        self.previous_auth_store = api_deps._auth_store
        api_deps._auth_store = self.store
        self.client = TestClient(app)

    def tearDown(self) -> None:
        api_deps._auth_store = self.previous_auth_store
        get_settings.cache_clear()
        self.tmpdir.cleanup()

    def _token_for(self, email: str, password: str) -> str:
        user = self.store.authenticate_user(email, password)
        self.assertIsNotNone(user)
        assert user is not None
        token, _ = self.store.create_session(user)
        return token

    def test_admin_stats_requires_auth(self) -> None:
        response = self.client.get("/admin/stats")

        self.assertEqual(response.status_code, 401)

    def test_admin_stats_requires_admin(self) -> None:
        token = self._token_for("user@example.com", "user12345")

        response = self.client.get(
            "/admin/stats",
            headers={"Authorization": f"Bearer {token}"},
        )

        self.assertEqual(response.status_code, 403)

    def test_admin_stats_returns_counts_for_admin(self) -> None:
        admin_token = self._token_for("admin@example.com", "admin12345")
        user = self.store.authenticate_user("user@example.com", "user12345")
        self.assertIsNotNone(user)
        assert user is not None
        conversation = self.store.create_conversation(user_id=user.id, title="Question")
        self.store.append_message(
            conversation_id=conversation["id"],
            role="user",
            content="Question",
        )
        self.store.append_message(
            conversation_id=conversation["id"],
            role="assistant",
            content="Answer",
        )
        ChatTraceService(self.store.database).record_chat_trace(
            user_id=user.id,
            question="Question",
            conversation_id=conversation["id"],
            insufficient_context=True,
            error="trace error",
        )

        response = self.client.get(
            "/admin/stats",
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["user"]["role"], "admin")
        self.assertEqual(payload["stats"]["totalUsers"], 2)
        self.assertEqual(payload["stats"]["activeUsers"], 2)
        self.assertEqual(payload["stats"]["adminUsers"], 1)
        self.assertEqual(payload["stats"]["totalConversations"], 1)
        self.assertEqual(payload["stats"]["totalMessages"], 2)
        self.assertEqual(payload["stats"]["activeSessions"], 1)
        self.assertEqual(payload["stats"]["totalTraces"], 1)
        self.assertEqual(payload["stats"]["tracesWithErrors"], 1)
        self.assertEqual(payload["stats"]["insufficientContextTraces"], 1)
        self.assertIn("runtime", payload)
        self.assertNotIn("conversations", payload)

    def test_admin_traces_requires_admin(self) -> None:
        token = self._token_for("user@example.com", "user12345")

        response = self.client.get(
            "/admin/traces",
            headers={"Authorization": f"Bearer {token}"},
        )

        self.assertEqual(response.status_code, 403)

    def test_admin_traces_list_returns_recent_traces(self) -> None:
        admin_token = self._token_for("admin@example.com", "admin12345")
        user = self.store.get_user_by_email("user@example.com")
        self.assertIsNotNone(user)
        assert user is not None
        ChatTraceService(self.store.database).record_chat_trace(
            user_id=user.id,
            question="Recent trace",
            request_id="request-recent",
            selected_contexts=[{"chunkId": "chunk-1"}],
            citations={"legal_basis": ["citation"], "evidence_quotes": []},
        )

        response = self.client.get(
            "/admin/traces?limit=10",
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["traces"]), 1)
        self.assertEqual(payload["traces"][0]["requestId"], "request-recent")
        self.assertEqual(payload["traces"][0]["selectedContextCount"], 1)
        self.assertEqual(payload["traces"][0]["citationCount"], 1)

    def test_admin_trace_detail_does_not_expose_secrets(self) -> None:
        with self._fake_secret_env():
            admin_token = self._token_for("admin@example.com", "admin12345")
            user = self.store.get_user_by_email("user@example.com")
            self.assertIsNotNone(user)
            assert user is not None
            trace = ChatTraceService(self.store.database).record_chat_trace(
                user_id=user.id,
                question="Detail trace",
                intent={"article_numbers": ["35"]},
                retrieved_hits=[{"chunkId": "chunk-1"}],
                selected_contexts=[{"chunkId": "chunk-1", "textPreview": "preview"}],
                citations={"legal_basis": ["citation"], "evidence_quotes": []},
            )
            response = self.client.get(
                f"/admin/traces/{trace['id']}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["trace"]["id"], trace["id"])
        payload_text = json.dumps(payload, sort_keys=True)
        self.assert_no_secret_leak(payload_text)

    def test_admin_health_does_not_expose_secrets(self) -> None:
        with self._fake_secret_env():
            admin_token = self._token_for("admin@example.com", "admin12345")
            response = self.client.get(
                "/admin/health",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        self.assertEqual(response.status_code, 200)
        payload_text = json.dumps(response.json(), sort_keys=True)
        self.assert_no_secret_leak(payload_text)

    def test_admin_health_settings_error_does_not_expose_secret_name(self) -> None:
        with patch.dict(os.environ, {"APP_ENV": "production"}, clear=True):
            settings = Settings(_env_file=None)

        payload = AdminService(self.store, settings=settings).get_health()

        self.assertEqual(payload["checks"]["settings"]["status"], "error")
        payload_text = json.dumps(payload, sort_keys=True)
        self.assertNotIn("AUTH_SECRET", payload_text)

    def test_admin_retrieval_config_does_not_expose_secrets(self) -> None:
        with self._fake_secret_env():
            admin_token = self._token_for("admin@example.com", "admin12345")
            response = self.client.get(
                "/admin/retrieval-config",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["rerankerTopN"], 24)
        self.assertIn("qdrantCollection", payload)
        payload_text = json.dumps(payload, sort_keys=True)
        self.assert_no_secret_leak(payload_text)

    def _fake_secret_env(self):
        get_settings.cache_clear()
        return patch.dict(
            os.environ,
            {
                "APP_ENV": "development",
                "AUTH_SECRET": "fake-auth-secret",
                "GROQ_API_KEY": "fake-groq-key",
                "QDRANT_API_KEY": "fake-qdrant-key",
                "EMBEDDING_API_TOKEN": "fake-embedding-token",
                "DEFAULT_USER_PASSWORD": "fake-user-password",
                "DEFAULT_ADMIN_PASSWORD": "fake-admin-password",
            },
            clear=False,
        )

    def assert_no_secret_leak(self, payload_text: str) -> None:
        forbidden_values = (
            "fake-auth-secret",
            "fake-groq-key",
            "fake-qdrant-key",
            "fake-embedding-token",
            "fake-user-password",
            "fake-admin-password",
        )
        forbidden_keys = (
            "AUTH_SECRET",
            "GROQ_API_KEY",
            "QDRANT_API_KEY",
            "EMBEDDING_API_TOKEN",
            "DEFAULT_USER_PASSWORD",
            "DEFAULT_ADMIN_PASSWORD",
        )
        for forbidden in (*forbidden_values, *forbidden_keys):
            self.assertNotIn(forbidden, payload_text)
