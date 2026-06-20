from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from urllib.error import HTTPError

from fastapi.testclient import TestClient

import vn_labor_law_ai_assistant.api.deps as api_deps
from helpers import create_test_auth_store
from vn_labor_law_ai_assistant.api import app
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


def _supabase_env(admin_emails: str = "") -> dict[str, str]:
    return {
        "AUTH_PROVIDER": "supabase",
        "SUPABASE_URL": "https://project-ref.supabase.co",
        "SUPABASE_ANON_KEY": "test-anon-key",
        "ADMIN_EMAILS": admin_emails,
        "AUTH_SEED_DEFAULT_USERS": "0",
    }


class SupabaseAuthTest(TestCase):
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
