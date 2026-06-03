from __future__ import annotations

import os
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from vn_labor_law_ai_assistant.core.config import (
    DEV_ONLY_AUTH_SECRET,
    Settings,
    get_settings,
    load_settings,
)


class SettingsTest(TestCase):
    def tearDown(self) -> None:
        get_settings.cache_clear()

    def test_get_settings_is_cached_and_parses_cors_origins(self) -> None:
        with patch.dict(
            os.environ,
            {"CORS_ALLOW_ORIGINS": "http://localhost:3000, https://example.com"},
            clear=False,
        ):
            get_settings.cache_clear()
            settings = get_settings()

            self.assertIs(settings, get_settings())
            self.assertEqual(
                settings.cors_origins_list(),
                ["http://localhost:3000", "https://example.com"],
            )

    def test_settings_parse_paths_and_secret_values(self) -> None:
        settings = Settings(
            _env_file=None,
            APP_DB_PATH="tmp/app.db",
            AUTH_SECRET="secret",
        )

        self.assertEqual(settings.app_db_path, Path("tmp/app.db"))
        self.assertEqual(settings.require_auth_secret(), "secret")

    def test_bool_settings_keep_legacy_default_on_empty_or_unknown_values(self) -> None:
        settings = Settings(
            _env_file=None,
            AUTH_SEED_DEFAULT_USERS="",
            QUERY_ROUTER_ENABLED="not-a-bool",
            QUERY_ROUTER_FALLBACK_TO_HEURISTIC="0",
        )

        self.assertTrue(settings.auth_seed_default_users)
        self.assertTrue(settings.query_router_enabled)
        self.assertFalse(settings.query_router_fallback_to_heuristic)

    def test_dev_auth_secret_fallback_and_production_guard(self) -> None:
        dev_settings = Settings(_env_file=None, APP_ENV="development", AUTH_SECRET="")
        self.assertEqual(dev_settings.require_auth_secret(), DEV_ONLY_AUTH_SECRET)

        prod_settings = Settings(_env_file=None, APP_ENV="production", AUTH_SECRET="")
        with self.assertRaisesRegex(RuntimeError, "AUTH_SECRET"):
            prod_settings.require_auth_secret()

    def test_production_rejects_default_seeded_user_passwords(self) -> None:
        settings = Settings(
            _env_file=None,
            APP_ENV="production",
            AUTH_SECRET="prod-secret",
            AUTH_SEED_DEFAULT_USERS=True,
        )

        with self.assertRaisesRegex(RuntimeError, "default passwords"):
            settings.validate_auth_seed_configuration()

    def test_production_allows_seed_disabled_or_custom_seed_passwords(self) -> None:
        seed_disabled = Settings(
            _env_file=None,
            APP_ENV="production",
            AUTH_SECRET="prod-secret",
            AUTH_SEED_DEFAULT_USERS=False,
        )
        seed_disabled.validate_auth_seed_configuration()

        custom_passwords = Settings(
            _env_file=None,
            APP_ENV="production",
            AUTH_SECRET="prod-secret",
            AUTH_SEED_DEFAULT_USERS=True,
            DEFAULT_USER_PASSWORD="custom-user-password",
            DEFAULT_ADMIN_PASSWORD="custom-admin-password",
        )
        custom_passwords.validate_auth_seed_configuration()

    def test_load_settings_reads_current_environment(self) -> None:
        with patch.dict(os.environ, {"APP_DB_PATH": "runtime/app.db"}, clear=True):
            self.assertEqual(load_settings().app_db_path, Path("runtime/app.db"))

    def test_qdrant_collection_tracks_explicit_configuration(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            default_settings = Settings(_env_file=None)
            configured_settings = Settings(_env_file=None, QDRANT_COLLECTION="custom")

            self.assertFalse(default_settings.qdrant_collection_was_configured())
            self.assertTrue(configured_settings.qdrant_collection_was_configured())
