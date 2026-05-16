from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from vn_labor_law_ai_assistant.auth_store import AuthStore


def create_test_users(auth_store: AuthStore) -> None:
    auth_store.create_user_if_missing(
        name="Test User",
        email="user@example.com",
        password="user12345",
        role="user",
    )
    auth_store.create_user_if_missing(
        name="Test Admin",
        email="admin@example.com",
        password="admin12345",
        role="admin",
    )


def create_test_auth_store(database_path: Path) -> AuthStore:
    with patch.dict(os.environ, {"AUTH_SEED_DEFAULT_USERS": "0"}, clear=False):
        auth_store = AuthStore(database_path)
    create_test_users(auth_store)
    return auth_store
