from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .auth.models import AuthUser, MessageRole, Role
from .auth.repository import AuthRepository
from .auth.service import AuthService, user_payload
from .conversations.repository import ConversationRepository
from .conversations.service import ConversationService
from .core.security import (
    DEFAULT_SESSION_TTL_SECONDS,
    PASSWORD_ITERATIONS,
    create_access_token,
    decode_and_verify_token,
    hash_password,
    token_hash,
    verify_password,
)
from .db.sqlite import SQLiteDatabase, default_database_path, utc_timestamp


class AuthStore:
    def __init__(self, database_path: Path | None = None) -> None:
        self.database = SQLiteDatabase(database_path)
        self.database_path = self.database.database_path
        self.auth_repository = AuthRepository(self.database)
        self.auth_service = AuthService(self.auth_repository)
        self.conversation_repository = ConversationRepository(self.database)
        self.conversation_service = ConversationService(self.conversation_repository)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        return self.database.connect()

    def initialize(self) -> None:
        self.database.initialize()
        self.seed_default_users()

    def seed_default_users(self) -> None:
        self.auth_service.seed_default_users()

    def create_user_if_missing(
        self,
        *,
        name: str,
        email: str,
        password: str,
        role: Role,
    ) -> AuthUser:
        return self.auth_service.create_user_if_missing(
            name=name,
            email=email,
            password=password,
            role=role,
        )

    def _row_to_user(self, row: sqlite3.Row | None) -> AuthUser | None:
        return self.auth_repository.row_to_user(row)

    def get_user_by_id(self, user_id: str) -> AuthUser | None:
        return self.auth_service.get_user_by_id(user_id)

    def get_user_by_email(self, email: str) -> AuthUser | None:
        return self.auth_service.get_user_by_email(email)

    def authenticate_user(self, email: str, password: str) -> AuthUser | None:
        return self.auth_service.authenticate_user(email, password)

    def create_session(self, user: AuthUser) -> tuple[str, int]:
        return self.auth_service.create_session(user)

    def revoke_session(self, token: str) -> None:
        self.auth_service.revoke_session(token)

    def get_user_by_token(self, token: str) -> AuthUser | None:
        return self.auth_service.get_user_by_token(token)

    def create_conversation(self, *, user_id: str, title: str) -> dict[str, Any]:
        return self.conversation_service.create_conversation(
            user_id=user_id,
            title=title,
        )

    def list_conversations(self, *, user_id: str) -> list[dict[str, Any]]:
        return self.conversation_service.list_conversations(user_id=user_id)

    def get_conversation(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        return self.conversation_service.get_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )

    def append_message(
        self,
        *,
        conversation_id: str,
        role: MessageRole,
        content: str,
        citations: Any | None = None,
        metadata: Any | None = None,
    ) -> dict[str, Any]:
        return self.conversation_service.append_message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            citations=citations,
            metadata=metadata,
        )

    def list_messages(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> list[dict[str, Any]] | None:
        return self.conversation_service.list_messages(
            user_id=user_id,
            conversation_id=conversation_id,
        )

    def ensure_conversation_for_question(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        question: str,
    ) -> dict[str, Any]:
        return self.conversation_service.ensure_conversation_for_question(
            user_id=user_id,
            conversation_id=conversation_id,
            question=question,
        )


__all__ = [
    "AuthStore",
    "AuthUser",
    "Role",
    "MessageRole",
    "DEFAULT_SESSION_TTL_SECONDS",
    "PASSWORD_ITERATIONS",
    "create_access_token",
    "decode_and_verify_token",
    "default_database_path",
    "hash_password",
    "token_hash",
    "utc_timestamp",
    "user_payload",
    "verify_password",
]
