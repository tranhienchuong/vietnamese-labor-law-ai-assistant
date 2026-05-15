from __future__ import annotations

import os
import time
import uuid
from typing import Any

from ..core.security import (
    create_access_token,
    hash_password,
    token_hash,
    verify_password,
    decode_and_verify_token,
)
from ..db.sqlite import utc_timestamp
from .models import AuthUser, Role
from .repository import AuthRepository


class AuthService:
    def __init__(self, repository: AuthRepository) -> None:
        self.repository = repository

    def seed_default_users(self) -> None:
        if os.getenv("AUTH_SEED_DEFAULT_USERS", "1").strip() in {"0", "false", "False"}:
            return

        self.create_user_if_missing(
            name=os.getenv("DEFAULT_USER_NAME", "Nguoi dung"),
            email=os.getenv("DEFAULT_USER_EMAIL", "user@example.com"),
            password=os.getenv("DEFAULT_USER_PASSWORD", "user12345"),
            role="user",
        )
        self.create_user_if_missing(
            name=os.getenv("DEFAULT_ADMIN_NAME", "Quan tri vien"),
            email=os.getenv("DEFAULT_ADMIN_EMAIL", "admin@example.com"),
            password=os.getenv("DEFAULT_ADMIN_PASSWORD", "admin12345"),
            role="admin",
        )

    def create_user_if_missing(
        self,
        *,
        name: str,
        email: str,
        password: str,
        role: Role,
    ) -> AuthUser:
        normalized_email = email.strip().lower()
        existing = self.get_user_by_email(normalized_email)
        if existing is not None:
            return existing

        now = utc_timestamp()
        user_id = str(uuid.uuid4())
        self.repository.create_user(
            user_id=user_id,
            name=name.strip() or normalized_email,
            email=normalized_email,
            password_hash=hash_password(password),
            role=role,
            now=now,
        )
        user = self.get_user_by_id(user_id)
        if user is None:
            raise RuntimeError("Failed to create user.")
        return user

    def get_user_by_id(self, user_id: str) -> AuthUser | None:
        return self.repository.get_user_by_id(user_id)

    def get_user_by_email(self, email: str) -> AuthUser | None:
        return self.repository.get_user_by_email(email)

    def authenticate_user(self, email: str, password: str) -> AuthUser | None:
        row = self.repository.get_password_user_row(email)
        if row is None or not row["is_active"] or not row["password_hash"]:
            return None
        if not verify_password(password, str(row["password_hash"])):
            return None
        return self.repository.user_from_row(row)

    def create_session(self, user: AuthUser) -> tuple[str, int]:
        session_id = str(uuid.uuid4())
        token, expires_at = create_access_token(session_id=session_id, user=user)
        self.repository.create_session(
            session_id=session_id,
            user_id=user.id,
            token_hash_value=token_hash(token),
            created_at=utc_timestamp(),
            expires_at=expires_at,
        )
        return token, expires_at

    def revoke_session(self, token: str) -> None:
        self.repository.revoke_session(
            token_hash_value=token_hash(token),
            revoked_at=utc_timestamp(),
        )

    def get_user_by_token(self, token: str) -> AuthUser | None:
        payload = decode_and_verify_token(token)
        if payload is None:
            return None
        session_id = str(payload.get("sid") or "")
        if not session_id:
            return None

        user = self.repository.get_user_for_session(
            session_id=session_id,
            token_hash_value=token_hash(token),
            now_epoch=int(time.time()),
        )
        if user is None or not user.is_active:
            return None
        return user


def user_payload(user: AuthUser) -> dict[str, Any]:
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
        "avatarUrl": user.avatar_url,
    }
