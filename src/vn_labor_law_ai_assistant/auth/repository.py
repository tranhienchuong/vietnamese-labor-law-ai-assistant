from __future__ import annotations

import sqlite3
import time
from typing import Any

from ..db.sqlite import SQLiteDatabase
from .models import AuthUser, Role


class AuthRepository:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def row_to_user(self, row: sqlite3.Row | None) -> AuthUser | None:
        if row is None:
            return None
        return AuthUser(
            id=str(row["id"]),
            name=str(row["name"]),
            email=str(row["email"]),
            role=str(row["role"]),  # type: ignore[arg-type]
            avatar_url=str(row["avatar_url"]) if row["avatar_url"] else None,
            is_active=bool(row["is_active"]),
        )

    def create_user(
        self,
        *,
        user_id: str,
        name: str,
        email: str,
        password_hash: str,
        role: Role,
        now: str,
    ) -> None:
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO users (
                    id, name, email, password_hash, auth_provider,
                    role, is_active, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, 'password', ?, 1, ?, ?)
                """,
                (
                    user_id,
                    name,
                    email,
                    password_hash,
                    role,
                    now,
                    now,
                ),
            )

    def get_user_by_id(self, user_id: str) -> AuthUser | None:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT id, name, email, role, avatar_url, is_active
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()
        return self.row_to_user(row)

    def get_user_by_email(self, email: str) -> AuthUser | None:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT id, name, email, role, avatar_url, is_active
                FROM users
                WHERE email = ?
                """,
                (email.strip().lower(),),
            ).fetchone()
        return self.row_to_user(row)

    def get_password_user_row(self, email: str) -> sqlite3.Row | None:
        with self.database.connect() as connection:
            return connection.execute(
                """
                SELECT id, name, email, password_hash, role, avatar_url, is_active
                FROM users
                WHERE email = ?
                """,
                (email.strip().lower(),),
            ).fetchone()

    def create_session(
        self,
        *,
        session_id: str,
        user_id: str,
        token_hash_value: str,
        created_at: str,
        expires_at: int,
    ) -> None:
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (id, user_id, token_hash, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, user_id, token_hash_value, created_at, expires_at),
            )

    def revoke_session(self, *, token_hash_value: str, revoked_at: str) -> None:
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE sessions
                SET revoked_at = ?
                WHERE token_hash = ? AND revoked_at IS NULL
                """,
                (revoked_at, token_hash_value),
            )

    def get_user_for_session(
        self,
        *,
        session_id: str,
        token_hash_value: str,
        now_epoch: int,
    ) -> AuthUser | None:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT u.id, u.name, u.email, u.role, u.avatar_url, u.is_active
                FROM sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.id = ?
                  AND s.token_hash = ?
                  AND s.revoked_at IS NULL
                  AND s.expires_at >= ?
                """,
                (session_id, token_hash_value, now_epoch),
            ).fetchone()
        return self.row_to_user(row)

    def user_from_row(self, row: sqlite3.Row | None) -> AuthUser | None:
        return self.row_to_user(row)

    def count_users(self) -> int:
        with self.database.connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM users").fetchone()
        return int(row["count"]) if row is not None else 0

    def count_active_users(self) -> int:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM users WHERE is_active = 1"
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    def count_admin_users(self) -> int:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM users WHERE role = 'admin'"
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    def count_active_sessions(self) -> int:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM sessions
                WHERE revoked_at IS NULL AND expires_at >= ?
                """,
                (int(time.time()),),
            ).fetchone()
        return int(row["count"]) if row is not None else 0
