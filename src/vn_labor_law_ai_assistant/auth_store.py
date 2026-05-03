from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


Role = Literal["user", "admin"]
MessageRole = Literal["user", "assistant", "system"]

DEFAULT_SESSION_TTL_SECONDS = 60 * 60 * 24 * 7
PASSWORD_ITERATIONS = 260_000


@dataclass(frozen=True)
class AuthUser:
    id: str
    name: str
    email: str
    role: Role
    avatar_url: str | None
    is_active: bool


def default_database_path() -> Path:
    return Path(os.getenv("APP_DB_PATH", "artifacts/app.db"))


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _b64_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _token_secret() -> bytes:
    secret = os.getenv("AUTH_SECRET", "").strip()
    if not secret:
        secret = "dev-only-change-me-vietnamese-labor-law-ai"
    return secret.encode("utf-8")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    return "pbkdf2_sha256${}${}${}".format(
        PASSWORD_ITERATIONS,
        _b64_encode(salt),
        _b64_encode(digest),
    )


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations_text, salt_text, digest_text = password_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_text)
        salt = _b64_decode(salt_text)
        expected = _b64_decode(digest_text)
    except (ValueError, TypeError):
        return False

    observed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(observed, expected)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_access_token(
    *,
    session_id: str,
    user: AuthUser,
    ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
) -> tuple[str, int]:
    expires_at = int(time.time()) + ttl_seconds
    payload = {
        "sid": session_id,
        "sub": user.id,
        "email": user.email,
        "name": user.name,
        "role": user.role,
        "exp": expires_at,
    }
    encoded_payload = _b64_encode(_json_dumps(payload).encode("utf-8"))
    signature = hmac.new(_token_secret(), encoded_payload.encode("ascii"), hashlib.sha256)
    token = "{}.{}".format(encoded_payload, _b64_encode(signature.digest()))
    return token, expires_at


def decode_and_verify_token(token: str) -> dict[str, Any] | None:
    try:
        payload_text, signature_text = token.split(".", 1)
    except ValueError:
        return None

    expected = hmac.new(_token_secret(), payload_text.encode("ascii"), hashlib.sha256)
    try:
        observed = _b64_decode(signature_text)
    except (ValueError, TypeError):
        return None
    if not hmac.compare_digest(observed, expected.digest()):
        return None

    try:
        payload = json.loads(_b64_decode(payload_text).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None
    if int(payload.get("exp") or 0) < int(time.time()):
        return None
    return payload if isinstance(payload, dict) else None


class AuthStore:
    def __init__(self, database_path: Path | None = None) -> None:
        self.database_path = database_path or default_database_path()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT,
                    auth_provider TEXT NOT NULL DEFAULT 'password',
                    provider_id TEXT,
                    role TEXT NOT NULL CHECK (role IN ('user', 'admin')),
                    avatar_url TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_message_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
                    ON conversations(user_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    citations_json TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
                    ON messages(conversation_id, created_at ASC);

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    expires_at INTEGER NOT NULL,
                    revoked_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_token_hash
                    ON sessions(token_hash);
                """
            )
        self.seed_default_users()

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

        now = _utc_timestamp()
        user_id = str(uuid.uuid4())
        with self.connect() as connection:
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
                    name.strip() or normalized_email,
                    normalized_email,
                    hash_password(password),
                    role,
                    now,
                    now,
                ),
            )
        user = self.get_user_by_id(user_id)
        if user is None:
            raise RuntimeError("Failed to create user.")
        return user

    def _row_to_user(self, row: sqlite3.Row | None) -> AuthUser | None:
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

    def get_user_by_id(self, user_id: str) -> AuthUser | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id, name, email, role, avatar_url, is_active
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()
        return self._row_to_user(row)

    def get_user_by_email(self, email: str) -> AuthUser | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id, name, email, role, avatar_url, is_active
                FROM users
                WHERE email = ?
                """,
                (email.strip().lower(),),
            ).fetchone()
        return self._row_to_user(row)

    def authenticate_user(self, email: str, password: str) -> AuthUser | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id, name, email, password_hash, role, avatar_url, is_active
                FROM users
                WHERE email = ?
                """,
                (email.strip().lower(),),
            ).fetchone()

        if row is None or not row["is_active"] or not row["password_hash"]:
            return None
        if not verify_password(password, str(row["password_hash"])):
            return None
        return self._row_to_user(row)

    def create_session(self, user: AuthUser) -> tuple[str, int]:
        session_id = str(uuid.uuid4())
        token, expires_at = create_access_token(session_id=session_id, user=user)
        now = _utc_timestamp()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (id, user_id, token_hash, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, user.id, token_hash(token), now, expires_at),
            )
        return token, expires_at

    def revoke_session(self, token: str) -> None:
        now = _utc_timestamp()
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE sessions
                SET revoked_at = ?
                WHERE token_hash = ? AND revoked_at IS NULL
                """,
                (now, token_hash(token)),
            )

    def get_user_by_token(self, token: str) -> AuthUser | None:
        payload = decode_and_verify_token(token)
        if payload is None:
            return None
        session_id = str(payload.get("sid") or "")
        if not session_id:
            return None

        with self.connect() as connection:
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
                (session_id, token_hash(token), int(time.time())),
            ).fetchone()
        user = self._row_to_user(row)
        if user is None or not user.is_active:
            return None
        return user

    def create_conversation(self, *, user_id: str, title: str) -> dict[str, Any]:
        now = _utc_timestamp()
        conversation_id = str(uuid.uuid4())
        clean_title = title.strip()[:120] or "Cuộc trò chuyện mới"
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (
                    id, user_id, title, created_at, updated_at, last_message_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, user_id, clean_title, now, now, now),
            )
        conversation = self.get_conversation(user_id=user_id, conversation_id=conversation_id)
        if conversation is None:
            raise RuntimeError("Failed to create conversation.")
        return conversation

    def list_conversations(self, *, user_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    c.id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    c.last_message_at,
                    COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                WHERE c.user_id = ?
                GROUP BY c.id
                ORDER BY COALESCE(c.last_message_at, c.updated_at) DESC
                """,
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT
                    c.id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    c.last_message_at,
                    COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                WHERE c.user_id = ? AND c.id = ?
                GROUP BY c.id
                """,
                (user_id, conversation_id),
            ).fetchone()
        return dict(row) if row is not None else None

    def append_message(
        self,
        *,
        conversation_id: str,
        role: MessageRole,
        content: str,
        citations: Any | None = None,
        metadata: Any | None = None,
    ) -> dict[str, Any]:
        now = _utc_timestamp()
        message_id = str(uuid.uuid4())
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO messages (
                    id, conversation_id, role, content,
                    citations_json, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    role,
                    content,
                    _json_dumps(citations) if citations is not None else None,
                    _json_dumps(metadata) if metadata is not None else None,
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?, last_message_at = ?
                WHERE id = ?
                """,
                (now, now, conversation_id),
            )
        return {
            "id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "citations": citations,
            "metadata": metadata,
            "created_at": now,
        }

    def list_messages(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> list[dict[str, Any]] | None:
        if self.get_conversation(user_id=user_id, conversation_id=conversation_id) is None:
            return None

        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, conversation_id, role, content,
                       citations_json, metadata_json, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            ).fetchall()

        messages: list[dict[str, Any]] = []
        for row in rows:
            messages.append(
                {
                    "id": row["id"],
                    "conversation_id": row["conversation_id"],
                    "role": row["role"],
                    "content": row["content"],
                    "citations": json.loads(row["citations_json"])
                    if row["citations_json"]
                    else None,
                    "metadata": json.loads(row["metadata_json"])
                    if row["metadata_json"]
                    else None,
                    "created_at": row["created_at"],
                }
            )
        return messages

    def ensure_conversation_for_question(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        question: str,
    ) -> dict[str, Any]:
        if conversation_id:
            conversation = self.get_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if conversation is None:
                raise PermissionError("Conversation not found for current user.")
            return conversation

        title = question.strip().replace("\n", " ")
        if len(title) > 80:
            title = title[:77].rstrip() + "..."
        return self.create_conversation(user_id=user_id, title=title)


def user_payload(user: AuthUser) -> dict[str, Any]:
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
        "avatarUrl": user.avatar_url,
    }
