from __future__ import annotations

import json
import uuid
from typing import Any

from ..auth.models import MessageRole
from ..db.sqlite import SQLiteDatabase, utc_timestamp


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class ConversationRepository:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def create_conversation(self, *, user_id: str, title: str) -> dict[str, Any]:
        now = utc_timestamp()
        conversation_id = str(uuid.uuid4())
        clean_title = title.strip()[:120] or "Cuá»™c trÃ² chuyá»‡n má»›i"
        with self.database.connect() as connection:
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
        with self.database.connect() as connection:
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
        with self.database.connect() as connection:
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
        now = utc_timestamp()
        message_id = str(uuid.uuid4())
        with self.database.connect() as connection:
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

        with self.database.connect() as connection:
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
