from __future__ import annotations

from typing import Any

from ..auth.models import MessageRole
from .repository import ConversationRepository


class ConversationService:
    def __init__(self, repository: ConversationRepository) -> None:
        self.repository = repository

    def create_conversation(self, *, user_id: str, title: str) -> dict[str, Any]:
        return self.repository.create_conversation(user_id=user_id, title=title)

    def list_conversations(self, *, user_id: str) -> list[dict[str, Any]]:
        return self.repository.list_conversations(user_id=user_id)

    def get_conversation(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        return self.repository.get_conversation(
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
        return self.repository.append_message(
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
        return self.repository.list_messages(
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
