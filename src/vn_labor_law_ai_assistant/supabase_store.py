from __future__ import annotations

from typing import Any

from .auth.models import AuthUser, MessageRole, Role
from .core.config import Settings, load_settings
from .db.postgres import SupabasePostgresDatabase
from .observability.models import ChatTraceCreate
from .observability.service import ChatTraceService


def _clean_title(title: str) -> str:
    return title.strip()[:120] or "New research"


def _jsonb(value: Any) -> Any:
    if value is None:
        return None
    try:
        from psycopg.types.json import Jsonb
    except ImportError:
        return value
    return Jsonb(value)


class SupabaseAppStore:
    def __init__(
        self,
        database: SupabasePostgresDatabase | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.database = database or SupabasePostgresDatabase(settings=self.settings)
        self.trace_service = ChatTraceService(
            repository=SupabaseChatTraceRepository(self.database)
        )

    def connect(self) -> Any:
        return self.database.connect()

    def upsert_external_user(
        self,
        *,
        user_id: str,
        name: str,
        email: str,
        auth_provider: str,
        provider_id: str,
        role: Role,
        avatar_url: str | None = None,
    ) -> AuthUser:
        normalized_email = email.strip().lower()
        clean_name = name.strip() or normalized_email
        with self.database.connect() as connection:
            row = connection.execute(
                """
                insert into public.profiles (
                    id, email, full_name, avatar_url, role, created_at, updated_at
                )
                values (%s, %s, %s, %s, %s, now(), now())
                on conflict (id) do update set
                    email = excluded.email,
                    full_name = excluded.full_name,
                    avatar_url = excluded.avatar_url,
                    role = excluded.role,
                    updated_at = now()
                returning id::text, email, full_name, avatar_url, role
                """,
                (user_id, normalized_email, clean_name, avatar_url, role),
            ).fetchone()
        if row is None:
            raise RuntimeError("Failed to upsert Supabase profile.")
        return self._row_to_user(row)

    def get_user_by_id(self, user_id: str) -> AuthUser | None:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                select id::text, email, full_name, avatar_url, role
                from public.profiles
                where id = %s
                """,
                (user_id,),
            ).fetchone()
        return self._row_to_user(row) if row is not None else None

    def get_user_by_email(self, email: str) -> AuthUser | None:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                select id::text, email, full_name, avatar_url, role
                from public.profiles
                where email = %s
                """,
                (email.strip().lower(),),
            ).fetchone()
        return self._row_to_user(row) if row is not None else None

    def create_conversation(self, *, user_id: str, title: str) -> dict[str, Any]:
        clean_title = _clean_title(title)
        with self.database.connect() as connection:
            row = connection.execute(
                """
                insert into public.conversations (
                    user_id, title, created_at, updated_at, last_message_at
                )
                values (%s, %s, now(), now(), now())
                returning id::text
                """,
                (user_id, clean_title),
            ).fetchone()
        if row is None:
            raise RuntimeError("Failed to create conversation.")
        conversation = self.get_conversation(
            user_id=user_id,
            conversation_id=str(row["id"]),
        )
        if conversation is None:
            raise RuntimeError("Failed to load created conversation.")
        return conversation

    def list_conversations(self, *, user_id: str) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                select
                    c.id::text as id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    c.last_message_at,
                    count(m.id)::int as message_count
                from public.conversations c
                left join public.messages m on m.conversation_id = c.id
                where c.user_id = %s
                group by c.id
                order by coalesce(c.last_message_at, c.updated_at) desc
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
                select
                    c.id::text as id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    c.last_message_at,
                    count(m.id)::int as message_count
                from public.conversations c
                left join public.messages m on m.conversation_id = c.id
                where c.user_id = %s and c.id = %s
                group by c.id
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
        with self.database.connect() as connection:
            row = connection.execute(
                """
                insert into public.messages (
                    conversation_id, role, content, citations_json, metadata_json, created_at
                )
                values (%s, %s, %s, %s, %s, now())
                returning
                    id::text,
                    conversation_id::text,
                    role,
                    content,
                    citations_json,
                    metadata_json,
                    created_at
                """,
                (
                    conversation_id,
                    role,
                    content,
                    _jsonb(citations),
                    _jsonb(metadata),
                ),
            ).fetchone()
            connection.execute(
                """
                update public.conversations
                set updated_at = now(), last_message_at = now()
                where id = %s
                """,
                (conversation_id,),
            )
        if row is None:
            raise RuntimeError("Failed to append message.")
        return self._message_row(row)

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
                select
                    id::text,
                    conversation_id::text,
                    role,
                    content,
                    citations_json,
                    metadata_json,
                    created_at
                from public.messages
                where conversation_id = %s
                order by created_at asc
                """,
                (conversation_id,),
            ).fetchall()
        return [self._message_row(row) for row in rows]

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

    def record_chat_trace(self, **kwargs: Any) -> dict[str, Any]:
        return self.trace_service.record_chat_trace(**kwargs)

    def list_recent_traces(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.trace_service.list_recent_traces(**kwargs)

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        return self.trace_service.get_trace(trace_id)

    def count_users(self) -> int:
        return self._count("select count(*)::int as count from public.profiles")

    def count_active_users(self) -> int:
        return self.count_users()

    def count_admin_users(self) -> int:
        return self._count(
            "select count(*)::int as count from public.profiles where role = 'admin'"
        )

    def count_active_sessions(self) -> int:
        return 0

    def count_conversations(self) -> int:
        return self._count("select count(*)::int as count from public.conversations")

    def count_messages(self) -> int:
        return self._count("select count(*)::int as count from public.messages")

    def count_traces(self) -> int:
        return self.trace_service.count_traces()

    def count_traces_with_errors(self) -> int:
        return self.trace_service.count_traces_with_errors()

    def count_insufficient_context_traces(self) -> int:
        return self.trace_service.count_insufficient_context_traces()

    def _count(self, sql: str) -> int:
        with self.database.connect() as connection:
            row = connection.execute(sql).fetchone()
        return int(row["count"]) if row is not None else 0

    def _row_to_user(self, row: dict[str, Any]) -> AuthUser:
        email = str(row["email"])
        return AuthUser(
            id=str(row["id"]),
            name=str(row.get("full_name") or email),
            email=email,
            role=str(row.get("role") or "user"),  # type: ignore[arg-type]
            avatar_url=str(row["avatar_url"]) if row.get("avatar_url") else None,
            is_active=True,
        )

    def _message_row(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(row["id"]),
            "conversation_id": str(row["conversation_id"]),
            "role": row["role"],
            "content": row["content"],
            "citations": row.get("citations_json"),
            "metadata": row.get("metadata_json"),
            "created_at": row["created_at"],
        }


class SupabaseChatTraceRepository:
    def __init__(self, database: SupabasePostgresDatabase) -> None:
        self.database = database

    def create_trace(self, trace: ChatTraceCreate) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                insert into public.chat_traces (
                    request_id, user_id, conversation_id, message_id,
                    question, provider, model, retrieve_only, insufficient_context,
                    latency_ms, retrieval_latency_ms, generation_latency_ms,
                    intent_json, retrieved_hits_json, selected_contexts_json,
                    citations_json, error, created_at
                )
                values (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now()
                )
                returning *
                """,
                (
                    trace.request_id,
                    trace.user_id,
                    trace.conversation_id,
                    trace.message_id,
                    trace.question,
                    trace.provider,
                    trace.model,
                    trace.retrieve_only,
                    trace.insufficient_context,
                    trace.latency_ms,
                    trace.retrieval_latency_ms,
                    trace.generation_latency_ms,
                    _jsonb(trace.intent or {}),
                    _jsonb(trace.retrieved_hits or []),
                    _jsonb(trace.selected_contexts or []),
                    _jsonb(trace.citations or {}),
                    trace.error,
                ),
            ).fetchone()
        if row is None:
            raise RuntimeError("Failed to create chat trace.")
        return self._row_to_detail(row)

    def list_recent_traces(
        self,
        *,
        limit: int = 50,
        user_id: str | None = None,
        conversation_id: str | None = None,
        insufficient_only: bool = False,
        error_only: bool = False,
    ) -> list[dict[str, Any]]:
        where_clauses: list[str] = []
        params: list[Any] = []
        if user_id:
            where_clauses.append("user_id = %s")
            params.append(user_id)
        if conversation_id:
            where_clauses.append("conversation_id = %s")
            params.append(conversation_id)
        if insufficient_only:
            where_clauses.append("insufficient_context is true")
        if error_only:
            where_clauses.append("error is not null and btrim(error) <> ''")

        where_sql = f"where {' and '.join(where_clauses)}" if where_clauses else ""
        params.append(max(1, min(int(limit), 200)))
        with self.database.connect() as connection:
            rows = connection.execute(
                f"""
                select *
                from public.chat_traces
                {where_sql}
                order by created_at desc
                limit %s
                """,
                tuple(params),
            ).fetchall()
        return [self._row_to_summary(row) for row in rows]

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "select * from public.chat_traces where id = %s",
                (trace_id,),
            ).fetchone()
        return self._row_to_detail(row) if row is not None else None

    def count_traces(self) -> int:
        return self._count("select count(*)::int as count from public.chat_traces")

    def count_traces_with_errors(self) -> int:
        return self._count(
            """
            select count(*)::int as count
            from public.chat_traces
            where error is not null and btrim(error) <> ''
            """
        )

    def count_insufficient_context_traces(self) -> int:
        return self._count(
            """
            select count(*)::int as count
            from public.chat_traces
            where insufficient_context is true
            """
        )

    def _count(self, sql: str) -> int:
        with self.database.connect() as connection:
            row = connection.execute(sql).fetchone()
        return int(row["count"]) if row is not None else 0

    def _row_to_summary(self, row: dict[str, Any]) -> dict[str, Any]:
        citations = row.get("citations_json") or {}
        selected_contexts = row.get("selected_contexts_json") or []
        legal_basis = citations.get("legal_basis") if isinstance(citations, dict) else []
        return {
            "id": str(row["id"]),
            "requestId": row.get("request_id"),
            "userId": str(row["user_id"]),
            "conversationId": str(row["conversation_id"])
            if row.get("conversation_id")
            else None,
            "messageId": str(row["message_id"]) if row.get("message_id") else None,
            "question": row["question"],
            "provider": row.get("provider"),
            "model": row.get("model"),
            "retrieveOnly": bool(row.get("retrieve_only")),
            "insufficientContext": bool(row.get("insufficient_context")),
            "latencyMs": row.get("latency_ms"),
            "retrievalLatencyMs": row.get("retrieval_latency_ms"),
            "generationLatencyMs": row.get("generation_latency_ms"),
            "citationCount": len(legal_basis) if isinstance(legal_basis, list) else 0,
            "selectedContextCount": len(selected_contexts)
            if isinstance(selected_contexts, list)
            else 0,
            "error": row.get("error"),
            "createdAt": row["created_at"],
        }

    def _row_to_detail(self, row: dict[str, Any]) -> dict[str, Any]:
        summary = self._row_to_summary(row)
        summary.update(
            {
                "intent": row.get("intent_json") or {},
                "retrievedHits": row.get("retrieved_hits_json") or [],
                "selectedContexts": row.get("selected_contexts_json") or [],
                "citations": row.get("citations_json") or {},
            }
        )
        return summary
