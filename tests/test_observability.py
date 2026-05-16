from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from helpers import create_test_auth_store
from vn_labor_law_ai_assistant.observability.models import ChatTraceCreate
from vn_labor_law_ai_assistant.observability.repository import ChatTraceRepository


class ChatTraceRepositoryTest(TestCase):
    def setUp(self) -> None:
        self.tmpdir = TemporaryDirectory()
        self.store = create_test_auth_store(Path(self.tmpdir.name) / "app.db")
        self.repository = ChatTraceRepository(self.store.database)
        user = self.store.get_user_by_email("user@example.com")
        self.assertIsNotNone(user)
        assert user is not None
        self.user = user

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_trace_repository_create_and_list(self) -> None:
        trace = self.repository.create_trace(
            ChatTraceCreate(
                request_id="request-1",
                user_id=self.user.id,
                conversation_id="conversation-1",
                message_id="message-1",
                question="Nguoi lao dong nghi viec the nao?",
                provider="groq",
                model="qwen/qwen3-32b",
                retrieve_only=False,
                insufficient_context=True,
                latency_ms=123,
                retrieval_latency_ms=20,
                generation_latency_ms=90,
                intent={"article_numbers": ["35"]},
                retrieved_hits=[{"chunkId": "chunk-1", "score": 0.5}],
                selected_contexts=[{"chunkId": "chunk-1", "textPreview": "preview"}],
                citations={"legal_basis": ["Bo luat"], "evidence_quotes": []},
                error="short error",
            )
        )

        traces = self.repository.list_recent_traces(
            limit=10,
            user_id=self.user.id,
            insufficient_only=True,
            error_only=True,
        )

        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["id"], trace["id"])
        self.assertEqual(traces[0]["requestId"], "request-1")
        self.assertEqual(traces[0]["citationCount"], 1)
        self.assertEqual(traces[0]["selectedContextCount"], 1)
        self.assertEqual(self.repository.count_traces(), 1)
        self.assertEqual(self.repository.count_traces_with_errors(), 1)
        self.assertEqual(self.repository.count_insufficient_context_traces(), 1)

    def test_trace_repository_get_detail(self) -> None:
        trace = self.repository.create_trace(
            ChatTraceCreate(
                user_id=self.user.id,
                question="Trace detail?",
                intent={"query_types": ["definition"]},
                retrieved_hits=[{"chunkId": "chunk-2"}],
                selected_contexts=[{"chunkId": "chunk-2"}],
                citations={
                    "legal_basis": ["citation"],
                    "evidence_quotes": [{"citation": "citation", "quote": "quote"}],
                },
            )
        )

        detail = self.repository.get_trace(trace["id"])

        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertEqual(detail["intent"]["query_types"], ["definition"])
        self.assertEqual(detail["retrievedHits"][0]["chunkId"], "chunk-2")
        self.assertEqual(detail["selectedContexts"][0]["chunkId"], "chunk-2")
        self.assertEqual(detail["citations"]["legal_basis"], ["citation"])
