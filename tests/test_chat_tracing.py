from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import vn_labor_law_ai_assistant.api.deps as api_deps
from vn_labor_law_ai_assistant.api import app
from vn_labor_law_ai_assistant.core.config import get_settings
from vn_labor_law_ai_assistant.heuristic_router import QueryIntent
from vn_labor_law_ai_assistant.observability import ChatTraceService
from vn_labor_law_ai_assistant.retriever import (
    RetrievalContext,
    RetrievalResult,
    SearchHit,
)
from helpers import create_test_auth_store


class FakeRetriever:
    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        prefetch_limit: int,
    ) -> RetrievalResult:
        citation = "Bo luat so 45/2019/QH14, Dieu 35, khoan 1"
        payload = {
            "chunk_id": "chunk-1",
            "citation_text": citation,
            "document_id": "45-2019-qh14",
            "article_number": "35",
            "clause_ref": "1",
            "topic": ["cham_dut_hop_dong_lao_dong"],
            "issue_type": ["don_phuong_cham_dut"],
        }
        return RetrievalResult(
            query=query,
            intent=QueryIntent(
                raw_query=query,
                normalized_query="nguoi lao dong nghi viec",
                actor_filters=("employee",),
                topic_filters=("cham_dut_hop_dong_lao_dong",),
                issue_filters=("don_phuong_cham_dut",),
                document_filters=("45-2019-qh14",),
                article_numbers=("35",),
            ),
            hits=(
                SearchHit(
                    chunk_id="chunk-1",
                    qdrant_point_id="point-1",
                    score=0.91,
                    citation_text=citation,
                    payload=payload,
                ),
            ),
            contexts=(
                RetrievalContext(
                    chunk_id="chunk-1",
                    citation_text=citation,
                    text="Nguoi lao dong co quyen don phuong cham dut hop dong.",
                    payload=payload,
                    score=0.91,
                    matched_chunk_ids=("chunk-1",),
                    matched_citations=(citation,),
                ),
            ),
        )


class EmptyRetriever:
    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        prefetch_limit: int,
    ) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            intent=QueryIntent(
                raw_query=query,
                normalized_query=query.lower(),
                actor_filters=(),
                topic_filters=(),
                issue_filters=(),
                document_filters=(),
            ),
            hits=(),
            contexts=(),
        )


class ChatTracingTest(TestCase):
    def setUp(self) -> None:
        self.env_patch = patch.dict(
            os.environ,
            {
                "APP_ENV": "development",
                "ENVIRONMENT": "",
                "AUTH_PROVIDER": "local",
                "APP_DATA_BACKEND": "sqlite",
                "AUTH_SEED_DEFAULT_USERS": "0",
            },
            clear=False,
        )
        self.env_patch.start()
        get_settings.cache_clear()
        self.tmpdir = TemporaryDirectory()
        self.store = create_test_auth_store(Path(self.tmpdir.name) / "app.db")
        self.previous_auth_store = api_deps._auth_store
        api_deps._auth_store = self.store
        self.client = TestClient(app)

    def tearDown(self) -> None:
        api_deps._auth_store = self.previous_auth_store
        self.tmpdir.cleanup()
        self.env_patch.stop()
        get_settings.cache_clear()

    def _token_for(self, email: str, password: str) -> str:
        user = self.store.authenticate_user(email, password)
        self.assertIsNotNone(user)
        assert user is not None
        token, _ = self.store.create_session(user)
        return token

    def test_chat_retrieve_only_records_trace(self) -> None:
        token = self._token_for("user@example.com", "user12345")

        with patch(
            "vn_labor_law_ai_assistant.api.routes.chat.get_retriever",
            return_value=FakeRetriever(),
        ):
            response = self.client.post(
                "/chat",
                json={
                    "messages": [{"role": "user", "content": "Toi muon nghi viec"}],
                    "retrieveOnly": True,
                },
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Request-Id": "request-chat-trace",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("x-request-id"), "request-chat-trace")
        traces = ChatTraceService(self.store.database).list_recent_traces(limit=10)
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["requestId"], "request-chat-trace")
        self.assertTrue(traces[0]["retrieveOnly"])
        self.assertEqual(traces[0]["selectedContextCount"], 1)
        detail = ChatTraceService(self.store.database).get_trace(traces[0]["id"])
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertEqual(detail["intent"]["article_numbers"], ["35"])
        self.assertEqual(detail["retrievedHits"][0]["chunkId"], "chunk-1")
        self.assertEqual(detail["selectedContexts"][0]["chunkId"], "chunk-1")

    def test_chat_trace_best_effort_does_not_break_chat(self) -> None:
        token = self._token_for("user@example.com", "user12345")

        with (
            patch(
                "vn_labor_law_ai_assistant.api.routes.chat.get_retriever",
                return_value=FakeRetriever(),
            ),
            patch.object(
                self.store,
                "record_chat_trace",
                side_effect=RuntimeError("trace database unavailable"),
            ),
            patch("vn_labor_law_ai_assistant.api.routes.chat.LOGGER.warning"),
        ):
            response = self.client.post(
                "/chat",
                json={
                    "messages": [{"role": "user", "content": "Toi muon nghi viec"}],
                    "retrieveOnly": True,
                },
                headers={"Authorization": f"Bearer {token}"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("X-Request-Id", response.headers)

    def test_chat_json_response_includes_sources(self) -> None:
        token = self._token_for("user@example.com", "user12345")

        with patch(
            "vn_labor_law_ai_assistant.api.routes.chat.get_retriever",
            return_value=FakeRetriever(),
        ):
            response = self.client.post(
                "/chat",
                json={
                    "messages": [{"role": "user", "content": "Toi muon nghi viec"}],
                    "provider": "extractive",
                    "responseFormat": "json",
                },
                headers={"Authorization": f"Bearer {token}"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("answer", payload)
        self.assertEqual(
            payload["legalBasis"],
            ["Bo luat so 45/2019/QH14, Dieu 35, khoan 1"],
        )
        self.assertEqual(
            payload["citations"]["legal_basis"],
            ["Bo luat so 45/2019/QH14, Dieu 35, khoan 1"],
        )
        self.assertEqual(len(payload["evidenceQuotes"]), 1)
        self.assertIn("Nguoi lao dong", payload["evidenceQuotes"][0]["quote"])

    def test_chat_no_context_fallback_uses_valid_vietnamese_text(self) -> None:
        token = self._token_for("user@example.com", "user12345")

        with patch(
            "vn_labor_law_ai_assistant.api.routes.chat.get_retriever",
            return_value=EmptyRetriever(),
        ):
            response = self.client.post(
                "/chat",
                json={"messages": [{"role": "user", "content": "Hoi ve nghi viec"}]},
                headers={"Authorization": f"Bearer {token}"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "Không tìm thấy ngữ cảnh phù hợp trong index.")
        self.assertNotIn("Ã", response.text)

    def test_chat_rejects_out_of_domain_question_before_retrieval(self) -> None:
        token = self._token_for("user@example.com", "user12345")
        retriever_factory = Mock()

        with patch(
            "vn_labor_law_ai_assistant.api.routes.chat.get_retriever",
            retriever_factory,
        ):
            response = self.client.post(
                "/chat",
                json={"messages": [{"role": "user", "content": "co tft"}]},
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Request-Id": "request-out-of-domain",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("ngoai pham vi", response.text)
        retriever_factory.assert_not_called()

        traces = ChatTraceService(self.store.database).list_recent_traces(limit=10)
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["requestId"], "request-out-of-domain")
        self.assertTrue(traces[0]["insufficientContext"])
        self.assertEqual(traces[0]["selectedContextCount"], 0)
