from __future__ import annotations

from unittest import TestCase

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vn_labor_law_ai_assistant.api import app
from vn_labor_law_ai_assistant.api.routes.chat import extract_last_user_message
from vn_labor_law_ai_assistant.api.schemas import ChatRequest


class ApiModularizationTest(TestCase):
    def test_legacy_api_import_exposes_app(self) -> None:
        self.assertIsInstance(app, FastAPI)

    def test_health_endpoint(self) -> None:
        response = TestClient(app).get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_extract_last_user_message(self) -> None:
        payload = {
            "messages": [
                {"role": "assistant", "content": "ignored"},
                {"role": "user", "content": " first "},
                {"role": "user", "content": " latest question "},
            ]
        }

        self.assertEqual(extract_last_user_message(payload), "latest question")
        self.assertEqual(extract_last_user_message({"messages": []}), "")
        self.assertEqual(extract_last_user_message({"messages": "invalid"}), "")

    def test_chat_request_parses_camel_case_fields(self) -> None:
        request = ChatRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hoi ve nghi viec"}],
                "conversationId": "conversation-1",
                "topK": 4,
                "prefetchLimit": 12,
                "maxContexts": 3,
                "maxContextChars": 1000,
                "maxContextTokens": 300,
                "includeCitations": False,
                "retrieveOnly": True,
            }
        )

        self.assertEqual(request.conversation_id, "conversation-1")
        self.assertEqual(request.top_k, 4)
        self.assertEqual(request.prefetch_limit, 12)
        self.assertEqual(request.max_contexts, 3)
        self.assertEqual(request.max_context_chars, 1000)
        self.assertEqual(request.max_context_tokens, 300)
        self.assertFalse(request.include_citations)
        self.assertTrue(request.retrieve_only)
        self.assertEqual(
            request.model_dump(by_alias=True)["conversationId"],
            "conversation-1",
        )
