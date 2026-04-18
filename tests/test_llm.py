from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from vn_labor_law_ai_assistant.llm import (
    chat_completion,
    default_model_for_provider,
    groq_response_format_for_model,
    normalize_provider,
    resolve_model_name,
)


class LLMTests(unittest.TestCase):
    def test_normalize_provider_validates_supported_values(self) -> None:
        self.assertEqual(normalize_provider("Groq"), "groq")
        self.assertEqual(normalize_provider("ollama"), "ollama")

        with self.assertRaises(ValueError):
            normalize_provider("openai")

    def test_resolve_model_name_uses_provider_defaults(self) -> None:
        self.assertEqual(resolve_model_name("ollama", ""), default_model_for_provider("ollama"))
        self.assertEqual(resolve_model_name("groq", None), default_model_for_provider("groq"))
        self.assertEqual(resolve_model_name("groq", "llama-3.3-70b-versatile"), "llama-3.3-70b-versatile")

    def test_groq_response_format_uses_schema_for_supported_models(self) -> None:
        response_format = groq_response_format_for_model("openai/gpt-oss-20b")

        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["strict"], True)
        self.assertIn("answer", response_format["json_schema"]["schema"]["properties"])

    def test_groq_response_format_falls_back_to_json_object(self) -> None:
        self.assertEqual(
            groq_response_format_for_model("llama-3.3-70b-versatile"),
            {"type": "json_object"},
        )

    @patch("vn_labor_law_ai_assistant.llm.require_ollama")
    def test_chat_completion_dispatches_to_ollama(self, mock_require_ollama: Mock) -> None:
        fake_ollama = Mock()
        fake_ollama.chat.return_value = {"message": {"content": '{"answer":"Co"}'}}
        mock_require_ollama.return_value = fake_ollama

        response = chat_completion(
            provider="ollama",
            model="qwen3:4b",
            messages=[{"role": "user", "content": "Xin chao"}],
            temperature=0,
        )

        self.assertEqual(response.provider, "ollama")
        self.assertEqual(response.model, "qwen3:4b")
        self.assertEqual(response.content, '{"answer":"Co"}')
        fake_ollama.chat.assert_called_once()

    @patch("vn_labor_law_ai_assistant.llm.build_groq_client")
    def test_chat_completion_dispatches_to_groq(self, mock_build_groq_client: Mock) -> None:
        fake_message = Mock(content='{"answer":"Co"}')
        fake_choice = Mock(message=fake_message)
        fake_client = Mock()
        fake_client.chat.completions.create.return_value = Mock(choices=[fake_choice])
        mock_build_groq_client.return_value = fake_client

        response = chat_completion(
            provider="groq",
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": "Xin chao"}],
            temperature=0,
        )

        self.assertEqual(response.provider, "groq")
        self.assertEqual(response.model, "openai/gpt-oss-20b")
        self.assertEqual(response.content, '{"answer":"Co"}')
        fake_client.chat.completions.create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
