from __future__ import annotations

import json
import unittest
from unittest.mock import Mock, patch

from vn_labor_law_ai_assistant.llm import (
    build_azure_openai_payload,
    chat_completion,
    create_groq_chat_completion_with_retries,
    DEFAULT_AZURE_OPENAI_MODEL,
    DEFAULT_GROQ_BENCHMARK_JUDGE_MODEL,
    default_benchmark_judge_model,
    default_benchmark_judge_provider,
    default_model_for_provider,
    extract_azure_openai_response_content,
    groq_rate_limit_sleep_seconds,
    groq_response_format_fallbacks,
    groq_response_format_for_model,
    is_azure_openai_content_filter_error,
    is_groq_json_validation_error,
    is_groq_rate_limit_error,
    normalize_provider,
    resolve_model_name,
)


class LLMTests(unittest.TestCase):
    class FakeRateLimitError(Exception):
        def __init__(self, message: str, status_code: int = 429) -> None:
            super().__init__(message)
            self.status_code = status_code

    class FakeAPIError(Exception):
        def __init__(self, message: str, status_code: int = 500) -> None:
            super().__init__(message)
            self.status_code = status_code

    class FakeJSONValidationError(Exception):
        def __init__(self, message: str, status_code: int = 400) -> None:
            super().__init__(message)
            self.status_code = status_code

    class FakeHTTPError(Exception):
        def __init__(
            self,
            message: str,
            *,
            status_code: int = 400,
            response_body: str = "",
        ) -> None:
            super().__init__(message)
            self.status_code = status_code
            self.response_body = response_body

    def test_normalize_provider_validates_supported_values(self) -> None:
        self.assertEqual(normalize_provider("Groq"), "groq")
        self.assertEqual(normalize_provider("azure_openai"), "azure_openai")

        with self.assertRaises(ValueError):
            normalize_provider("ollama")

    def test_resolve_model_name_uses_provider_defaults(self) -> None:
        self.assertEqual(resolve_model_name("groq", None), default_model_for_provider("groq"))
        self.assertEqual(resolve_model_name("azure_openai", None), DEFAULT_AZURE_OPENAI_MODEL)
        self.assertEqual(resolve_model_name("groq", "llama-3.3-70b-versatile"), "llama-3.3-70b-versatile")

    def test_default_benchmark_judge_provider_defaults_to_groq(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(default_benchmark_judge_provider(), "groq")

    def test_default_benchmark_judge_provider_rejects_unsupported_override(self) -> None:
        with patch.dict("os.environ", {"BENCHMARK_JUDGE_PROVIDER": "ollama"}, clear=True):
            with self.assertRaises(ValueError):
                default_benchmark_judge_provider()

    def test_default_benchmark_judge_provider_accepts_azure_openai(self) -> None:
        with patch.dict("os.environ", {"BENCHMARK_JUDGE_PROVIDER": "azure_openai"}, clear=True):
            self.assertEqual(default_benchmark_judge_provider(), "azure_openai")

    def test_default_benchmark_judge_model_defaults_to_groq_judge_model(self) -> None:
        with patch.dict("os.environ", {"BENCHMARK_JUDGE_MODEL": ""}, clear=True):
            self.assertEqual(
                default_benchmark_judge_model("groq"),
                DEFAULT_GROQ_BENCHMARK_JUDGE_MODEL,
            )

    def test_default_benchmark_judge_model_defaults_to_azure_model(self) -> None:
        with patch.dict("os.environ", {"BENCHMARK_JUDGE_MODEL": ""}, clear=True):
            self.assertEqual(
                default_benchmark_judge_model("azure_openai"),
                DEFAULT_AZURE_OPENAI_MODEL,
            )

    def test_default_benchmark_judge_model_reads_override(self) -> None:
        with patch.dict(
            "os.environ",
            {"BENCHMARK_JUDGE_MODEL": "openai/gpt-oss-120b"},
            clear=True,
        ):
            self.assertEqual(default_benchmark_judge_model("groq"), "openai/gpt-oss-120b")

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

    def test_groq_response_format_accepts_custom_schema(self) -> None:
        response_format = groq_response_format_for_model(
            "openai/gpt-oss-20b",
            json_schema={"type": "object", "properties": {"score": {"type": "integer"}}},
            schema_name="benchmark_judge",
        )

        self.assertEqual(response_format["json_schema"]["name"], "benchmark_judge")
        self.assertIn("score", response_format["json_schema"]["schema"]["properties"])

    def test_groq_response_format_fallbacks_include_plain_text_fallback(self) -> None:
        self.assertEqual(
            groq_response_format_fallbacks("qwen/qwen3-32b"),
            ({"type": "json_object"}, None),
        )
        self.assertEqual(
            groq_response_format_fallbacks("openai/gpt-oss-20b")[1:],
            ({"type": "json_object"}, None),
        )

    def test_is_groq_rate_limit_error_detects_429(self) -> None:
        self.assertTrue(
            is_groq_rate_limit_error(
                self.FakeRateLimitError("Rate limit reached. Please try again in 250ms.")
            )
        )
        self.assertFalse(is_groq_rate_limit_error(self.FakeAPIError("Internal server error")))

    def test_is_groq_json_validation_error_detects_400(self) -> None:
        self.assertTrue(
            is_groq_json_validation_error(
                self.FakeJSONValidationError("Error code: 400 - {'code': 'json_validate_failed'}")
            )
        )
        self.assertFalse(is_groq_json_validation_error(self.FakeAPIError("Internal server error")))

    def test_is_azure_openai_content_filter_error_detects_400_filter(self) -> None:
        self.assertTrue(
            is_azure_openai_content_filter_error(
                self.FakeHTTPError(
                    "Azure OpenAI request failed",
                    response_body='{"error":{"code":"content_filter"}}',
                )
            )
        )
        self.assertFalse(
            is_azure_openai_content_filter_error(
                self.FakeHTTPError("Azure OpenAI request failed", status_code=500)
            )
        )

    def test_build_azure_openai_payload_uses_responses_api_shape(self) -> None:
        payload = build_azure_openai_payload(
            model="GPT-5.4-MINI",
            messages=[{"role": "user", "content": "Cham diem JSON"}],
            temperature=0,
            text_format={"type": "json_object"},
        )

        self.assertEqual(payload["model"], "GPT-5.4-MINI")
        self.assertEqual(payload["input"], [{"role": "user", "content": "Cham diem JSON"}])
        self.assertEqual(payload["text"], {"format": {"type": "json_object"}})

    def test_extract_azure_openai_response_content_reads_output_text(self) -> None:
        self.assertEqual(
            extract_azure_openai_response_content({"output_text": '{"score":5}'}),
            '{"score":5}',
        )
        self.assertEqual(
            extract_azure_openai_response_content(
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": '{"score":4}'},
                            ],
                        }
                    ]
                }
            ),
            '{"score":4}',
        )

    def test_groq_rate_limit_sleep_seconds_uses_retry_after_hint(self) -> None:
        delay = groq_rate_limit_sleep_seconds(
            self.FakeRateLimitError("Please try again in 250ms."),
            attempt=0,
        )

        self.assertEqual(delay, 0.25)

    @patch("vn_labor_law_ai_assistant.llm.time.sleep")
    def test_create_groq_chat_completion_with_retries_retries_on_rate_limit(
        self,
        mock_sleep: Mock,
    ) -> None:
        fake_message = Mock(content='{"answer":"Co"}')
        fake_choice = Mock(message=fake_message)
        fake_client = Mock()
        fake_client.chat.completions.create.side_effect = [
            self.FakeRateLimitError("Rate limit reached. Please try again in 250ms."),
            Mock(choices=[fake_choice]),
        ]

        response = create_groq_chat_completion_with_retries(
            client=fake_client,
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "Xin chao"}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        self.assertEqual(response.choices[0].message.content, '{"answer":"Co"}')
        self.assertEqual(fake_client.chat.completions.create.call_count, 2)
        mock_sleep.assert_called_once_with(0.25)

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
            json_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
            json_schema_name="custom_schema",
        )

        self.assertEqual(response.provider, "groq")
        self.assertEqual(response.model, "openai/gpt-oss-20b")
        self.assertEqual(response.content, '{"answer":"Co"}')
        fake_client.chat.completions.create.assert_called_once()
        kwargs = fake_client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["response_format"]["json_schema"]["name"], "custom_schema")

    @patch("vn_labor_law_ai_assistant.llm.build_groq_client")
    def test_chat_completion_falls_back_to_plain_text_after_json_validation_error(
        self,
        mock_build_groq_client: Mock,
    ) -> None:
        fake_message = Mock(content='{"answer":"Co"}')
        fake_choice = Mock(message=fake_message)
        fake_client = Mock()
        fake_client.chat.completions.create.side_effect = [
            self.FakeJSONValidationError(
                "Error code: 400 - {'message': 'Failed to validate JSON.', 'code': 'json_validate_failed'}"
            ),
            Mock(choices=[fake_choice]),
        ]
        mock_build_groq_client.return_value = fake_client

        response = chat_completion(
            provider="groq",
            model="qwen/qwen3-32b",
            messages=[{"role": "user", "content": "Xin chao"}],
            temperature=0,
        )

        self.assertEqual(response.provider, "groq")
        self.assertEqual(response.model, "qwen/qwen3-32b")
        self.assertEqual(response.content, '{"answer":"Co"}')
        self.assertEqual(fake_client.chat.completions.create.call_count, 2)
        first_kwargs = fake_client.chat.completions.create.call_args_list[0].kwargs
        second_kwargs = fake_client.chat.completions.create.call_args_list[1].kwargs
        self.assertEqual(first_kwargs["response_format"], {"type": "json_object"})
        self.assertNotIn("response_format", second_kwargs)

    @patch("vn_labor_law_ai_assistant.llm.urlopen")
    def test_chat_completion_dispatches_to_azure_openai_responses(
        self,
        mock_urlopen: Mock,
    ) -> None:
        fake_http_response = Mock()
        fake_http_response.read.return_value = json.dumps(
            {"output": [{"content": [{"type": "output_text", "text": '{"score":5}'}]}]}
        ).encode("utf-8")
        fake_http_response.__enter__ = Mock(return_value=fake_http_response)
        fake_http_response.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = fake_http_response

        with patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_API_KEY": "test-key",
                "AZURE_OPENAI_RESPONSES_ENDPOINT": (
                    "https://example.openai.azure.com/openai/responses"
                ),
                "AZURE_OPENAI_API_VERSION": "2025-04-01-preview",
            },
            clear=True,
        ):
            response = chat_completion(
                provider="azure_openai",
                model="GPT-5.4-MINI",
                messages=[{"role": "user", "content": "Cham diem"}],
                temperature=0,
                json_schema={"type": "object", "properties": {"score": {"type": "integer"}}},
                json_schema_name="judge_score",
            )

        self.assertEqual(response.provider, "azure_openai")
        self.assertEqual(response.model, "GPT-5.4-MINI")
        self.assertEqual(response.content, '{"score":5}')
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://example.openai.azure.com/openai/responses?api-version=2025-04-01-preview",
        )
        request_payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request_payload["model"], "GPT-5.4-MINI")
        self.assertEqual(request_payload["text"]["format"]["type"], "json_schema")
        self.assertEqual(request_payload["text"]["format"]["name"], "judge_score")


if __name__ == "__main__":
    unittest.main()
