from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from vn_labor_law_ai_assistant.embeddings import (
    embed_texts_via_http,
    embedding_api_timeout_seconds,
    embedding_provider,
    is_custom_http_embedding_provider,
    normalize_embedding_api_url,
)


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


class EmbeddingHttpTests(unittest.TestCase):
    def test_embedding_provider_defaults_to_sentence_transformers(self) -> None:
        with patch.dict("os.environ", {"EMBEDDING_PROVIDER": ""}, clear=False):
            self.assertEqual(embedding_provider(), "sentence_transformers")
            self.assertFalse(is_custom_http_embedding_provider())

    def test_embedding_provider_accepts_custom_http(self) -> None:
        with patch.dict("os.environ", {"EMBEDDING_PROVIDER": "custom_http"}, clear=False):
            self.assertTrue(is_custom_http_embedding_provider())

    def test_timeout_uses_env_value(self) -> None:
        with patch.dict("os.environ", {"EMBEDDING_API_TIMEOUT_SECONDS": "12.5"}, clear=False):
            self.assertEqual(embedding_api_timeout_seconds(), 12.5)

    def test_normalize_embedding_api_url_converts_space_page_url(self) -> None:
        self.assertEqual(
            normalize_embedding_api_url("https://huggingface.co/spaces/chuong0306/my-api-embedding"),
            "https://chuong0306-my-api-embedding.hf.space/v1/embeddings",
        )

    def test_normalize_embedding_api_url_appends_default_path_for_hf_space_root(self) -> None:
        self.assertEqual(
            normalize_embedding_api_url("https://chuong0306-my-api-embedding.hf.space/"),
            "https://chuong0306-my-api-embedding.hf.space/v1/embeddings",
        )

    def test_embed_texts_via_http_sends_batch_request(self) -> None:
        requests = []

        def fake_urlopen(http_request, timeout):
            requests.append((http_request, timeout))
            body = json.loads(http_request.data.decode("utf-8"))
            self.assertEqual(body, {"input": ["cau mot", "cau hai"]})
            self.assertEqual(http_request.get_header("Authorization"), "Bearer hf_test")
            return FakeResponse({"embedding": [[1, 2], [3.5, 4]], "model": "test-model"})

        with (
            patch.dict(
                "os.environ",
                {
                    "EMBEDDING_API_URL": "https://embedding.example/v1/embeddings",
                    "EMBEDDING_API_TOKEN": "hf_test",
                    "EMBEDDING_API_TIMEOUT_SECONDS": "8",
                },
                clear=False,
            ),
            patch("vn_labor_law_ai_assistant.embeddings.request.urlopen", side_effect=fake_urlopen),
        ):
            vectors = embed_texts_via_http(["cau mot", "cau hai"], batch_size=2)

        self.assertEqual(vectors, [[1.0, 2.0], [3.5, 4.0]])
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0][0].full_url, "https://embedding.example/v1/embeddings")
        self.assertEqual(requests[0][1], 8)

    def test_embed_texts_via_http_normalizes_single_vector_response(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {"EMBEDDING_API_URL": "https://embedding.example/v1/embeddings"},
                clear=False,
            ),
            patch(
                "vn_labor_law_ai_assistant.embeddings.request.urlopen",
                return_value=FakeResponse({"embedding": [0.25, -0.5], "model": "test-model"}),
            ),
        ):
            vectors = embed_texts_via_http(["mot cau"], batch_size=1)

        self.assertEqual(vectors, [[0.25, -0.5]])

    def test_embed_texts_via_http_requires_url(self) -> None:
        with patch.dict("os.environ", {"EMBEDDING_API_URL": ""}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "EMBEDDING_API_URL"):
                embed_texts_via_http(["mot cau"])


if __name__ == "__main__":
    unittest.main()
