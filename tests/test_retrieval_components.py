from __future__ import annotations

import unittest

from vn_labor_law_ai_assistant.rag.retrieval import SemanticReranker
from vn_labor_law_ai_assistant.retriever import (
    HybridRetriever,
    RetrievalContext,
    SearchHit,
    format_context_for_prompt,
    record_from_qdrant_payload,
    select_contexts_for_prompt,
)


class RetrieverCompatibilityExportTests(unittest.TestCase):
    def test_retriever_compat_exports(self) -> None:
        self.assertEqual(HybridRetriever.__name__, "HybridRetriever")
        self.assertEqual(RetrievalContext.__name__, "RetrievalContext")
        self.assertEqual(SearchHit.__name__, "SearchHit")
        self.assertTrue(callable(format_context_for_prompt))


class RetrievalComponentTests(unittest.TestCase):
    def test_record_from_qdrant_payload_requires_chunk_id(self) -> None:
        with self.assertRaises(ValueError):
            record_from_qdrant_payload({"citation_text": "Dieu 46"})

    def test_select_contexts_for_prompt_basic_budget_behavior(self) -> None:
        contexts = (
            RetrievalContext(
                chunk_id="chunk-a",
                citation_text="Dieu 46",
                text="A" * 40,
                payload={"document_id": "doc", "article_number": "46"},
                score=0.9,
                matched_chunk_ids=("chunk-a",),
                matched_citations=("Dieu 46",),
            ),
            RetrievalContext(
                chunk_id="chunk-b",
                citation_text="Dieu 47",
                text="B" * 40,
                payload={"document_id": "doc", "article_number": "47"},
                score=0.8,
                matched_chunk_ids=("chunk-b",),
                matched_citations=("Dieu 47",),
            ),
        )

        selected = select_contexts_for_prompt(contexts, max_contexts=1, max_chars=1000)

        self.assertEqual(selected, contexts[:1])

    def test_semantic_reranker_disabled_returns_same_hits(self) -> None:
        hits = (
            SearchHit("chunk-a", "point-a", 0.9, "Dieu 41", {"chunk_id": "chunk-a"}),
            SearchHit("chunk-b", "point-b", 0.8, "Dieu 35", {"chunk_id": "chunk-b"}),
        )
        reranker = SemanticReranker(model_name="", device="cpu")

        self.assertEqual(reranker.semantic_rerank_hits("Cau hoi", hits, {}), hits)


if __name__ == "__main__":
    unittest.main()

