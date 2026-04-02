from __future__ import annotations

import unittest

from vn_labor_law_ai_assistant.retriever import (
    HybridRetriever,
    RetrievedRecord,
    RetrievalContext,
    SearchHit,
    dedupe_preserve_order,
    format_context_for_prompt,
    route_query,
)


class QueryRoutingTests(unittest.TestCase):
    def test_route_query_extracts_filters_and_legal_refs(self) -> None:
        intent = route_query(
            "Tôi bị công ty đuổi việc trái luật, muốn đòi bồi thường theo Điều 41 thì làm thế nào?"
        )

        self.assertIn("nguoi_lao_dong", intent.actor_filters)
        self.assertIn("nguoi_su_dung_lao_dong", intent.actor_filters)
        self.assertIn("cham_dut_hop_dong_lao_dong", intent.topic_filters)
        self.assertIn("trai_phap_luat", intent.issue_filters)
        self.assertIn("boi_thuong", intent.issue_filters)
        self.assertEqual(intent.article_number, "41")


class RetrievalAssemblyTests(unittest.TestCase):
    def test_dedupe_preserve_order_keeps_first_occurrence(self) -> None:
        self.assertEqual(
            dedupe_preserve_order(("a", "b", "a", "c", "b")),
            ("a", "b", "c"),
        )

    def test_format_context_for_prompt_includes_context_blocks(self) -> None:
        context = RetrievalContext(
            chunk_id="chunk-1",
            citation_text="Bộ luật số 45/2019/QH 14, Điều 46",
            text="Người sử dụng lao động có trách nhiệm chi trả trợ cấp thôi việc...",
            payload={"level": "clause"},
            score=0.9,
            matched_chunk_ids=("chunk-1",),
            matched_citations=("Bộ luật số 45/2019/QH 14, Điều 46",),
        )

        prompt = format_context_for_prompt((context,))

        self.assertIn("[NGU CANH 1]", prompt)
        self.assertIn("Co so phap ly: Bộ luật số 45/2019/QH 14, Điều 46", prompt)
        self.assertIn("Người sử dụng lao động có trách nhiệm", prompt)

    def test_assemble_contexts_deduplicates_shared_parent(self) -> None:
        parent = RetrievedRecord(
            chunk_id="parent-1",
            parent_chunk_id=None,
            citation_text="Bộ luật số 45/2019/QH 14, Điều 36, khoản 1",
            text="Người sử dụng lao động có quyền đơn phương chấm dứt hợp đồng trong các trường hợp sau đây...",
            dense_text="parent dense",
            sparse_text="parent sparse",
            payload={"level": "clause"},
        )
        child_a = RetrievedRecord(
            chunk_id="child-a",
            parent_chunk_id="parent-1",
            citation_text="Bộ luật số 45/2019/QH 14, Điều 36, khoản 1, điểm a",
            text="Người lao động thường xuyên không hoàn thành công việc...",
            dense_text="child a dense",
            sparse_text="child a sparse",
            payload={"level": "point"},
        )
        child_b = RetrievedRecord(
            chunk_id="child-b",
            parent_chunk_id="parent-1",
            citation_text="Bộ luật số 45/2019/QH 14, Điều 36, khoản 1, điểm b",
            text="Người lao động bị ốm đau kéo dài...",
            dense_text="child b dense",
            sparse_text="child b sparse",
            payload={"level": "point"},
        )

        retriever = HybridRetriever.__new__(HybridRetriever)
        records = {
            "parent-1": parent,
            "child-a": child_a,
            "child-b": child_b,
        }
        retriever._fetch_records = lambda chunk_ids: {
            chunk_id: records[chunk_id] for chunk_id in chunk_ids if chunk_id in records
        }

        hits = (
            SearchHit(
                chunk_id="child-a",
                qdrant_point_id="point-a",
                score=0.9,
                citation_text=child_a.citation_text,
                payload={"chunk_id": "child-a"},
            ),
            SearchHit(
                chunk_id="child-b",
                qdrant_point_id="point-b",
                score=0.8,
                citation_text=child_b.citation_text,
                payload={"chunk_id": "child-b"},
            ),
        )

        contexts = HybridRetriever._assemble_contexts(retriever, hits)

        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0].chunk_id, "parent-1")
        self.assertEqual(contexts[0].matched_chunk_ids, ("child-a", "child-b"))
        self.assertEqual(
            contexts[0].matched_citations,
            (child_a.citation_text, child_b.citation_text),
        )


if __name__ == "__main__":
    unittest.main()
