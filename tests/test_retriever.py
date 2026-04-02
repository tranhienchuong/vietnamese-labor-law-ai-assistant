from __future__ import annotations

import unittest

from qdrant_client import models

from vn_labor_law_ai_assistant.retriever import (
    ARTICLE_REF_RE,
    HybridRetriever,
    RetrievedRecord,
    RetrievalContext,
    SearchHit,
    dedupe_preserve_order,
    format_context_for_prompt,
    parse_reference_values,
    route_query,
)


class QueryRoutingTests(unittest.TestCase):
    def test_route_query_extracts_filters_and_legal_refs(self) -> None:
        intent = route_query(
            "Toi bi cong ty duoi viec trai luat, muon doi boi thuong theo Dieu 41 thi lam the nao?"
        )

        self.assertIn("nguoi_lao_dong", intent.actor_filters)
        self.assertIn("nguoi_su_dung_lao_dong", intent.actor_filters)
        self.assertIn("cham_dut_hop_dong_lao_dong", intent.topic_filters)
        self.assertIn("trai_phap_luat", intent.issue_filters)
        self.assertIn("boi_thuong", intent.issue_filters)
        self.assertEqual(intent.article_number, "41")
        self.assertEqual(intent.article_numbers, ("41",))

    def test_parse_reference_values_collects_all_matches(self) -> None:
        values = parse_reference_values(
            ARTICLE_REF_RE,
            "so sanh dieu 46 va dieu 47, doi chieu dieu 46",
        )

        self.assertEqual(values, ("46", "47"))

    def test_route_query_keeps_multiple_article_refs(self) -> None:
        intent = route_query("So sanh tro cap thoi viec o Dieu 46 va Dieu 47.")

        self.assertEqual(intent.article_numbers, ("46", "47"))

    def test_build_query_filter_does_not_hard_require_legal_reference(self) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._qdrant_models = models
        intent = route_query("Toi tu y nghi viec 5 ngay co bi sa thai theo Dieu 35 khong?")

        query_filter = HybridRetriever._build_query_filter(retriever, intent)
        boost_filter = HybridRetriever._build_reference_boost_filter(retriever, intent)

        must_keys = [condition.key for condition in (query_filter.must or [])]
        boost_keys = [condition.key for condition in (boost_filter.must or [])]

        self.assertNotIn("article_number", must_keys)
        self.assertIn("article_number", boost_keys)
        article_condition = next(condition for condition in boost_filter.must if condition.key == "article_number")
        self.assertEqual(article_condition.match.any, ["35"])


class RetrievalAssemblyTests(unittest.TestCase):
    def test_dedupe_preserve_order_keeps_first_occurrence(self) -> None:
        self.assertEqual(
            dedupe_preserve_order(("a", "b", "a", "c", "b")),
            ("a", "b", "c"),
        )

    def test_format_context_for_prompt_includes_context_blocks(self) -> None:
        context = RetrievalContext(
            chunk_id="chunk-1",
            citation_text="Bo luat so 45/2019/QH 14, Dieu 46",
            text="Nguoi su dung lao dong co trach nhiem chi tra tro cap thoi viec...",
            payload={"level": "clause"},
            score=0.9,
            matched_chunk_ids=("chunk-1",),
            matched_citations=("Bo luat so 45/2019/QH 14, Dieu 46",),
        )

        prompt = format_context_for_prompt((context,))

        self.assertIn("[NGU CANH 1]", prompt)
        self.assertIn("Co so phap ly: Bo luat so 45/2019/QH 14, Dieu 46", prompt)
        self.assertIn("Nguoi su dung lao dong co trach nhiem", prompt)

    def test_assemble_contexts_deduplicates_shared_parent(self) -> None:
        parent = RetrievedRecord(
            chunk_id="parent-1",
            parent_chunk_id=None,
            citation_text="Bo luat so 45/2019/QH 14, Dieu 36, khoan 1",
            text="Nguoi su dung lao dong co quyen don phuong cham dut hop dong trong cac truong hop sau day...",
            dense_text="parent dense",
            sparse_text="parent sparse",
            payload={"level": "clause"},
        )
        child_a = RetrievedRecord(
            chunk_id="child-a",
            parent_chunk_id="parent-1",
            citation_text="Bo luat so 45/2019/QH 14, Dieu 36, khoan 1, diem a",
            text="Nguoi lao dong thuong xuyen khong hoan thanh cong viec...",
            dense_text="child a dense",
            sparse_text="child a sparse",
            payload={"level": "point"},
        )
        child_b = RetrievedRecord(
            chunk_id="child-b",
            parent_chunk_id="parent-1",
            citation_text="Bo luat so 45/2019/QH 14, Dieu 36, khoan 1, diem b",
            text="Nguoi lao dong bi om dau keo dai...",
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
