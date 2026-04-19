from __future__ import annotations

import unittest

from qdrant_client import models

from vn_labor_law_ai_assistant.retriever import (
    ARTICLE_REF_RE,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_RERANKER_TOP_N,
    HybridRetriever,
    RetrievedRecord,
    RetrievalContext,
    SearchHit,
    build_context_block,
    dedupe_preserve_order,
    format_context_for_prompt,
    parse_reference_values,
    prioritize_issue_filters,
    route_query,
    select_contexts_for_prompt,
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

    def test_route_query_maps_labor_code_keywords_to_current_document_id(self) -> None:
        intent = route_query("Tro cap thoi viec theo Dieu 46 Bo luat Lao dong 2019 duoc tinh the nao?")

        self.assertIn("45-2019-qh14", intent.document_filters)
        self.assertNotIn("du-lieu-cham-dut-hop-dong-lao-dong", intent.document_filters)
        self.assertEqual(intent.article_numbers, ("46",))

    def test_route_query_detects_termination_benefit_scenario_without_article_reference(self) -> None:
        intent = route_query(
            "Toi lam o cong ty 6 nam, da dong bao hiem that nghiep 4 nam, gio xin nghi dung quy dinh. Cong ty phai tra tro cap thoi viec cho toi nhu the nao?"
        )

        self.assertIn("cham_dut_hop_dong_lao_dong", intent.topic_filters)
        self.assertIn("tro_cap", intent.topic_filters)
        self.assertIn("tro_cap_thoi_viec", intent.issue_filters)
        self.assertIn("nghia_vu_khi_cham_dut", intent.issue_filters)

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

    def test_build_query_filter_ignores_generic_actor_filters(self) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._qdrant_models = models
        intent = route_query(
            "Toi lam o cong ty 6 nam, da dong bao hiem that nghiep 4 nam, gio xin nghi dung quy dinh. Cong ty phai tra tro cap thoi viec cho toi nhu the nao?"
        )

        query_filter = HybridRetriever._build_query_filter(retriever, intent)

        should_keys = [condition.key for condition in (query_filter.min_should.conditions if query_filter and query_filter.min_should else [])]

        self.assertNotIn("actor", should_keys)
        self.assertIn("topic", should_keys)
        self.assertIn("issue_type", should_keys)

    def test_prioritize_issue_filters_drops_generic_termination_obligation_when_specific_benefit_issue_exists(self) -> None:
        prioritized = prioritize_issue_filters(("tro_cap_thoi_viec", "nghia_vu_khi_cham_dut"))

        self.assertEqual(prioritized, ("tro_cap_thoi_viec",))


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

    def test_select_contexts_for_prompt_respects_char_budget(self) -> None:
        contexts = (
            RetrievalContext(
                chunk_id="chunk-1",
                citation_text="Dieu 46",
                text="A" * 120,
                payload={},
                score=1.0,
                matched_chunk_ids=("chunk-1",),
                matched_citations=("Dieu 46",),
            ),
            RetrievalContext(
                chunk_id="chunk-2",
                citation_text="Dieu 47",
                text="B" * 120,
                payload={},
                score=0.9,
                matched_chunk_ids=("chunk-2",),
                matched_citations=("Dieu 47",),
            ),
        )

        selected = select_contexts_for_prompt(contexts, max_contexts=6, max_chars=220)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].chunk_id, "chunk-1")

    def test_select_contexts_for_prompt_respects_token_budget(self) -> None:
        contexts = (
            RetrievalContext(
                chunk_id="chunk-1",
                citation_text="Dieu 46",
                text="mot hai ba bon nam sau bay tam chin muoi",
                payload={},
                score=1.0,
                matched_chunk_ids=("chunk-1",),
                matched_citations=("Dieu 46",),
            ),
            RetrievalContext(
                chunk_id="chunk-2",
                citation_text="Dieu 47",
                text="mot hai ba bon nam sau bay tam chin muoi",
                payload={},
                score=0.9,
                matched_chunk_ids=("chunk-2",),
                matched_citations=("Dieu 47",),
            ),
        )

        selected = select_contexts_for_prompt(contexts, max_contexts=6, max_chars=1000, max_tokens=20)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].chunk_id, "chunk-1")

    def test_format_context_for_prompt_preserves_first_oversized_block_without_ellipsis(self) -> None:
        context = RetrievalContext(
            chunk_id="chunk-1",
            citation_text="Dieu 46",
            text="A" * (DEFAULT_MAX_CONTEXT_CHARS + 500),
            payload={},
            score=1.0,
            matched_chunk_ids=("chunk-1",),
            matched_citations=("Dieu 46",),
        )

        prompt = format_context_for_prompt((context,), max_chars=200)

        self.assertGreater(len(prompt), 200)
        self.assertIn("[NGU CANH 1]", prompt)
        self.assertIn("Co so phap ly: Dieu 46", prompt)
        self.assertFalse(prompt.endswith("..."))

    def test_build_context_block_omits_redundant_match_section(self) -> None:
        context = RetrievalContext(
            chunk_id="chunk-1",
            citation_text="Dieu 46",
            text="Noi dung",
            payload={},
            score=1.0,
            matched_chunk_ids=("chunk-1",),
            matched_citations=("Dieu 46",),
        )

        block = build_context_block(context, 1)

        self.assertNotIn("Match goc:", block)

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

    def test_rerank_hits_demotes_delegation_clause_for_calculation_query(self) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)
        intent = route_query("Tro cap thoi viec duoc tinh nhu the nao theo Dieu 46?")
        clause_1 = RetrievedRecord(
            chunk_id="dieu-46-k1",
            parent_chunk_id=None,
            citation_text="Bo luat so 45/2019/QH14, Dieu 46, khoan 1",
            text="Moi nam lam viec duoc tro cap mot nua thang tien luong.",
            dense_text="",
            sparse_text="",
            payload={
                "article_number": "46",
                "clause_ref": "1",
                "topic": ["tro_cap"],
                "issue_type": ["tro_cap_thoi_viec"],
            },
        )
        clause_2 = RetrievedRecord(
            chunk_id="dieu-46-k2",
            parent_chunk_id=None,
            citation_text="Bo luat so 45/2019/QH14, Dieu 46, khoan 2",
            text="Thoi gian lam viec de tinh tro cap thoi viec la tong thoi gian lam viec thuc te.",
            dense_text="",
            sparse_text="",
            payload={
                "article_number": "46",
                "clause_ref": "2",
                "topic": ["tro_cap"],
                "issue_type": ["tro_cap_thoi_viec"],
            },
        )
        clause_4 = RetrievedRecord(
            chunk_id="dieu-46-k4",
            parent_chunk_id=None,
            citation_text="Bo luat so 45/2019/QH14, Dieu 46, khoan 4",
            text="Chinh phu quy dinh chi tiet Dieu nay.",
            dense_text="",
            sparse_text="",
            payload={
                "article_number": "46",
                "clause_ref": "4",
                "topic": ["tro_cap"],
                "issue_type": ["tro_cap_thoi_viec"],
            },
        )
        hits = (
            SearchHit("dieu-46-k4", "point-4", 1.1, clause_4.citation_text, {"chunk_id": "dieu-46-k4"}),
            SearchHit("dieu-46-k2", "point-2", 0.7, clause_2.citation_text, {"chunk_id": "dieu-46-k2"}),
            SearchHit("dieu-46-k1", "point-1", 0.5, clause_1.citation_text, {"chunk_id": "dieu-46-k1"}),
        )

        reranked = HybridRetriever._rerank_hits(
            retriever,
            hits,
            intent,
            {
                clause_1.chunk_id: clause_1,
                clause_2.chunk_id: clause_2,
                clause_4.chunk_id: clause_4,
            },
        )

        self.assertEqual(reranked[-1].chunk_id, "dieu-46-k4")
        self.assertEqual({reranked[0].chunk_id, reranked[1].chunk_id}, {"dieu-46-k1", "dieu-46-k2"})

    def test_rerank_hits_keeps_delegation_clause_when_query_asks_for_detail_guidance(self) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)
        intent = route_query("Chinh phu huong dan chi tiet Dieu 46 nhu the nao?")
        clause_2 = RetrievedRecord(
            chunk_id="dieu-46-k2",
            parent_chunk_id=None,
            citation_text="Bo luat so 45/2019/QH14, Dieu 46, khoan 2",
            text="Thoi gian lam viec de tinh tro cap thoi viec la tong thoi gian lam viec thuc te.",
            dense_text="",
            sparse_text="",
            payload={
                "article_number": "46",
                "clause_ref": "2",
                "topic": ["tro_cap"],
                "issue_type": ["tro_cap_thoi_viec"],
            },
        )
        clause_4 = RetrievedRecord(
            chunk_id="dieu-46-k4",
            parent_chunk_id=None,
            citation_text="Bo luat so 45/2019/QH14, Dieu 46, khoan 4",
            text="Chinh phu quy dinh chi tiet Dieu nay.",
            dense_text="",
            sparse_text="",
            payload={
                "article_number": "46",
                "clause_ref": "4",
                "topic": ["tro_cap"],
                "issue_type": ["tro_cap_thoi_viec"],
            },
        )
        hits = (
            SearchHit("dieu-46-k4", "point-4", 1.1, clause_4.citation_text, {"chunk_id": "dieu-46-k4"}),
            SearchHit("dieu-46-k2", "point-2", 0.7, clause_2.citation_text, {"chunk_id": "dieu-46-k2"}),
        )

        reranked = HybridRetriever._rerank_hits(
            retriever,
            hits,
            intent,
            {
                clause_2.chunk_id: clause_2,
                clause_4.chunk_id: clause_4,
            },
        )

        self.assertEqual(reranked[0].chunk_id, "dieu-46-k4")

    def test_rerank_hits_penalizes_maternity_heading_for_termination_benefit_query(self) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)
        intent = route_query(
            "Toi lam o cong ty 6 nam, da dong bao hiem that nghiep 4 nam, gio xin nghi dung quy dinh. Cong ty phai tra tro cap thoi viec cho toi nhu the nao?"
        )
        termination = RetrievedRecord(
            chunk_id="dieu-46-k2",
            parent_chunk_id=None,
            citation_text="Bo luat so 45/2019/QH14, Dieu 46, khoan 2",
            text="Thoi gian lam viec de tinh tro cap thoi viec la tong thoi gian lam viec thuc te tru di thoi gian da tham gia bao hiem that nghiep.",
            dense_text="",
            sparse_text="",
            payload={
                "article_number": "46",
                "article_title": "Tro cap thoi viec",
                "heading": "Dieu 46. Tro cap thoi viec",
                "section_heading": "Muc 3. Cham dut hop dong lao dong",
                "topic": ["cham_dut_hop_dong_lao_dong", "tro_cap"],
                "issue_type": ["tro_cap_thoi_viec"],
            },
        )
        maternity = RetrievedRecord(
            chunk_id="dieu-139-k4",
            parent_chunk_id=None,
            citation_text="Bo luat so 45/2019/QH14, Dieu 139, khoan 4",
            text="Lao dong nu co the tro lai lam viec truoc khi het thoi gian nghi thai san.",
            dense_text="",
            sparse_text="",
            payload={
                "article_number": "139",
                "article_title": "Nghi thai san",
                "heading": "Dieu 139. Nghi thai san",
                "section_heading": "Muc 2. Lao dong nu",
                "topic": ["bao_ve_thai_san"],
                "issue_type": ["bao_ve_thai_san"],
            },
        )
        hits = (
            SearchHit("dieu-139-k4", "point-139", 0.9, maternity.citation_text, {"chunk_id": "dieu-139-k4"}),
            SearchHit("dieu-46-k2", "point-46", 0.7, termination.citation_text, {"chunk_id": "dieu-46-k2"}),
        )

        reranked = HybridRetriever._rerank_hits(
            retriever,
            hits,
            intent,
            {
                termination.chunk_id: termination,
                maternity.chunk_id: maternity,
            },
        )

        self.assertEqual(reranked[0].chunk_id, "dieu-46-k2")

    def test_rerank_hits_demotes_delegation_clause_for_termination_benefit_fact_pattern(self) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)
        intent = route_query(
            "Toi lam o cong ty 6 nam, da dong bao hiem that nghiep 4 nam, gio xin nghi dung quy dinh. Cong ty phai tra tro cap thoi viec cho toi nhu the nao?"
        )
        clause_1 = RetrievedRecord(
            chunk_id="dieu-46-k1",
            parent_chunk_id=None,
            citation_text="Bo luat so 45/2019/QH14, Dieu 46, khoan 1",
            text="Moi nam lam viec duoc tro cap mot nua thang tien luong.",
            dense_text="",
            sparse_text="",
            payload={
                "article_number": "46",
                "article_title": "Tro cap thoi viec",
                "heading": "Dieu 46. Tro cap thoi viec",
                "section_heading": "Muc 3. Cham dut hop dong lao dong",
                "clause_ref": "1",
                "topic": ["cham_dut_hop_dong_lao_dong", "tro_cap"],
                "issue_type": ["tro_cap_thoi_viec"],
            },
        )
        clause_2 = RetrievedRecord(
            chunk_id="dieu-46-k2",
            parent_chunk_id=None,
            citation_text="Bo luat so 45/2019/QH14, Dieu 46, khoan 2",
            text="Thoi gian lam viec de tinh tro cap thoi viec la tong thoi gian lam viec tru di thoi gian da tham gia bao hiem that nghiep.",
            dense_text="",
            sparse_text="",
            payload={
                "article_number": "46",
                "article_title": "Tro cap thoi viec",
                "heading": "Dieu 46. Tro cap thoi viec",
                "section_heading": "Muc 3. Cham dut hop dong lao dong",
                "clause_ref": "2",
                "topic": ["cham_dut_hop_dong_lao_dong", "tro_cap"],
                "issue_type": ["tro_cap_thoi_viec"],
            },
        )
        clause_4 = RetrievedRecord(
            chunk_id="dieu-46-k4",
            parent_chunk_id=None,
            citation_text="Bo luat so 45/2019/QH14, Dieu 46, khoan 4",
            text="Chinh phu quy dinh chi tiet Dieu nay.",
            dense_text="",
            sparse_text="",
            payload={
                "article_number": "46",
                "article_title": "Tro cap thoi viec",
                "heading": "Dieu 46. Tro cap thoi viec",
                "section_heading": "Muc 3. Cham dut hop dong lao dong",
                "clause_ref": "4",
                "topic": ["cham_dut_hop_dong_lao_dong", "tro_cap"],
                "issue_type": ["tro_cap_thoi_viec"],
            },
        )
        hits = (
            SearchHit("dieu-46-k4", "point-4", 1.1, clause_4.citation_text, {"chunk_id": "dieu-46-k4"}),
            SearchHit("dieu-46-k2", "point-2", 0.7, clause_2.citation_text, {"chunk_id": "dieu-46-k2"}),
            SearchHit("dieu-46-k1", "point-1", 0.5, clause_1.citation_text, {"chunk_id": "dieu-46-k1"}),
        )

        reranked = HybridRetriever._rerank_hits(
            retriever,
            hits,
            intent,
            {
                clause_1.chunk_id: clause_1,
                clause_2.chunk_id: clause_2,
                clause_4.chunk_id: clause_4,
            },
        )

        self.assertEqual(reranked[-1].chunk_id, "dieu-46-k4")
        self.assertEqual({reranked[0].chunk_id, reranked[1].chunk_id}, {"dieu-46-k1", "dieu-46-k2"})

    def test_rerank_hits_prefers_primary_law_clause_over_decree_point_for_general_benefit_query(self) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)
        intent = route_query(
            "Toi lam o cong ty 6 nam, da dong bao hiem that nghiep 4 nam, gio xin nghi dung quy dinh. Cong ty phai tra tro cap thoi viec cho toi nhu the nao?"
        )
        law_clause = RetrievedRecord(
            chunk_id="dieu-46-k1",
            parent_chunk_id=None,
            citation_text="Bo luat so 45/2019/QH14, Dieu 46, khoan 1",
            text="Moi nam lam viec duoc tro cap mot nua thang tien luong.",
            dense_text="",
            sparse_text="",
            payload={
                "document_id": "45-2019-qh14",
                "article_number": "46",
                "article_title": "Tro cap thoi viec",
                "heading": "Dieu 46. Tro cap thoi viec",
                "section_heading": "Muc 3. Cham dut hop dong lao dong",
                "level": "clause",
                "clause_ref": "1",
                "topic": ["cham_dut_hop_dong_lao_dong", "tro_cap"],
                "issue_type": ["tro_cap_thoi_viec"],
            },
        )
        decree_point = RetrievedRecord(
            chunk_id="nd145-d8-b",
            parent_chunk_id=None,
            citation_text="Nghi dinh 145/2020/ND-CP, Dieu 8, khoan 4, diem b",
            text="Truong hop dac thu duoc tinh tro cap thoi viec.",
            dense_text="",
            sparse_text="",
            payload={
                "document_id": "nghi-dinh-145-2020-nd-cp",
                "article_number": "8",
                "article_title": "Tro cap thoi viec, tro cap mat viec lam",
                "heading": "Dieu 8. Tro cap thoi viec, tro cap mat viec lam",
                "section_heading": "Muc 2. Cham dut hop dong lao dong",
                "level": "point",
                "clause_ref": "4",
                "point_ref": "b",
                "topic": ["cham_dut_hop_dong_lao_dong", "tro_cap"],
                "issue_type": ["tro_cap_thoi_viec"],
            },
        )
        hits = (
            SearchHit("nd145-d8-b", "point-b", 1.0, decree_point.citation_text, {"chunk_id": "nd145-d8-b"}),
            SearchHit("dieu-46-k1", "point-46", 0.7, law_clause.citation_text, {"chunk_id": "dieu-46-k1"}),
        )

        reranked = HybridRetriever._rerank_hits(
            retriever,
            hits,
            intent,
            {
                law_clause.chunk_id: law_clause,
                decree_point.chunk_id: decree_point,
            },
        )

        self.assertEqual(reranked[0].chunk_id, "dieu-46-k1")

    def test_semantic_rerank_hits_promotes_cross_encoder_preferred_candidate(self) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._reranker_model_name = "BAAI/bge-reranker-v2-m3"
        retriever._reranker_top_n = DEFAULT_RERANKER_TOP_N

        hits = (
            SearchHit("chunk-a", "point-a", 0.9, "Dieu 41", {"chunk_id": "chunk-a"}),
            SearchHit("chunk-b", "point-b", 0.8, "Dieu 35", {"chunk_id": "chunk-b"}),
            SearchHit("chunk-c", "point-c", 0.7, "Dieu 46", {"chunk_id": "chunk-c"}),
        )

        retriever._predict_reranker_scores = lambda query, candidate_hits, direct_records: {
            "chunk-a": 0.1,
            "chunk-b": 0.95,
            "chunk-c": 0.2,
        }

        reranked = HybridRetriever._semantic_rerank_hits(retriever, "Cau hoi", hits, {})

        self.assertEqual(reranked[0].chunk_id, "chunk-b")
        self.assertEqual(reranked[1].chunk_id, "chunk-a")
        self.assertEqual(reranked[2].chunk_id, "chunk-c")
        self.assertGreater(reranked[0].score, reranked[1].score)

    def test_semantic_rerank_hits_keeps_order_when_disabled(self) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._reranker_model_name = ""
        retriever._reranker_top_n = DEFAULT_RERANKER_TOP_N

        hits = (
            SearchHit("chunk-a", "point-a", 0.9, "Dieu 41", {"chunk_id": "chunk-a"}),
            SearchHit("chunk-b", "point-b", 0.8, "Dieu 35", {"chunk_id": "chunk-b"}),
        )

        reranked = HybridRetriever._semantic_rerank_hits(retriever, "Cau hoi", hits, {})

        self.assertEqual(reranked, hits)


if __name__ == "__main__":
    unittest.main()
