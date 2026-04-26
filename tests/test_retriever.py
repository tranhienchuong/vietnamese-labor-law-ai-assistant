from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qdrant_client import models

from vn_labor_law_ai_assistant.retriever import (
    ARTICLE_REF_RE,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_RERANKER_TOP_N,
    HybridRetriever,
    RECORD_SOURCE_QDRANT_PAYLOAD,
    RetrievedRecord,
    RetrievalContext,
    SearchHit,
    TERMINATION_ARTICLE_MAP,
    build_context_block,
    build_query_variants,
    dedupe_preserve_order,
    format_context_for_prompt,
    parse_reference_values,
    prioritize_issue_filters,
    record_from_qdrant_payload,
    route_query,
    select_contexts_for_prompt,
)


class QueryRoutingTests(unittest.TestCase):
    def test_termination_article_map_covers_required_articles(self) -> None:
        self.assertEqual(
            set(TERMINATION_ARTICLE_MAP),
            {
                "34",
                "35",
                "36",
                "37",
                "38",
                "39",
                "40",
                "41",
                "46",
                "47",
                "48",
                "122",
                "124",
                "125",
                "128",
                "129",
            },
        )

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

    def test_route_query_adds_query_expansion_for_retrieval_miss_phrases(self) -> None:
        cases = {
            "Cong ty tra luong cham thi co bi gi khong?": {"97", "35"},
            "Cong ty tra l\u01b0\u01a1ng ch\u1eadm thi co bi gi khong?": {"97", "35"},
            "Nghi phep nam chua dung thi thanh toan the nao?": {"113", "114", "37"},
            "Nghi ph\u00e9p n\u0103m chua dung thi thanh toan the nao?": {"113", "114", "37"},
            "Toi xin nghi viec rieng co huong luong khong?": {"115", "37"},
            "Cong ty giam so BHXH khong tra thi lam sao?": {"48", "17"},
            "Khong co giay uy quyen thi giam doc nhan su co duoc ky cham dut khong?": {"18", "45"},
            "Khong co gi\u1ea5y \u1ee7y quy\u1ec1n thi giam doc nhan su co duoc ky cham dut khong?": {"18", "45"},
            "Toi muon khoi kien cong ty thi thu tuc the nao?": {"188", "190"},
            "Toi muon kh\u1edfi ki\u1ec7n cong ty thi thu tuc the nao?": {"188", "190"},
        }

        for query, expected_articles in cases.items():
            with self.subTest(query=query):
                intent = route_query(query)
                self.assertTrue(expected_articles.issubset(set(intent.inferred_article_numbers)))
                self.assertTrue(intent.query_expansions)

        private_leave_intent = route_query("Toi xin nghi viec rieng co huong luong khong?")
        self.assertNotIn("35", private_leave_intent.inferred_article_numbers)

    def test_route_query_uses_rule_based_article_map_for_termination_questions(self) -> None:
        intent = route_query("Nhan vien tu y bo viec 5 ngay co bi sa thai khong?")

        self.assertIn("125", intent.inferred_article_numbers)
        self.assertIn("36", intent.inferred_article_numbers)
        self.assertIn("ky_luat_sa_thai", intent.topic_filters)
        self.assertIn("sa_thai", intent.issue_filters)

    def test_route_query_expands_common_plain_vietnamese_terms(self) -> None:
        cases = {
            "Cong ty ep viet don nghi thi toi doi boi thuong duoc khong?": {"34", "35", "36", "41"},
            "Luong thang 13 co bat buoc khong?": {"104"},
            "Cong ty can tru phep nam cua toi co dung khong?": {"113", "114", "48"},
            "Cong ty chuyen toi lam viec khac so voi hop dong duoc khong?": {"29"},
            "Khong dat KPI co bi cong ty don phuong cho nghi viec khong?": {"36"},
        }

        for query, expected_articles in cases.items():
            with self.subTest(query=query):
                intent = route_query(query)
                self.assertTrue(expected_articles.issubset(set(intent.inferred_article_numbers)))
                self.assertTrue(intent.query_expansions)

    def test_build_query_variants_includes_expanded_issue_and_citation_queries(self) -> None:
        intent = route_query("Cong ty no luong 1 thang toi nghi luon duoc khong?")

        variants = build_query_variants(intent)

        self.assertGreaterEqual(len(variants), 3)
        self.assertEqual(variants[0], "Cong ty no luong 1 thang toi nghi luon duoc khong?")
        self.assertTrue(any("khong duoc tra du luong" in variant for variant in variants))
        self.assertTrue(any("Dieu 97" in variant and "Dieu 35" in variant for variant in variants))

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

    def test_build_query_filter_uses_only_document_and_explicit_refs_as_hard_filters(self) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._qdrant_models = models
        intent = route_query("Toi tu y nghi viec 5 ngay co bi sa thai theo Dieu 35 khong?")

        query_filter = HybridRetriever._build_query_filter(retriever, intent)
        boost_filter = HybridRetriever._build_reference_boost_filter(retriever, intent)

        must_keys = [condition.key for condition in (query_filter.must or [])]
        boost_keys = [condition.key for condition in (boost_filter.must or [])]

        self.assertEqual(must_keys, ["article_number"])
        self.assertIn("article_number", boost_keys)
        hard_article_condition = next(
            condition for condition in query_filter.must if condition.key == "article_number"
        )
        self.assertEqual(hard_article_condition.match.any, ["35"])
        article_condition = next(condition for condition in boost_filter.must if condition.key == "article_number")
        self.assertTrue({"35", "36", "125"}.issubset(set(article_condition.match.any)))

    def test_build_query_filter_does_not_hard_filter_inferred_articles_or_taxonomy(self) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._qdrant_models = models
        intent = route_query(
            "Toi lam o cong ty 6 nam, da dong bao hiem that nghiep 4 nam, gio xin nghi dung quy dinh. Cong ty phai tra tro cap thoi viec cho toi nhu the nao?"
        )

        query_filter = HybridRetriever._build_query_filter(retriever, intent)

        self.assertIsNone(query_filter)
        self.assertIn("46", intent.inferred_article_numbers)

    def test_prioritize_issue_filters_drops_generic_termination_obligation_when_specific_benefit_issue_exists(self) -> None:
        prioritized = prioritize_issue_filters(("tro_cap_thoi_viec", "nghia_vu_khi_cham_dut"))

        self.assertEqual(prioritized, ("tro_cap_thoi_viec",))

    def test_sparse_query_uses_query_expansion_and_inferred_article_tokens(self) -> None:
        class FakeSegmenter:
            def segment(self, text: str) -> list[str]:
                return text.lower().split()

        class FakeSparseEncoder:
            def __init__(self) -> None:
                self.tokens: list[str] = []

            def encode_query(self, tokens: list[str]) -> SimpleNamespace:
                self.tokens = tokens
                return SimpleNamespace(indices=[1], values=[1.0])

        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._segmenter = FakeSegmenter()
        retriever._sparse_encoder = FakeSparseEncoder()
        retriever._qdrant_models = models
        intent = route_query("Cong ty giam so BHXH khong tra thi lam sao?")

        tokens, _ = HybridRetriever._encode_sparse_query(retriever, intent)

        self.assertIn("dieu_48", tokens)
        self.assertIn("dieu_17", tokens)
        self.assertIn("bao", tokens)
        self.assertIn("hiem", tokens)

    def test_dense_query_uses_custom_http_provider(self) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)

        with (
            patch.dict(
                "os.environ",
                {
                    "EMBEDDING_PROVIDER": "custom_http",
                    "EMBEDDING_API_URL": "https://embedding.example/v1/embeddings",
                },
                clear=False,
            ),
            patch(
                "vn_labor_law_ai_assistant.retriever.embed_query_via_http",
                return_value=[0.1, 0.2],
            ) as embed_http,
            patch.object(HybridRetriever, "_get_dense_model") as get_dense_model,
        ):
            vector = HybridRetriever._encode_dense_query(retriever, "cong ty cham dut trai luat")

        self.assertEqual(vector, [0.1, 0.2])
        embed_http.assert_called_once_with("cong ty cham dut trai luat")
        get_dense_model.assert_not_called()


class RetrievalAssemblyTests(unittest.TestCase):
    def test_dedupe_preserve_order_keeps_first_occurrence(self) -> None:
        self.assertEqual(
            dedupe_preserve_order(("a", "b", "a", "c", "b")),
            ("a", "b", "c"),
        )

    def test_record_from_qdrant_payload_builds_runtime_record(self) -> None:
        payload = {
            "chunk_id": "chunk-1",
            "parent_chunk_id": "parent-1",
            "citation_text": "Dieu 46",
            "text": "Noi dung chunk",
            "dense_text": "dense",
            "sparse_text": "sparse",
            "article_number": "46",
        }

        record = record_from_qdrant_payload(payload)

        self.assertEqual(record.chunk_id, "chunk-1")
        self.assertEqual(record.parent_chunk_id, "parent-1")
        self.assertEqual(record.citation_text, "Dieu 46")
        self.assertEqual(record.text, "Noi dung chunk")
        self.assertEqual(record.payload["article_number"], "46")

    def test_assemble_contexts_can_read_directly_from_qdrant_payload(self) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._record_source = RECORD_SOURCE_QDRANT_PAYLOAD

        hit = SearchHit(
            chunk_id="chunk-1",
            qdrant_point_id="point-1",
            score=0.9,
            citation_text="Dieu 46",
            payload={
                "chunk_id": "chunk-1",
                "qdrant_point_id": "point-1",
                "parent_chunk_id": None,
                "citation_text": "Dieu 46",
                "text": "Noi dung tu Qdrant payload",
                "dense_text": "dense",
                "sparse_text": "sparse",
            },
        )

        contexts = HybridRetriever._assemble_contexts(retriever, (hit,))

        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0].chunk_id, "chunk-1")
        self.assertEqual(contexts[0].text, "Noi dung tu Qdrant payload")

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

    def test_select_contexts_for_prompt_diversifies_articles_before_fill(self) -> None:
        contexts = (
            RetrievalContext(
                chunk_id="dieu-41-k1",
                citation_text="Dieu 41 khoan 1",
                text="Noi dung 41.1",
                payload={"document_id": "45-2019-qh14", "article_number": "41"},
                score=1.0,
                matched_chunk_ids=("dieu-41-k1",),
                matched_citations=("Dieu 41 khoan 1",),
            ),
            RetrievalContext(
                chunk_id="dieu-41-k2",
                citation_text="Dieu 41 khoan 2",
                text="Noi dung 41.2",
                payload={"document_id": "45-2019-qh14", "article_number": "41"},
                score=0.9,
                matched_chunk_ids=("dieu-41-k2",),
                matched_citations=("Dieu 41 khoan 2",),
            ),
            RetrievalContext(
                chunk_id="dieu-48-k1",
                citation_text="Dieu 48 khoan 1",
                text="Noi dung 48.1",
                payload={"document_id": "45-2019-qh14", "article_number": "48"},
                score=0.8,
                matched_chunk_ids=("dieu-48-k1",),
                matched_citations=("Dieu 48 khoan 1",),
            ),
        )

        selected = select_contexts_for_prompt(contexts, max_contexts=2, max_chars=1000)

        self.assertEqual([context.chunk_id for context in selected], ["dieu-41-k1", "dieu-48-k1"])

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

    def test_assemble_contexts_adds_article_siblings_for_remedy_queries(self) -> None:
        parent = RetrievedRecord(
            chunk_id="dieu-41-k1",
            parent_chunk_id=None,
            citation_text="Bo luat so 45/2019/QH14, Dieu 41, khoan 1",
            text="Nhan lai nguoi lao dong tro lai lam viec va boi thuong.",
            dense_text="",
            sparse_text="",
            payload={
                "document_id": "45-2019-qh14",
                "article_number": "41",
                "level": "clause",
            },
        )
        child = RetrievedRecord(
            chunk_id="dieu-41-k1-a",
            parent_chunk_id="dieu-41-k1",
            citation_text="Bo luat so 45/2019/QH14, Dieu 41, khoan 1, diem a",
            text="Noi dung diem a.",
            dense_text="",
            sparse_text="",
            payload={
                "document_id": "45-2019-qh14",
                "article_number": "41",
                "level": "point",
            },
        )
        sibling = RetrievedRecord(
            chunk_id="dieu-41-k2",
            parent_chunk_id=None,
            citation_text="Bo luat so 45/2019/QH14, Dieu 41, khoan 2",
            text="Nguoi lao dong khong muon tiep tuc lam viec.",
            dense_text="",
            sparse_text="",
            payload={
                "document_id": "45-2019-qh14",
                "article_number": "41",
                "level": "clause",
            },
        )
        retriever = HybridRetriever.__new__(HybridRetriever)
        records = {parent.chunk_id: parent, child.chunk_id: child}
        retriever._fetch_records = lambda chunk_ids: {
            chunk_id: records[chunk_id] for chunk_id in chunk_ids if chunk_id in records
        }
        retriever._fetch_article_siblings = lambda **kwargs: (sibling,)
        intent = route_query("Cong ty cham dut trai luat phai boi thuong the nao?")

        contexts = HybridRetriever._assemble_contexts(
            retriever,
            (
                SearchHit(
                    chunk_id=child.chunk_id,
                    qdrant_point_id="point-a",
                    score=0.9,
                    citation_text=child.citation_text,
                    payload={"chunk_id": child.chunk_id},
                ),
            ),
            intent=intent,
        )

        self.assertEqual([context.chunk_id for context in contexts], ["dieu-41-k1", "dieu-41-k2"])

    def test_reference_fallback_fetches_explicit_clause_even_when_article_is_present(self) -> None:
        existing = RetrievedRecord(
            chunk_id="dieu-35-k1",
            parent_chunk_id=None,
            citation_text="Bo luat so 45/2019/QH14, Dieu 35, khoan 1",
            text="Khoan 1",
            dense_text="",
            sparse_text="",
            payload={"article_number": "35", "clause_ref": "1"},
        )
        fallback = RetrievedRecord(
            chunk_id="dieu-35-k2",
            parent_chunk_id=None,
            citation_text="Bo luat so 45/2019/QH14, Dieu 35, khoan 2",
            text="Khoan 2",
            dense_text="",
            sparse_text="",
            payload={"article_number": "35", "clause_ref": "2"},
        )
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._fetch_records = lambda chunk_ids: {existing.chunk_id: existing}
        retriever._fetch_records_by_reference = lambda **kwargs: (fallback,)
        intent = route_query("Dieu 35 khoan 2 quy dinh gi?")
        hits = (
            SearchHit(
                chunk_id=existing.chunk_id,
                qdrant_point_id="point-1",
                score=0.9,
                citation_text=existing.citation_text,
                payload={"chunk_id": existing.chunk_id},
            ),
        )

        expanded_hits = HybridRetriever._append_reference_fallback_hits(
            retriever,
            hits,
            intent,
            limit=4,
        )

        self.assertEqual([hit.chunk_id for hit in expanded_hits], ["dieu-35-k1", "dieu-35-k2"])

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
