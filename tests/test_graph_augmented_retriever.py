from __future__ import annotations

import unittest

from vn_labor_law_ai_assistant.heuristic_router import route_query
from vn_labor_law_ai_assistant.rag.retrieval.retriever import HybridRetriever
from vn_labor_law_ai_assistant.rag.retrieval.models import (
    RetrievedRecord,
    RetrievalContext,
    RetrievalResult,
    SearchHit,
)
from vn_labor_law_ai_assistant.rag.retrieval.scoring import RelevanceScorer


class FakeQdrantSearcher:
    def search(self, **kwargs):
        return (
            SearchHit(
                chunk_id="seed",
                qdrant_point_id="point-seed",
                score=0.9,
                citation_text="Dieu 35",
                payload={
                    "chunk_id": "seed",
                    "text": "seed text",
                    "dense_text": "",
                    "sparse_text": "",
                    "citation_text": "Dieu 35",
                },
            ),
        )


class FakeReferenceExpander:
    def append_forced_reference_hits(self, hits, intent, *, limit):
        return hits

    def append_reference_fallback_hits(self, hits, intent, *, limit):
        return hits

    def pin_forced_reference_hits(self, hits, intent):
        return hits


class FakeRecordStore:
    def __init__(self) -> None:
        self.records = {
            "seed": RetrievedRecord(
                chunk_id="seed",
                parent_chunk_id=None,
                citation_text="Dieu 35",
                text="seed text",
                dense_text="",
                sparse_text="",
                payload={"chunk_id": "seed", "article_number": "35"},
            ),
            "expanded": RetrievedRecord(
                chunk_id="expanded",
                parent_chunk_id=None,
                citation_text="Dieu 46",
                text="expanded text",
                dense_text="",
                sparse_text="",
                payload={"chunk_id": "expanded", "article_number": "46"},
            ),
        }
        self.fetch_records_calls: list[tuple[str, ...]] = []

    def fetch_records_from_hits(self, hits):
        return {
            hit.chunk_id: self.records[hit.chunk_id]
            for hit in hits
            if hit.chunk_id in self.records
        }

    def fetch_records(self, chunk_ids):
        self.fetch_records_calls.append(tuple(chunk_ids))
        return {
            chunk_id: self.records[chunk_id]
            for chunk_id in chunk_ids
            if chunk_id in self.records
        }


class FakeScorer:
    def rerank_hits(self, hits, intent, direct_records):
        return tuple(hits)


class FakeSemanticReranker:
    def semantic_rerank_hits(self, query, hits, direct_records):
        return tuple(hits)


class FakeContextAssembler:
    def assemble_contexts(self, hits, intent=None):
        return tuple(
            RetrievalContext(
                chunk_id=hit.chunk_id,
                citation_text=hit.citation_text,
                text=hit.chunk_id,
                payload=hit.payload,
                score=hit.score,
                matched_chunk_ids=(hit.chunk_id,),
                matched_citations=(hit.citation_text,),
            )
            for hit in hits
        )


class FakeGraphExpander:
    def expand_from_hits(self, *, hits, direct_records, intent):
        return (
            SearchHit(
                chunk_id="expanded",
                qdrant_point_id="point-expanded",
                score=0.7,
                citation_text="",
                payload={
                    "chunk_id": "expanded",
                    "retrieval_source": "graph",
                    "retrieval_method": "neo4j_graph_expansion",
                    "graph_seed_chunk_ids": ["seed"],
                    "seed_chunk_ids": ["seed"],
                    "graph_edge_path": ["REFERENCES"],
                    "graph_edge_types": ["REFERENCES"],
                    "graph_node_path": ["chunk:seed", "chunk:expanded"],
                    "graph_path": ["chunk:seed", "chunk:expanded"],
                    "graph_paths": [["chunk:seed", "chunk:expanded"]],
                    "graph_depth": 2,
                    "graph_score": 0.7,
                },
            ),
        )


def make_retriever(*, graph_enabled: bool) -> HybridRetriever:
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever._route_query = lambda query: route_query(query)
    retriever._qdrant_searcher = FakeQdrantSearcher()
    retriever._reference_expander = FakeReferenceExpander()
    retriever._record_store = FakeRecordStore()
    retriever._scorer = FakeScorer()
    retriever._semantic_reranker = FakeSemanticReranker()
    retriever._context_assembler = FakeContextAssembler()
    retriever._legal_graph_expander = FakeGraphExpander() if graph_enabled else None
    return retriever


class GraphAugmentedRetrieverTests(unittest.TestCase):
    def test_graph_disabled_retriever_output_unchanged(self) -> None:
        retriever = make_retriever(graph_enabled=False)

        result = HybridRetriever.retrieve(retriever, "Dieu 35", top_k=8)

        self.assertIsInstance(result, RetrievalResult)
        self.assertEqual([hit.chunk_id for hit in result.hits], ["seed"])

    def test_graph_enabled_adds_expanded_hit_with_provenance(self) -> None:
        retriever = make_retriever(graph_enabled=True)

        result = HybridRetriever.retrieve(retriever, "So sanh Dieu 35 va Dieu 46", top_k=8)

        self.assertEqual([hit.chunk_id for hit in result.hits], ["seed", "expanded"])
        graph_hit = result.hits[1]
        self.assertEqual(graph_hit.payload["retrieval_source"], "graph")
        self.assertEqual(graph_hit.payload["retrieval_method"], "neo4j_graph_expansion")
        self.assertEqual(graph_hit.payload["graph_seed_chunk_ids"], ["seed"])
        self.assertEqual(graph_hit.payload["graph_edge_types"], ["REFERENCES"])
        self.assertIn(("expanded",), retriever._record_store.fetch_records_calls)

    def test_context_preserves_graph_debug_and_citation_metadata(self) -> None:
        retriever = make_retriever(graph_enabled=True)

        result = HybridRetriever.retrieve(retriever, "So sanh Dieu 35 va Dieu 46", top_k=8)
        graph_context = next(context for context in result.contexts if context.chunk_id == "expanded")

        self.assertEqual(graph_context.citation_text, "Dieu 46")
        self.assertEqual(graph_context.payload["citation_text"], "Dieu 46")
        self.assertEqual(graph_context.payload["retrieval_source"], "graph")
        self.assertEqual(graph_context.payload["graph_edge_types"], ["REFERENCES"])

    def test_merge_deduplicates_chunk_id_and_marks_hybrid_source(self) -> None:
        retriever = make_retriever(graph_enabled=False)
        vector_hit = SearchHit(
            chunk_id="seed",
            qdrant_point_id="point-seed",
            score=0.8,
            citation_text="Dieu 35",
            payload={"chunk_id": "seed", "retrieval_source": "vector", "vector_score": 0.8},
        )
        graph_hit = SearchHit(
            chunk_id="seed",
            qdrant_point_id="point-seed",
            score=0.7,
            citation_text="Dieu 35",
            payload={
                "chunk_id": "seed",
                "retrieval_source": "graph",
                "graph_score": 0.7,
                "graph_edge_types": ["REFERENCES"],
                "graph_path": ["chunk:seed"],
                "seed_chunk_ids": ["other"],
            },
        )

        merged = retriever._merge_vector_and_graph_hits((vector_hit,), (graph_hit,))

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].chunk_id, "seed")
        self.assertEqual(merged[0].payload["retrieval_source"], "hybrid")
        self.assertEqual(merged[0].payload["vector_score"], 0.8)
        self.assertEqual(merged[0].payload["graph_score"], 0.7)

    def test_normative_rank_ordering_prefers_higher_rank_on_close_scores(self) -> None:
        scorer = RelevanceScorer()
        intent = route_query("quy dinh ve hop dong lao dong")
        law_hit = SearchHit("law", "law", 0.5, "BLLD", {"chunk_id": "law"})
        decree_hit = SearchHit("decree", "decree", 0.5, "ND145", {"chunk_id": "decree"})
        records = {
            "law": RetrievedRecord(
                "law",
                None,
                "BLLD",
                "text",
                "",
                "",
                {"chunk_id": "law", "normative_rank": 1, "document_id": "45-2019-qh14"},
            ),
            "decree": RetrievedRecord(
                "decree",
                None,
                "ND145",
                "text",
                "",
                "",
                {"chunk_id": "decree", "normative_rank": 2, "document_id": "nghi-dinh-145-2020-nd-cp"},
            ),
        }

        ranked = scorer.rerank_hits((decree_hit, law_hit), intent, records)

        self.assertEqual([hit.chunk_id for hit in ranked], ["law", "decree"])

    def test_contract_content_ranking_demotes_unrelated_deposit_article(self) -> None:
        scorer = RelevanceScorer()
        intent = route_query("H\u1ee3p \u0111\u1ed3ng lao \u0111\u1ed9ng c\u1ea7n c\u00f3 nh\u1eefng n\u1ed9i dung g\u00ec?")
        hits = (
            SearchHit("nd145-17", "nd145-17", 2.0, "ND145 Dieu 17", {"applied_query_intent": ["labor_contract_content"]}),
            SearchHit("tt10-3", "tt10-3", 1.0, "TT10 Dieu 3", {"applied_query_intent": ["labor_contract_content"]}),
            SearchHit("bll-21", "bll-21", 1.0, "BLLD Dieu 21", {"applied_query_intent": ["labor_contract_content"]}),
        )
        records = {
            "nd145-17": RetrievedRecord("nd145-17", None, "ND145 Dieu 17", "ky quy", "", "", {"document_id": "nghi-dinh-145-2020-nd-cp", "article_number": "17"}),
            "tt10-3": RetrievedRecord("tt10-3", None, "TT10 Dieu 3", "noi dung hop dong", "", "", {"document_id": "thong-tu-10-2020-tt-bldtbxh", "article_number": "3"}),
            "bll-21": RetrievedRecord("bll-21", None, "BLLD Dieu 21", "noi dung hop dong", "", "", {"document_id": "45-2019-qh14", "article_number": "21", "clause_ref": "1"}),
        }

        ranked = scorer.rerank_hits(hits, intent, records)

        self.assertEqual(ranked[0].chunk_id, "bll-21")
        self.assertEqual(ranked[-1].chunk_id, "nd145-17")
        self.assertLess([hit.chunk_id for hit in ranked].index("tt10-3"), [hit.chunk_id for hit in ranked].index("nd145-17"))

    def test_illegal_employee_termination_ranking_prefers_article_40(self) -> None:
        scorer = RelevanceScorer()
        intent = route_query(
            "Ng\u01b0\u1eddi lao \u0111\u1ed9ng \u0111\u01a1n ph\u01b0\u01a1ng ch\u1ea5m d\u1ee9t h\u1ee3p \u0111\u1ed3ng tr\u00e1i lu\u1eadt th\u00ec ph\u1ea3i b\u1ed3i th\u01b0\u1eddng g\u00ec?"
        )
        hits = (
            SearchHit("article-46", "article-46", 1.5, "BLLD Dieu 46", {"applied_query_intent": ["illegal_unilateral_termination_by_employee"]}),
            SearchHit("article-40", "article-40", 0.8, "BLLD Dieu 40", {"applied_query_intent": ["illegal_unilateral_termination_by_employee"]}),
            SearchHit("article-39", "article-39", 0.7, "BLLD Dieu 39", {"applied_query_intent": ["illegal_unilateral_termination_by_employee"]}),
        )
        records = {
            "article-46": RetrievedRecord("article-46", None, "BLLD Dieu 46", "tro cap thoi viec", "", "", {"document_id": "45-2019-qh14", "article_number": "46"}),
            "article-40": RetrievedRecord("article-40", None, "BLLD Dieu 40", "boi thuong nua thang tien luong", "", "", {"document_id": "45-2019-qh14", "article_number": "40"}),
            "article-39": RetrievedRecord("article-39", None, "BLLD Dieu 39", "trai phap luat", "", "", {"document_id": "45-2019-qh14", "article_number": "39"}),
        }

        ranked = scorer.rerank_hits(hits, intent, records)

        self.assertEqual([hit.chunk_id for hit in ranked[:2]], ["article-40", "article-39"])

    def test_structural_change_ranking_demotes_article_40(self) -> None:
        scorer = RelevanceScorer()
        intent = route_query("C\u00f4ng ty thay \u0111\u1ed5i c\u01a1 c\u1ea5u th\u00ec ph\u1ea3i tr\u1ea3 tr\u1ee3 c\u1ea5p g\u00ec?")
        hits = (
            SearchHit("article-40", "article-40", 1.5, "BLLD Dieu 40", {"applied_query_intent": ["structural_change_job_loss_allowance"]}),
            SearchHit("article-47", "article-47", 0.8, "BLLD Dieu 47", {"applied_query_intent": ["structural_change_job_loss_allowance"]}),
            SearchHit("nd145-8", "nd145-8", 0.7, "ND145 Dieu 8", {"applied_query_intent": ["structural_change_job_loss_allowance"]}),
        )
        records = {
            "article-40": RetrievedRecord("article-40", None, "BLLD Dieu 40", "boi thuong", "", "", {"document_id": "45-2019-qh14", "article_number": "40"}),
            "article-47": RetrievedRecord("article-47", None, "BLLD Dieu 47", "tro cap mat viec", "", "", {"document_id": "45-2019-qh14", "article_number": "47"}),
            "nd145-8": RetrievedRecord("nd145-8", None, "ND145 Dieu 8", "tro cap mat viec", "", "", {"document_id": "nghi-dinh-145-2020-nd-cp", "article_number": "8"}),
        }

        ranked = scorer.rerank_hits(hits, intent, records)

        self.assertEqual(ranked[-1].chunk_id, "article-40")

    def test_legal_definition_ranking_prefers_exact_definition_clause(self) -> None:
        scorer = RelevanceScorer()
        intent = route_query("Nguoi lao dong duoc dinh nghia nhu the nao theo Bo luat Lao dong 2019?")
        hits = (
            SearchHit("article-3-intro", "article-3-intro", 1.5, "BLLD Dieu 3", {"applied_query_intent": ["legal_definition_lookup"]}),
            SearchHit("article-8", "article-8", 1.8, "BLLD Dieu 8", {"applied_query_intent": ["legal_definition_lookup"]}),
            SearchHit("article-3-k1", "article-3-k1", 0.8, "BLLD Dieu 3 khoan 1", {"applied_query_intent": ["legal_definition_lookup"]}),
        )
        records = {
            "article-3-intro": RetrievedRecord("article-3-intro", None, "BLLD Dieu 3", "giai thich tu ngu", "", "", {"document_id": "45-2019-qh14", "article_number": "3"}),
            "article-8": RetrievedRecord("article-8", None, "BLLD Dieu 8", "hanh vi bi cam", "", "", {"document_id": "45-2019-qh14", "article_number": "8"}),
            "article-3-k1": RetrievedRecord("article-3-k1", None, "BLLD Dieu 3 khoan 1", "nguoi lao dong la nguoi lam viec theo thoa thuan", "", "", {"document_id": "45-2019-qh14", "article_number": "3", "clause_ref": "1"}),
        }

        ranked = scorer.rerank_hits(hits, intent, records)

        self.assertEqual(ranked[0].chunk_id, "article-3-k1")

    def test_comparison_ranking_keeps_structural_change_side_in_top_results(self) -> None:
        scorer = RelevanceScorer()
        intent = route_query(
            "So sanh trach nhiem khi nguoi lao dong don phuong cham dut hop dong trai luat voi truong hop cong ty thay doi co cau phai tro cap?"
        )
        applied = [
            "illegal_unilateral_termination_by_employee",
            "structural_change_job_loss_allowance",
            "compare_employee_unlawful_termination_vs_structural_change",
        ]
        hits = (
            SearchHit("article-40", "article-40", 1.5, "BLLD Dieu 40", {"applied_query_intent": applied}),
            SearchHit("article-47", "article-47", 1.2, "BLLD Dieu 47", {"applied_query_intent": applied}),
            SearchHit("article-42", "article-42", 0.6, "BLLD Dieu 42", {"applied_query_intent": applied}),
            SearchHit("article-46", "article-46", 1.4, "BLLD Dieu 46", {"applied_query_intent": applied}),
        )
        records = {
            "article-40": RetrievedRecord("article-40", None, "BLLD Dieu 40", "boi thuong khi nguoi lao dong cham dut trai phap luat", "", "", {"document_id": "45-2019-qh14", "article_number": "40"}),
            "article-47": RetrievedRecord("article-47", None, "BLLD Dieu 47", "tro cap mat viec lam", "", "", {"document_id": "45-2019-qh14", "article_number": "47"}),
            "article-42": RetrievedRecord("article-42", None, "BLLD Dieu 42", "thay doi co cau cong nghe hoac vi ly do kinh te", "", "", {"document_id": "45-2019-qh14", "article_number": "42"}),
            "article-46": RetrievedRecord("article-46", None, "BLLD Dieu 46", "tro cap thoi viec", "", "", {"document_id": "45-2019-qh14", "article_number": "46"}),
        }

        ranked = scorer.rerank_hits(hits, intent, records)
        ranked_ids = [hit.chunk_id for hit in ranked]

        self.assertLess(ranked_ids.index("article-42"), ranked_ids.index("article-46"))
        self.assertTrue({"article-40", "article-42", "article-47"}.issubset(set(ranked_ids[:3])))

    def test_no_notice_ranking_prefers_article_35_clause_2(self) -> None:
        scorer = RelevanceScorer()
        intent = route_query(
            "Khi n\u00e0o ng\u01b0\u1eddi lao \u0111\u1ed9ng \u0111\u01b0\u1ee3c ngh\u1ec9 vi\u1ec7c kh\u00f4ng c\u1ea7n b\u00e1o tr\u01b0\u1edbc?"
        )
        hits = (
            SearchHit("article-40", "article-40", 1.5, "BLLD Dieu 40", {"applied_query_intent": ["no_notice_resignation"]}),
            SearchHit("article-35-k1", "article-35-k1", 1.0, "BLLD Dieu 35 khoan 1", {"applied_query_intent": ["no_notice_resignation"]}),
            SearchHit("article-35-k2", "article-35-k2", 0.8, "BLLD Dieu 35 khoan 2", {"applied_query_intent": ["no_notice_resignation"]}),
        )
        records = {
            "article-40": RetrievedRecord("article-40", None, "BLLD Dieu 40", "boi thuong", "", "", {"document_id": "45-2019-qh14", "article_number": "40"}),
            "article-35-k1": RetrievedRecord("article-35-k1", None, "BLLD Dieu 35 khoan 1", "bao truoc", "", "", {"document_id": "45-2019-qh14", "article_number": "35", "clause_ref": "1"}),
            "article-35-k2": RetrievedRecord("article-35-k2", None, "BLLD Dieu 35 khoan 2", "khong can bao truoc", "", "", {"document_id": "45-2019-qh14", "article_number": "35", "clause_ref": "2"}),
        }

        ranked = scorer.rerank_hits(hits, intent, records)

        self.assertEqual(ranked[0].chunk_id, "article-35-k2")
        self.assertGreater([hit.chunk_id for hit in ranked].index("article-40"), [hit.chunk_id for hit in ranked].index("article-35-k2"))

    def test_overtime_conditions_ranking_prefers_article_107_over_article_98(self) -> None:
        scorer = RelevanceScorer()
        intent = route_query(
            "Tr\u01b0\u1eddng h\u1ee3p n\u00e0o \u0111\u01b0\u1ee3c l\u00e0m th\u00eam gi\u1edd v\u00e0 gi\u1edbi h\u1ea1n l\u00e0m th\u00eam theo th\u00e1ng l\u00e0 bao nhi\u00eau?"
        )
        hits = (
            SearchHit("article-98", "article-98", 1.5, "BLLD Dieu 98", {"applied_query_intent": ["overtime_conditions_and_limits"]}),
            SearchHit("article-107-k2", "article-107-k2", 0.8, "BLLD Dieu 107 khoan 2", {"applied_query_intent": ["overtime_conditions_and_limits"]}),
            SearchHit("nd145-60", "nd145-60", 0.7, "ND145 Dieu 60", {"applied_query_intent": ["overtime_conditions_and_limits"]}),
        )
        records = {
            "article-98": RetrievedRecord("article-98", None, "BLLD Dieu 98", "tien luong lam them gio", "", "", {"document_id": "45-2019-qh14", "article_number": "98", "clause_ref": "1"}),
            "article-107-k2": RetrievedRecord("article-107-k2", None, "BLLD Dieu 107 khoan 2", "khong qua 40 gio trong 01 thang", "", "", {"document_id": "45-2019-qh14", "article_number": "107", "clause_ref": "2"}),
            "nd145-60": RetrievedRecord("nd145-60", None, "ND145 Dieu 60", "gioi han so gio lam them", "", "", {"document_id": "nghi-dinh-145-2020-nd-cp", "article_number": "60"}),
        }

        ranked = scorer.rerank_hits(hits, intent, records)

        self.assertEqual(ranked[0].chunk_id, "article-107-k2")
        self.assertLess([hit.chunk_id for hit in ranked].index("article-107-k2"), [hit.chunk_id for hit in ranked].index("article-98"))

    def test_overtime_pay_ranking_prefers_article_98_over_article_107(self) -> None:
        scorer = RelevanceScorer()
        intent = route_query(
            "L\u01b0\u01a1ng l\u00e0m th\u00eam gi\u1edd v\u00e0o ban \u0111\u00eam \u0111\u01b0\u1ee3c tr\u1ea3 nh\u01b0 th\u1ebf n\u00e0o?"
        )
        hits = (
            SearchHit("article-107", "article-107", 1.5, "BLLD Dieu 107", {"applied_query_intent": ["overtime_pay"]}),
            SearchHit("article-98", "article-98", 0.8, "BLLD Dieu 98", {"applied_query_intent": ["overtime_pay"]}),
            SearchHit("nd145-55", "nd145-55", 0.7, "ND145 Dieu 55", {"applied_query_intent": ["overtime_pay"]}),
        )
        records = {
            "article-107": RetrievedRecord("article-107", None, "BLLD Dieu 107", "gioi han lam them gio", "", "", {"document_id": "45-2019-qh14", "article_number": "107", "clause_ref": "2"}),
            "article-98": RetrievedRecord("article-98", None, "BLLD Dieu 98", "tien luong lam them gio lam viec ban dem", "", "", {"document_id": "45-2019-qh14", "article_number": "98", "clause_ref": "1"}),
            "nd145-55": RetrievedRecord("nd145-55", None, "ND145 Dieu 55", "tien luong lam them gio", "", "", {"document_id": "nghi-dinh-145-2020-nd-cp", "article_number": "55"}),
        }

        ranked = scorer.rerank_hits(hits, intent, records)

        self.assertEqual(ranked[0].chunk_id, "article-98")
        self.assertLess([hit.chunk_id for hit in ranked].index("article-98"), [hit.chunk_id for hit in ranked].index("article-107"))


if __name__ == "__main__":
    unittest.main()
