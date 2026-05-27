from __future__ import annotations

import unittest

from vn_labor_law_ai_assistant.heuristic_router import route_query
from vn_labor_law_ai_assistant.rag.graph import GraphExpansionResult, LegalGraphConfig
from vn_labor_law_ai_assistant.rag.graph.expander import (
    Neo4jLegalGraphExpander,
    classify_graph_query_intent,
    dedupe_search_hits,
    graph_priority_references_for_intent,
)
from vn_labor_law_ai_assistant.rag.retrieval.models import RetrievedRecord, SearchHit


class FakeGraphStore:
    def __init__(self, result: GraphExpansionResult) -> None:
        self.result = result
        self.calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def expand_from_chunk_ids(self, chunk_ids, **kwargs):
        self.calls.append((tuple(chunk_ids), kwargs))
        return self.result


def make_hit(chunk_id: str, score: float = 0.9) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        qdrant_point_id=f"point-{chunk_id}",
        score=score,
        citation_text=f"Dieu {chunk_id}",
        payload={"chunk_id": chunk_id, "text": "text", "dense_text": "", "sparse_text": ""},
    )


def make_record(chunk_id: str) -> RetrievedRecord:
    return RetrievedRecord(
        chunk_id=chunk_id,
        parent_chunk_id=None,
        citation_text=f"Dieu {chunk_id}",
        text="noi dung",
        dense_text="",
        sparse_text="",
        payload={"chunk_id": chunk_id},
    )


class LegalGraphExpanderTests(unittest.TestCase):
    def test_natural_language_query_triggers_graph_without_explicit_article(self) -> None:
        store = FakeGraphStore(GraphExpansionResult(seed_chunk_ids=("seed",), expanded_chunk_ids=()))
        expander = Neo4jLegalGraphExpander(
            store=store,
            config=LegalGraphConfig(enabled=True, complex_query_only=True),
        )

        graph_hits = expander.expand_from_hits(
            hits=(make_hit("seed"),),
            direct_records={"seed": make_record("seed")},
            intent=route_query("Cong ty phai lam gi khi cho nghi viec trai luat?"),
        )

        self.assertEqual(graph_hits, ())
        self.assertEqual(store.calls[0][0], ("seed",))

    def test_expand_from_seed_chunk_adds_provenance(self) -> None:
        result = GraphExpansionResult(
            seed_chunk_ids=("seed",),
            expanded_chunk_ids=("expanded",),
            paths=(
                {
                    "chunk_id": "expanded",
                    "graph_depth": 2,
                    "graph_edge_path": ["SOURCE_OF", "REFERENCES", "HAS_SOURCE_CHUNK"],
                    "graph_node_path": ["chunk:seed", "clause:35:1", "article:46", "chunk:expanded"],
                    "graph_confidence": 0.9,
                },
            ),
        )
        expander = Neo4jLegalGraphExpander(
            store=FakeGraphStore(result),
            config=LegalGraphConfig(enabled=True, complex_query_only=False),
        )

        graph_hits = expander.expand_from_hits(
            hits=(make_hit("seed"),),
            direct_records={"seed": make_record("seed")},
            intent=route_query("So sanh Dieu 35 va Dieu 46"),
        )

        self.assertEqual(len(graph_hits), 1)
        self.assertEqual(graph_hits[0].chunk_id, "expanded")
        self.assertLess(graph_hits[0].score, 0.9)
        self.assertEqual(graph_hits[0].payload["retrieval_source"], "graph")
        self.assertEqual(graph_hits[0].payload["retrieval_method"], "neo4j_graph_expansion")
        self.assertEqual(graph_hits[0].payload["graph_seed_chunk_ids"], ["seed"])
        self.assertEqual(graph_hits[0].payload["graph_edge_types"], ["SOURCE_OF", "REFERENCES", "HAS_SOURCE_CHUNK"])
        self.assertEqual(graph_hits[0].payload["graph_depth"], 2)

    def test_multi_hop_query_uses_depth_greater_than_two(self) -> None:
        store = FakeGraphStore(
            GraphExpansionResult(
                seed_chunk_ids=("seed",),
                expanded_chunk_ids=("expanded",),
                paths=(
                    {
                        "chunk_id": "expanded",
                        "graph_depth": 3,
                        "graph_edge_path": [
                            "SOURCE_OF",
                            "MENTIONS_CONCEPT",
                            "MENTIONS_CONCEPT",
                        ],
                        "graph_node_path": [
                            "chunk:seed",
                            "clause:35:1",
                            "concept:don_phuong",
                            "chunk:expanded",
                        ],
                        "graph_confidence": 0.7,
                    },
                ),
            )
        )
        expander = Neo4jLegalGraphExpander(
            store=store,
            config=LegalGraphConfig(enabled=True, complex_query_only=True, expansion_depth=2),
        )

        graph_hits = expander.expand_from_hits(
            hits=(make_hit("seed"),),
            direct_records={"seed": make_record("seed")},
            intent=route_query("Dieu kien va ngoai le khi don phuong cham dut hop dong la gi?"),
        )

        self.assertGreater(store.calls[0][1]["depth"], 2)
        self.assertEqual(graph_hits[0].payload["graph_depth"], 3)
        self.assertEqual(graph_hits[0].payload["graph_edge_path"][1], "MENTIONS_CONCEPT")

    def test_no_seed_node_does_not_crash(self) -> None:
        expander = Neo4jLegalGraphExpander(
            store=FakeGraphStore(
                GraphExpansionResult(seed_chunk_ids=(), expanded_chunk_ids=())
            ),
            config=LegalGraphConfig(enabled=True, complex_query_only=False),
        )

        graph_hits = expander.expand_from_hits(
            hits=(make_hit("seed"),),
            direct_records={},
            intent=route_query("Dieu 35"),
        )

        self.assertEqual(graph_hits, ())

    def test_direct_article_lookup_uses_depth_one_and_no_expansion(self) -> None:
        store = FakeGraphStore(GraphExpansionResult(seed_chunk_ids=("seed",), expanded_chunk_ids=()))
        expander = Neo4jLegalGraphExpander(
            store=store,
            config=LegalGraphConfig(enabled=True, complex_query_only=True, expansion_depth=3),
        )
        intent = route_query("Dieu 35 quy dinh gi?")

        graph_hits = expander.expand_from_hits(
            hits=(make_hit("seed"),),
            direct_records={"seed": make_record("seed")},
            intent=intent,
        )

        self.assertEqual(graph_hits, ())
        self.assertEqual(expander.expansion_depth_for_intent(intent), 1)
        self.assertIn("direct_article_lookup", classify_graph_query_intent(intent))
        self.assertEqual(store.calls, [])

    def test_minor_worker_policy_uses_multidocument_depth_and_references(self) -> None:
        intent = route_query("Ng\u01b0\u1eddi 14 tu\u1ed5i c\u00f3 \u0111\u01b0\u1ee3c l\u00e0m vi\u1ec7c kh\u00f4ng?")
        expander = Neo4jLegalGraphExpander(
            store=FakeGraphStore(GraphExpansionResult(seed_chunk_ids=(), expanded_chunk_ids=())),
            config=LegalGraphConfig(enabled=True, expansion_depth=2),
        )

        query_types = classify_graph_query_intent(intent)
        references = graph_priority_references_for_intent(intent)

        self.assertIn("minor_worker", query_types)
        self.assertEqual(expander.expansion_depth_for_intent(intent), 4)
        self.assertTrue(any(ref.document_id == "45-2019-qh14" and "145" in ref.article_numbers for ref in references))
        self.assertTrue(any(ref.document_id == "thong-tu-09-2020-tt-bldtbxh" for ref in references))

    def test_retirement_contract_and_dispute_policy_classification(self) -> None:
        retirement = route_query("N\u1eef ngh\u1ec9 h\u01b0u n\u0103m 2026 th\u00ec bao nhi\u00eau tu\u1ed5i?")
        contract = route_query("H\u1ee3p \u0111\u1ed3ng lao \u0111\u1ed9ng c\u1ea7n c\u00f3 nh\u1eefng n\u1ed9i dung g\u00ec?")
        dispute = route_query("Tranh ch\u1ea5p sa th\u1ea3i c\u00f3 c\u1ea7n h\u00f2a gi\u1ea3i tr\u01b0\u1edbc khi ki\u1ec7n kh\u00f4ng?")

        self.assertIn("retirement_age", classify_graph_query_intent(retirement))
        self.assertTrue(
            any(ref.document_id == "nghi-dinh-135-2020-nd-cp" for ref in graph_priority_references_for_intent(retirement))
        )
        self.assertIn("labor_contract", classify_graph_query_intent(contract))
        self.assertIn("labor_contract_content", classify_graph_query_intent(contract))
        self.assertTrue(
            any(ref.document_id == "thong-tu-10-2020-tt-bldtbxh" for ref in graph_priority_references_for_intent(contract))
        )
        self.assertIn("labor_dispute", classify_graph_query_intent(dispute))
        self.assertIn("litigation", classify_graph_query_intent(dispute))
        self.assertIn("labor_dispute_litigation", classify_graph_query_intent(dispute))
        self.assertTrue(
            any(ref.document_id == "92-2015-qh13-labor-only" for ref in graph_priority_references_for_intent(dispute))
        )

    def test_overtime_policy_classification_distinguishes_limits_from_pay(self) -> None:
        limits = route_query(
            "Tr\u01b0\u1eddng h\u1ee3p n\u00e0o \u0111\u01b0\u1ee3c l\u00e0m th\u00eam gi\u1edd v\u00e0 gi\u1edbi h\u1ea1n l\u00e0m th\u00eam theo th\u00e1ng l\u00e0 bao nhi\u00eau?"
        )
        pay = route_query(
            "L\u01b0\u01a1ng l\u00e0m th\u00eam gi\u1edd v\u00e0o ban \u0111\u00eam \u0111\u01b0\u1ee3c tr\u1ea3 nh\u01b0 th\u1ebf n\u00e0o?"
        )

        self.assertIn("overtime_conditions_and_limits", classify_graph_query_intent(limits))
        self.assertNotIn("overtime_pay", classify_graph_query_intent(limits))
        self.assertTrue(
            any(
                ref.document_id == "45-2019-qh14" and "107" in ref.article_numbers
                for ref in graph_priority_references_for_intent(limits)
            )
        )
        self.assertTrue(
            any(
                ref.document_id == "nghi-dinh-145-2020-nd-cp" and "60" in ref.article_numbers
                for ref in graph_priority_references_for_intent(limits)
            )
        )
        self.assertIn("overtime_pay", classify_graph_query_intent(pay))
        self.assertTrue(
            any(
                ref.document_id == "45-2019-qh14" and "98" in ref.article_numbers
                for ref in graph_priority_references_for_intent(pay)
            )
        )

    def test_specific_termination_policy_classification(self) -> None:
        unlawful_employee = route_query(
            "Ng\u01b0\u1eddi lao \u0111\u1ed9ng \u0111\u01a1n ph\u01b0\u01a1ng ch\u1ea5m d\u1ee9t h\u1ee3p \u0111\u1ed3ng tr\u00e1i lu\u1eadt th\u00ec ph\u1ea3i b\u1ed3i th\u01b0\u1eddng g\u00ec?"
        )
        structural = route_query("C\u00f4ng ty thay \u0111\u1ed5i c\u01a1 c\u1ea5u th\u00ec ph\u1ea3i tr\u1ea3 tr\u1ee3 c\u1ea5p g\u00ec?")
        no_notice = route_query(
            "Khi n\u00e0o ng\u01b0\u1eddi lao \u0111\u1ed9ng \u0111\u01b0\u1ee3c ngh\u1ec9 vi\u1ec7c kh\u00f4ng c\u1ea7n b\u00e1o tr\u01b0\u1edbc?"
        )

        self.assertIn(
            "illegal_unilateral_termination_by_employee",
            classify_graph_query_intent(unlawful_employee),
        )
        self.assertTrue(
            any(
                ref.document_id == "45-2019-qh14" and "40" in ref.article_numbers
                for ref in graph_priority_references_for_intent(unlawful_employee)
            )
        )
        self.assertIn("structural_change_job_loss_allowance", classify_graph_query_intent(structural))
        self.assertTrue(
            any(
                ref.document_id == "nghi-dinh-145-2020-nd-cp" and "8" in ref.article_numbers
                for ref in graph_priority_references_for_intent(structural)
            )
        )
        self.assertIn("no_notice_resignation", classify_graph_query_intent(no_notice))

    def test_legal_definition_policy_classification_targets_article_3(self) -> None:
        definition = route_query("Nguoi lao dong duoc dinh nghia nhu the nao theo Bo luat Lao dong 2019?")

        self.assertIn("legal_definition_lookup", classify_graph_query_intent(definition))
        self.assertTrue(
            any(
                ref.document_id == "45-2019-qh14" and "3" in ref.article_numbers
                for ref in graph_priority_references_for_intent(definition)
            )
        )

    def test_comparison_policy_classification_retrieves_both_sides(self) -> None:
        comparison = route_query(
            "So sanh trach nhiem khi nguoi lao dong don phuong cham dut hop dong trai luat voi truong hop cong ty thay doi co cau phai tro cap?"
        )

        self.assertIn(
            "compare_employee_unlawful_termination_vs_structural_change",
            classify_graph_query_intent(comparison),
        )
        priority_refs = graph_priority_references_for_intent(comparison)
        self.assertTrue(
            any(
                ref.document_id == "45-2019-qh14" and "40" in ref.article_numbers
                for ref in priority_refs
            )
        )
        self.assertTrue(
            any(
                ref.document_id == "45-2019-qh14" and "42" in ref.article_numbers
                for ref in priority_refs
            )
        )
        self.assertTrue(
            any(
                ref.document_id == "45-2019-qh14" and "47" in ref.article_numbers
                for ref in priority_refs
            )
        )

    def test_dedupe_search_hits_keeps_first_chunk_id(self) -> None:
        hits = dedupe_search_hits((make_hit("a", 0.5), make_hit("b", 0.4), make_hit("a", 0.9)))

        self.assertEqual([hit.chunk_id for hit in hits], ["a", "b"])
        self.assertEqual(hits[0].score, 0.5)


if __name__ == "__main__":
    unittest.main()
