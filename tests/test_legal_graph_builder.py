from __future__ import annotations

import json
from pathlib import Path
import unittest

from vn_labor_law_ai_assistant.rag.graph import LegalGraphBuilder
from vn_labor_law_ai_assistant.rag.graph.ontology import EdgeType, NodeType
from vn_labor_law_ai_assistant.rag.retrieval.models import RetrievedRecord


def make_record(
    chunk_id: str = "doc-dieu-35-k1",
    *,
    text: str = "1. Nguoi lao dong co quyen don phuong cham dut hop dong lao dong.",
) -> RetrievedRecord:
    payload = {
        "chunk_id": chunk_id,
        "document_id": "45-2019-qh14",
        "document_title": "Bo luat Lao dong 2019",
        "document_type": "bo_luat",
        "normative_rank": 1,
        "rank_label": "highest",
        "issuing_authority": "Quoc hoi",
        "article_number": "35",
        "article_title": "Quyen don phuong cham dut hop dong lao dong",
        "heading": "Dieu 35. Quyen don phuong cham dut hop dong lao dong",
        "chapter_heading": "Chuong III",
        "section_heading": "Muc 3",
        "level": "clause",
        "clause_ref": "1",
        "point_ref": None,
        "point_refs": ["a"],
        "chunk_type": "clause",
        "parent_chunk_id": None,
        "topic": ["hop_dong_lao_dong"],
        "actor": ["nguoi_lao_dong"],
        "issue_type": ["don_phuong_cham_dut"],
        "citation_text": "Bo luat Lao dong 2019, Dieu 35, khoan 1",
        "retrieval_text": "Bo luat Lao dong 2019, Dieu 35, khoan 1 quy dinh ve cham dut hop dong.",
        "source_file": "45_2019_QH14.txt",
        "document_hierarchy": {"chapter_heading": "Chuong III", "section_heading": "Muc 3"},
        "text": text,
    }
    return RetrievedRecord(
        chunk_id=chunk_id,
        parent_chunk_id=None,
        citation_text=str(payload["citation_text"]),
        text=text,
        dense_text="",
        sparse_text="",
        payload=payload,
    )


class LegalGraphBuilderTests(unittest.TestCase):
    def test_build_graph_from_mock_records(self) -> None:
        result = LegalGraphBuilder().build((make_record(),))

        node_types = {node.node_type for node in result.nodes}
        edge_types = {edge.edge_type for edge in result.edges}
        self.assertIn(NodeType.LEGAL_DOCUMENT, node_types)
        self.assertIn(NodeType.LEGAL_ARTICLE, node_types)
        self.assertIn(NodeType.LEGAL_CLAUSE, node_types)
        self.assertIn(NodeType.LEGAL_POINT, node_types)
        self.assertIn(NodeType.EVIDENCE_CHUNK, node_types)
        self.assertIn(NodeType.LEGAL_CONCEPT, node_types)
        self.assertIn(EdgeType.HAS_ARTICLE, edge_types)
        self.assertIn(EdgeType.HAS_CLAUSE, edge_types)
        self.assertIn(EdgeType.HAS_POINT, edge_types)
        self.assertIn(EdgeType.HAS_SOURCE_CHUNK, edge_types)
        self.assertIn(EdgeType.SOURCE_OF, edge_types)
        self.assertIn(EdgeType.MENTIONS_CONCEPT, edge_types)
        self.assertEqual(result.summary["documents"], 1)
        self.assertEqual(result.summary["articles"], 1)
        self.assertEqual(result.summary["clauses"], 1)
        self.assertEqual(result.summary["evidence_chunks"], 1)
        for edge in result.edges:
            self.assertTrue(edge.source_chunk_id)
            self.assertTrue(edge.extraction_method)
            self.assertGreaterEqual(edge.confidence, 0.0)
            self.assertIn("citation_text", edge.properties)

    def test_document_node_aggregates_chunk_provenance_and_build_metadata(self) -> None:
        result = LegalGraphBuilder().build(
            (
                make_record("doc-dieu-35-k1"),
                make_record("doc-dieu-35-k2"),
            ),
            build_metadata={"index_path": "artifacts/index", "manifest_hash": "abc"},
        )

        document_node = next(node for node in result.nodes if node.node_type == NodeType.LEGAL_DOCUMENT)
        self.assertEqual(document_node.source_chunk_id, "")
        self.assertEqual(
            document_node.properties["source_chunk_ids"],
            ["doc-dieu-35-k1", "doc-dieu-35-k2"],
        )
        self.assertEqual(document_node.properties["source_chunk_count"], 2)
        self.assertEqual(result.summary["build_metadata"]["manifest_hash"], "abc")

    def test_structural_only_skips_concepts_and_references(self) -> None:
        result = LegalGraphBuilder(
            with_concepts=False,
            with_references=False,
        ).build((make_record(text="quy dinh tai Dieu 46 ve tro cap thoi viec"),))

        edge_types = {edge.edge_type for edge in result.edges}
        node_types = {node.node_type for node in result.nodes}
        self.assertNotIn(NodeType.LEGAL_CONCEPT, node_types)
        self.assertNotIn(EdgeType.MENTIONS_CONCEPT, edge_types)
        self.assertNotIn(EdgeType.REFERENCES, edge_types)

    def test_builds_reference_and_guided_by_edges_from_decree_record(self) -> None:
        record = make_record(
            chunk_id="nd145-dieu-7",
            text="Thoi han bao truoc quy dinh tai diem d khoan 1 Dieu 35 cua Bo luat Lao dong.",
        )
        payload = dict(record.payload)
        payload.update(
            {
                "document_id": "nghi-dinh-145-2020-nd-cp",
                "document_title": "Nghi dinh 145/2020/ND-CP",
                "article_number": "7",
                "article_title": "Thoi han bao truoc",
                "heading": "Dieu 7. Thoi han bao truoc",
                "citation_text": "Nghi dinh 145/2020/ND-CP, Dieu 7",
            }
        )
        decree_record = RetrievedRecord(
            chunk_id="nd145-dieu-7",
            parent_chunk_id=None,
            citation_text="Nghi dinh 145/2020/ND-CP, Dieu 7",
            text=str(payload["text"]),
            dense_text="",
            sparse_text="",
            payload=payload,
        )

        result = LegalGraphBuilder(
            reference_edges=[
                {
                    "edge_id": "ref-1",
                    "source_id": "clause:nghi-dinh-145-2020-nd-cp:7:1",
                    "target_id": "point:45-2019-qh14:35:1:d",
                    "edge_type": "REFERENCES",
                    "source_chunk_id": "nd145-dieu-7",
                    "source_document_id": "nghi-dinh-145-2020-nd-cp",
                    "target_document_id": "45-2019-qh14",
                    "citation_text": "Nghi dinh 145/2020/ND-CP, Dieu 7",
                    "original_matched_text": "diem d khoan 1 Dieu 35 cua Bo luat Lao dong",
                    "normalized_matched_text": "diem d khoan 1 dieu 35 cua bo luat lao dong",
                    "extraction_method": "test",
                    "confidence": 0.9,
                    "resolved": True,
                },
                {
                    "edge_id": "ref-2",
                    "source_id": "document:45-2019-qh14",
                    "target_id": "document:nghi-dinh-145-2020-nd-cp",
                    "edge_type": "GUIDED_BY",
                    "source_chunk_id": "nd145-dieu-7",
                    "source_document_id": "nghi-dinh-145-2020-nd-cp",
                    "target_document_id": "45-2019-qh14",
                    "citation_text": "Nghi dinh 145/2020/ND-CP, Dieu 7",
                    "original_matched_text": "diem d khoan 1 Dieu 35 cua Bo luat Lao dong",
                    "normalized_matched_text": "diem d khoan 1 dieu 35 cua bo luat lao dong",
                    "extraction_method": "test",
                    "confidence": 0.9,
                    "resolved": True,
                },
            ]
        ).build((decree_record,))
        edges = {(edge.source_id, edge.edge_type, edge.target_id) for edge in result.edges}

        self.assertIn(
            (
                "clause:nghi-dinh-145-2020-nd-cp:dieu-7:khoan-1",
                EdgeType.REFERENCES,
                "point:45-2019-qh14:dieu-35:khoan-1:diem-d",
            ),
            edges,
        )
        self.assertIn(
            (
                "document:45-2019-qh14",
                EdgeType.GUIDED_BY,
                "document:nghi-dinh-145-2020-nd-cp",
            ),
            edges,
        )

    def test_graph_preserves_chunk_metadata_and_taxonomy_edges(self) -> None:
        result = LegalGraphBuilder(with_concepts=False, with_references=False).build((make_record(),))

        evidence_node = next(node for node in result.nodes if node.node_type == NodeType.EVIDENCE_CHUNK)
        self.assertEqual(evidence_node.properties["document_type"], "bo_luat")
        self.assertEqual(evidence_node.properties["normative_rank"], 1)
        self.assertEqual(evidence_node.properties["rank_label"], "highest")
        self.assertEqual(evidence_node.properties["retrieval_text"], make_record().payload["retrieval_text"])
        self.assertEqual(evidence_node.properties["source_file"], "45_2019_QH14.txt")
        self.assertEqual(
            evidence_node.properties["document_hierarchy"],
            {"chapter_heading": "Chuong III", "section_heading": "Muc 3"},
        )

        edge_types = {edge.edge_type for edge in result.edges}
        self.assertIn(EdgeType.MENTIONS_TOPIC, edge_types)
        self.assertIn(EdgeType.APPLIES_TO_ACTOR, edge_types)
        self.assertIn(EdgeType.HAS_ISSUE_TYPE, edge_types)
        self.assertEqual(result.summary["topic_nodes"], 1)
        self.assertEqual(result.summary["actor_nodes"], 1)
        self.assertEqual(result.summary["issue_type_nodes"], 1)
        self.assertEqual(result.summary["taxonomy_edges"], 3)

    def test_normative_hierarchy_edges_connect_labor_code_to_guiding_documents(self) -> None:
        decree = make_record("nd135-dieu-1")
        decree_payload = dict(decree.payload)
        decree_payload.update(
            {
                "document_id": "nghi-dinh-135-2020-nd-cp",
                "document_title": "Nghi dinh 135/2020/ND-CP",
                "document_type": "nghi_dinh",
                "normative_rank": 2,
                "rank_label": "middle",
                "article_number": "1",
                "citation_text": "Nghi dinh 135/2020/ND-CP, Dieu 1",
            }
        )
        decree_record = RetrievedRecord(
            chunk_id="nd135-dieu-1",
            parent_chunk_id=None,
            citation_text="Nghi dinh 135/2020/ND-CP, Dieu 1",
            text=str(decree_payload["text"]),
            dense_text="",
            sparse_text="",
            payload=decree_payload,
        )

        result = LegalGraphBuilder(
            with_concepts=False,
            with_references=False,
            with_normative_hierarchy=True,
        ).build((make_record(), decree_record))
        edges = {(edge.source_id, edge.edge_type, edge.target_id) for edge in result.edges}

        self.assertIn(
            (
                "document:45-2019-qh14",
                EdgeType.SUPERIOR_TO,
                "document:nghi-dinh-135-2020-nd-cp",
            ),
            edges,
        )
        self.assertIn(
            (
                "document:nghi-dinh-135-2020-nd-cp",
                EdgeType.MUST_COMPLY_WITH,
                "document:45-2019-qh14",
            ),
            edges,
        )
        document_nodes = {
            node.properties["document_id"]: node
            for node in result.nodes
            if node.node_type == NodeType.LEGAL_DOCUMENT
        }
        self.assertEqual(document_nodes["45-2019-qh14"].properties["normative_rank"], 1)
        self.assertEqual(document_nodes["nghi-dinh-135-2020-nd-cp"].properties["normative_rank"], 2)

    def test_reference_artifact_loading_skips_unresolved_and_balances_detail_inverse_edges(self) -> None:
        result = LegalGraphBuilder(
            with_concepts=False,
            reference_edges=[
                {
                    "edge_id": "detail",
                    "source_id": "article:nghi-dinh-145-2020-nd-cp:1",
                    "target_id": "article:45-2019-qh14:169",
                    "edge_type": "DETAILS",
                    "source_chunk_id": "nd145-dieu-1",
                    "source_document_id": "nghi-dinh-145-2020-nd-cp",
                    "target_document_id": "45-2019-qh14",
                    "citation_text": "Nghi dinh 145",
                    "original_matched_text": "Dieu 169 cua Bo luat Lao dong",
                    "normalized_matched_text": "dieu 169 cua bo luat lao dong",
                    "extraction_method": "test",
                    "confidence": 0.9,
                    "resolved": True,
                },
                {
                    "edge_id": "guided",
                    "source_id": "article:45-2019-qh14:169",
                    "target_id": "article:nghi-dinh-145-2020-nd-cp:1",
                    "edge_type": "GUIDED_BY",
                    "source_chunk_id": "nd145-dieu-1",
                    "source_document_id": "nghi-dinh-145-2020-nd-cp",
                    "target_document_id": "45-2019-qh14",
                    "citation_text": "Nghi dinh 145",
                    "original_matched_text": "Dieu 169 cua Bo luat Lao dong",
                    "normalized_matched_text": "dieu 169 cua bo luat lao dong",
                    "extraction_method": "inverse:test",
                    "confidence": 0.9,
                    "resolved": True,
                },
                {
                    "edge_id": "unresolved",
                    "source_id": "article:nghi-dinh-145-2020-nd-cp:1",
                    "target_id": "article:45-2019-qh14:999",
                    "edge_type": "REFERENCES",
                    "source_chunk_id": "nd145-dieu-1",
                    "source_document_id": "nghi-dinh-145-2020-nd-cp",
                    "target_document_id": "45-2019-qh14",
                    "resolved": False,
                },
            ],
        ).build((make_record(),))

        loaded_reference_ids = {
            edge.edge_id
            for edge in result.edges
            if edge.edge_type in {EdgeType.DETAILS, EdgeType.GUIDED_BY, EdgeType.REFERENCES}
        }
        self.assertIn("detail", loaded_reference_ids)
        self.assertIn("guided", loaded_reference_ids)
        self.assertNotIn("unresolved", loaded_reference_ids)
        self.assertEqual(result.summary["details_edges"], 1)
        self.assertEqual(result.summary["guided_by_edges"], 1)
        self.assertTrue(result.summary["details_guided_by_balanced"])
        self.assertEqual(result.summary["unresolved_edges_skipped"], 1)
        detail_edge = next(edge for edge in result.edges if edge.edge_id == "detail")
        self.assertEqual(detail_edge.properties["artifact_source_id"], "article:nghi-dinh-145-2020-nd-cp:1")
        self.assertEqual(detail_edge.properties["artifact_target_id"], "article:45-2019-qh14:169")
        self.assertEqual(detail_edge.properties["source_id"], "article:nghi-dinh-145-2020-nd-cp:dieu-1")
        self.assertEqual(detail_edge.properties["target_id"], "article:45-2019-qh14:dieu-169")
        self.assertEqual(detail_edge.properties["source_artifact"], "reference_edges")

    def test_normative_guides_edges_do_not_count_as_reference_artifact_edges(self) -> None:
        decree = make_record("nd135-dieu-1")
        decree_payload = dict(decree.payload)
        decree_payload.update(
            {
                "document_id": "nghi-dinh-135-2020-nd-cp",
                "document_title": "Nghi dinh 135/2020/ND-CP",
                "document_type": "nghi_dinh",
                "normative_rank": 2,
                "rank_label": "middle",
                "article_number": "1",
                "citation_text": "Nghi dinh 135/2020/ND-CP, Dieu 1",
            }
        )
        decree_record = RetrievedRecord(
            chunk_id="nd135-dieu-1",
            parent_chunk_id=None,
            citation_text="Nghi dinh 135/2020/ND-CP, Dieu 1",
            text=str(decree_payload["text"]),
            dense_text="",
            sparse_text="",
            payload=decree_payload,
        )

        result = LegalGraphBuilder(
            with_concepts=False,
            with_references=False,
            with_normative_hierarchy=True,
        ).build((make_record(), decree_record))
        edges = {(edge.source_id, edge.edge_type, edge.target_id) for edge in result.edges}

        self.assertIn(
            (
                "document:nghi-dinh-135-2020-nd-cp",
                EdgeType.GUIDES,
                "document:45-2019-qh14",
            ),
            edges,
        )
        self.assertEqual(result.summary["reference_edges"], 0)
        self.assertEqual(result.summary["normative_hierarchy_edges"], 5)
        self.assertEqual(result.summary["normative_hierarchy_edges_by_type"]["GUIDES"], 1)

    def test_no_orphan_chunks_and_appendix_chunks_attach_to_appendix(self) -> None:
        appendix_payload = dict(make_record("appendix-chunk").payload)
        appendix_payload.update(
            {
                "article_number": None,
                "article_title": None,
                "level": "appendix",
                "chunk_type": "appendix_table",
                "document_hierarchy": {
                    "appendix_id": "ND135_2020_Phu_Luc_I",
                    "appendix_heading": "PHU LUC I",
                },
            }
        )
        appendix_record = RetrievedRecord(
            chunk_id="appendix-chunk",
            parent_chunk_id=None,
            citation_text=str(appendix_payload["citation_text"]),
            text=str(appendix_payload["text"]),
            dense_text="",
            sparse_text="",
            payload=appendix_payload,
        )

        result = LegalGraphBuilder(with_concepts=False, with_references=False).build((appendix_record,))
        node_types = {node.node_type for node in result.nodes}
        edge_types = {edge.edge_type for edge in result.edges}

        self.assertIn(NodeType.LEGAL_APPENDIX, node_types)
        self.assertIn(EdgeType.HAS_APPENDIX, edge_types)
        self.assertEqual(result.summary["appendices"], 1)
        self.assertEqual(result.summary["orphan_evidence_chunks"], 0)
        self.assertTrue(result.summary["validation"]["no_orphan_evidence_chunks"])

    def test_full_enriched_artifacts_satisfy_graph_validation_requirements(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        chunks_path = repo_root / "artifacts" / "chunks" / "legal_chunks_enriched.jsonl"
        reference_edges_path = repo_root / "artifacts" / "graph" / "reference_edges.jsonl"
        if not chunks_path.exists() or not reference_edges_path.exists():
            self.skipTest("legal graph artifacts are not available")

        chunks = [
            json.loads(line)
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        reference_edges = [
            json.loads(line)
            for line in reference_edges_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        result = LegalGraphBuilder(
            with_concepts=False,
            with_references=True,
            with_normative_hierarchy=True,
            reference_edges=reference_edges,
        ).build(chunks, build_metadata={"manifest_record_count": 1556})

        self.assertEqual(result.summary["evidence_chunks"], 1556)
        self.assertEqual(result.summary["documents"], 6)
        self.assertEqual(result.summary["unresolved_edges_skipped"], 111)
        self.assertEqual(result.summary["orphan_evidence_chunks"], 0)
        self.assertEqual(result.summary["details_edges"], result.summary["guided_by_edges"])
        self.assertEqual(result.summary["nd145_labor_code_article_count"], 75)
        self.assertTrue(result.summary["validation"]["evidence_chunk_count_matches_index"])
        self.assertTrue(result.summary["validation"]["all_documents_have_correct_normative_rank"])
        self.assertTrue(result.summary["validation"]["labor_code_hierarchy_connections_present"])
        self.assertTrue(result.summary["validation"]["tt09_labor_code_article_links_present"])
        self.assertTrue(result.summary["validation"]["tt10_labor_code_article_links_present"])
        self.assertTrue(result.summary["validation"]["nd135_labor_code_article_169_link_present"])
        self.assertTrue(result.summary["validation"]["nd145_connects_to_multiple_labor_code_articles"])
        self.assertTrue(result.summary["validation"]["blttds_labor_litigation_taxonomy_present"])


if __name__ == "__main__":
    unittest.main()
