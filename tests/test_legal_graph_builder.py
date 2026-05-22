from __future__ import annotations

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
        "citation_text": "Bo luat Lao dong 2019, Dieu 35, khoan 1",
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

        result = LegalGraphBuilder().build((decree_record,))
        edges = {(edge.source_id, edge.edge_type, edge.target_id) for edge in result.edges}

        self.assertIn(
            (
                "clause:nghi-dinh-145-2020-nd-cp:7:1",
                EdgeType.REFERENCES,
                "point:45-2019-qh14:35:1:d",
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


if __name__ == "__main__":
    unittest.main()
