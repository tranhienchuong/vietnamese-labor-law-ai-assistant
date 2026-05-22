from __future__ import annotations

import unittest

from vn_labor_law_ai_assistant.rag.graph.cross_reference_parser import build_reference_edges
from vn_labor_law_ai_assistant.rag.graph.ontology import EdgeType
from vn_labor_law_ai_assistant.rag.retrieval.models import RetrievedRecord


class CrossReferenceParserTests(unittest.TestCase):
    def test_reference_edges_use_normalized_span_not_raw_source_span(self) -> None:
        payload = {
            "chunk_id": "nd145-dieu-7",
            "document_id": "nghi-dinh-145-2020-nd-cp",
            "document_title": "Nghi dinh 145/2020/ND-CP",
            "article_number": "7",
            "heading": "Dieu 7. Thoi han bao truoc",
            "level": "clause",
            "clause_ref": "1",
            "citation_text": "Nghi dinh 145/2020/ND-CP, Dieu 7, khoan 1",
        }
        record = RetrievedRecord(
            chunk_id="nd145-dieu-7",
            parent_chunk_id=None,
            citation_text=str(payload["citation_text"]),
            text="Theo quy định tại điểm d khoản 1 Điều 35 của Bộ luật Lao động.",
            dense_text="",
            sparse_text="",
            payload=payload,
        )

        reference_edge = next(
            edge for edge in build_reference_edges(record) if edge.edge_type == EdgeType.REFERENCES
        )

        self.assertIn("normalized_matched_text", reference_edge.properties)
        self.assertIn("normalized_source_span", reference_edge.properties)
        self.assertIn("original_matched_text", reference_edge.properties)
        self.assertIn("raw_source_span", reference_edge.properties)
        self.assertNotIn("source_span", reference_edge.properties)
        self.assertIn("điểm d khoản 1 Điều 35", reference_edge.properties["original_matched_text"])


if __name__ == "__main__":
    unittest.main()
