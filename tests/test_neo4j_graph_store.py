from __future__ import annotations

import unittest

from vn_labor_law_ai_assistant.rag.graph import LegalGraphEdge, LegalGraphNode, Neo4jLegalGraphStore
from vn_labor_law_ai_assistant.rag.graph.ontology import EdgeType, NodeType


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def run(self, query: str, **kwargs: object):
        self.calls.append((query, kwargs))
        return []


class FakeDriver:
    def __init__(self) -> None:
        self.session_obj = FakeSession()
        self.closed = False

    def session(self, *, database: str):
        return self.session_obj

    def close(self) -> None:
        self.closed = True


class Neo4jLegalGraphStoreTests(unittest.TestCase):
    def test_upsert_node_and_edge_uses_neo4j_labels_and_provenance(self) -> None:
        driver = FakeDriver()
        store = Neo4jLegalGraphStore(
            uri="bolt://fake",
            user="neo4j",
            password="password",
            driver=driver,
            verify_connectivity=False,
        )
        node = LegalGraphNode(
            node_id="article:doc:35",
            node_type=NodeType.LEGAL_ARTICLE,
            name="Dieu 35",
            normalized_name="dieu 35",
            source_chunk_id="chunk-35",
            properties={
                "citation_text": "Dieu 35",
                "extraction_method": "structural_metadata",
                "confidence": 1.0,
            },
        )
        edge = LegalGraphEdge(
            edge_id="document:doc|HAS_ARTICLE|article:doc:35|chunk-35",
            source_id="document:doc",
            target_id="article:doc:35",
            edge_type=EdgeType.HAS_ARTICLE,
            confidence=1.0,
            source_chunk_id="chunk-35",
            extraction_method="structural_metadata",
            properties={"citation_text": "Dieu 35"},
        )

        store.setup_schema()
        store.upsert_nodes((node,))
        store.upsert_edges((edge,))
        store.close()

        queries = "\n".join(query for query, _ in driver.session_obj.calls)
        self.assertIn("CREATE CONSTRAINT legal_node_node_id", queries)
        self.assertIn("CREATE INDEX evidence_chunk_topic", queries)
        self.assertIn("CREATE INDEX evidence_chunk_actor", queries)
        self.assertIn("CREATE INDEX evidence_chunk_issue_type", queries)
        self.assertIn("UNWIND $rows AS row", queries)
        self.assertIn("SET n:Legal_Article", queries)
        self.assertIn("MERGE (source)-[r:HAS_ARTICLE", queries)
        node_call = next(call for call in driver.session_obj.calls if "SET n:Legal_Article" in call[0])
        self.assertEqual(node_call[1]["rows"][0]["properties"]["source_chunk_id"], "chunk-35")
        edge_call = next(call for call in driver.session_obj.calls if "HAS_ARTICLE" in call[0])
        self.assertEqual(
            edge_call[1]["rows"][0]["properties"]["extraction_method"],
            "structural_metadata",
        )
        self.assertTrue(driver.closed)


if __name__ == "__main__":
    unittest.main()
