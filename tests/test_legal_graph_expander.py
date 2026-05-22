from __future__ import annotations

import unittest

from vn_labor_law_ai_assistant.heuristic_router import route_query
from vn_labor_law_ai_assistant.rag.graph import GraphExpansionResult, LegalGraphConfig
from vn_labor_law_ai_assistant.rag.graph.expander import Neo4jLegalGraphExpander, dedupe_search_hits
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
        self.assertEqual(graph_hits[0].payload["retrieval_source"], "neo4j_graph_expansion")
        self.assertEqual(graph_hits[0].payload["graph_seed_chunk_ids"], ["seed"])
        self.assertEqual(graph_hits[0].payload["graph_depth"], 2)

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

    def test_dedupe_search_hits_keeps_first_chunk_id(self) -> None:
        hits = dedupe_search_hits((make_hit("a", 0.5), make_hit("b", 0.4), make_hit("a", 0.9)))

        self.assertEqual([hit.chunk_id for hit in hits], ["a", "b"])
        self.assertEqual(hits[0].score, 0.5)


if __name__ == "__main__":
    unittest.main()
