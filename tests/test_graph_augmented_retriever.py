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
                    "retrieval_source": "neo4j_graph_expansion",
                    "graph_seed_chunk_ids": ["seed"],
                    "graph_edge_path": ["REFERENCES"],
                    "graph_node_path": ["chunk:seed", "chunk:expanded"],
                    "graph_depth": 2,
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
        self.assertEqual(graph_hit.payload["retrieval_source"], "neo4j_graph_expansion")
        self.assertEqual(graph_hit.payload["graph_seed_chunk_ids"], ["seed"])
        self.assertIn(("expanded",), retriever._record_store.fetch_records_calls)


if __name__ == "__main__":
    unittest.main()
