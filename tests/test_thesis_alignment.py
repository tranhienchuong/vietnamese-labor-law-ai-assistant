from __future__ import annotations

import json
from pathlib import Path
import unittest

import yaml

from vn_labor_law_ai_assistant.heuristic_router import route_query
from vn_labor_law_ai_assistant.rag.answering import validate_grounded_answer
from vn_labor_law_ai_assistant.rag.answering.schema import ParsedAnswer
from vn_labor_law_ai_assistant.rag.graph import GraphExpansionResult, LegalGraphConfig
from vn_labor_law_ai_assistant.rag.graph.expander import Neo4jLegalGraphExpander
from vn_labor_law_ai_assistant.rag.retrieval.models import RetrievedRecord, RetrievalContext, SearchHit
from vn_labor_law_ai_assistant.rag.scope_guard import assess_scope


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DOCUMENT_IDS = {
    "45-2019-qh14",
    "92-2015-qh13-labor-only",
    "nghi-dinh-135-2020-nd-cp",
    "nghi-dinh-145-2020-nd-cp",
    "thong-tu-09-2020-tt-bldtbxh",
    "thong-tu-10-2020-tt-bldtbxh",
}


def read_json(path: str) -> dict[str, object]:
    return json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))


def make_context(
    *,
    chunk_id: str = "45-2019-qh14::article-3",
    citation_text: str = "Labor Code 2019, Dieu 3",
    text: str = "Dieu 3 dinh nghia nguoi lao dong.",
) -> RetrievalContext:
    return RetrievalContext(
        chunk_id=chunk_id,
        citation_text=citation_text,
        text=text,
        payload={
            "chunk_id": chunk_id,
            "document_id": "45-2019-qh14",
            "article_number": "3",
            "citation_text": citation_text,
            "retrieval_text": text,
        },
        score=1.0,
        matched_chunk_ids=(chunk_id,),
        matched_citations=(citation_text,),
    )


class FakeGraphStore:
    def __init__(self, result: GraphExpansionResult) -> None:
        self.result = result

    def expand_from_chunk_ids(self, *_args, **_kwargs) -> GraphExpansionResult:
        return self.result


class ThesisAlignmentTests(unittest.TestCase):
    def test_chunk_artifact_has_no_duplicate_or_missing_required_fields(self) -> None:
        chunk_path = REPO_ROOT / "artifacts/chunks/legal_chunks_enriched.jsonl"
        seen_chunk_ids: set[str] = set()
        duplicate_chunk_ids: set[str] = set()
        missing_citation_text: list[str] = []
        missing_article_number_on_legal_units: list[str] = []

        for line_number, line in enumerate(chunk_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            chunk_id = str(payload.get("chunk_id") or f"line:{line_number}")
            if chunk_id in seen_chunk_ids:
                duplicate_chunk_ids.add(chunk_id)
            seen_chunk_ids.add(chunk_id)
            if not str(payload.get("citation_text") or "").strip():
                missing_citation_text.append(chunk_id)
            if str(payload.get("level") or "") != "appendix" and not str(
                payload.get("article_number") or ""
            ).strip():
                missing_article_number_on_legal_units.append(chunk_id)

        summary = read_json("artifacts/chunks/legal_chunks_summary.json")

        self.assertEqual(len(seen_chunk_ids), 1556)
        self.assertEqual(duplicate_chunk_ids, set())
        self.assertEqual(missing_citation_text, [])
        self.assertEqual(missing_article_number_on_legal_units, [])
        self.assertEqual(summary["chunk_count"], 1556)
        self.assertEqual(summary["duplicate_chunk_id_count"], 0)
        self.assertEqual(summary["chunks_missing_citation_text"], [])
        self.assertEqual(summary["chunks_missing_article_number"], [])

    def test_index_manifest_is_consistent_with_six_document_corpus(self) -> None:
        manifest = read_json("artifacts/index/current.json")
        documents = {item["document_id"] for item in manifest["indexed_documents"]}

        self.assertEqual(manifest["document_count"], 6)
        self.assertEqual(manifest["chunk_count"], 1556)
        self.assertEqual(manifest["embedding_model"], "keepitreal/vietnamese-sbert")
        self.assertEqual(manifest["dense_model_name"], "keepitreal/vietnamese-sbert")
        self.assertEqual(manifest["vector_dimension"], 768)
        self.assertEqual(manifest["collection_name"], "vietnamese_labor_law_chunks")
        self.assertEqual(manifest["record_source"], "qdrant_payload")
        self.assertEqual(documents, OFFICIAL_DOCUMENT_IDS)
        self.assertTrue(manifest["validation"]["passed"])

    def test_routing_config_covers_all_official_documents(self) -> None:
        config_path = REPO_ROOT / "src/vn_labor_law_ai_assistant/rules/routing_config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(set(config["documents"]), OFFICIAL_DOCUMENT_IDS)
        self.assertEqual(set(config["document_descriptions"]), OFFICIAL_DOCUMENT_IDS)
        alias_values = set(config["label_aliases"].values())
        self.assertTrue(OFFICIAL_DOCUMENT_IDS.issubset(alias_values))

    def test_graph_summary_and_mock_expansion_return_valid_chunk_ids(self) -> None:
        summary = read_json("artifacts/graph/legal_graph_build_summary.json")
        self.assertEqual(summary["documents"], 6)
        self.assertEqual(summary["evidence_chunks"], 1556)
        self.assertEqual(summary["orphan_evidence_chunks"], 0)
        self.assertTrue(summary["neo4j_validation"]["passed"])

        expanded_chunk_id = "45-2019-qh14::article-3"
        expander = Neo4jLegalGraphExpander(
            store=FakeGraphStore(
                GraphExpansionResult(
                    seed_chunk_ids=("seed",),
                    expanded_chunk_ids=(expanded_chunk_id,),
                    paths=(
                        {
                            "chunk_id": expanded_chunk_id,
                            "graph_depth": 2,
                            "graph_edge_path": ["SOURCE_OF", "REFERENCES"],
                            "graph_node_path": ["chunk:seed", f"chunk:{expanded_chunk_id}"],
                            "graph_confidence": 0.9,
                        },
                    ),
                )
            ),
            config=LegalGraphConfig(enabled=True, complex_query_only=False),
        )
        hits = expander.expand_from_hits(
            hits=(
                SearchHit(
                    chunk_id="seed",
                    qdrant_point_id="point-seed",
                    score=1.0,
                    citation_text="Labor Code 2019, Dieu 3",
                    payload={"chunk_id": "seed"},
                ),
            ),
            direct_records={
                "seed": RetrievedRecord(
                    chunk_id="seed",
                    parent_chunk_id=None,
                    citation_text="Labor Code 2019, Dieu 3",
                    text="seed text",
                    dense_text="",
                    sparse_text="",
                    payload={"chunk_id": "seed"},
                )
            },
            intent=route_query("So sanh Dieu 3 va dieu lien quan"),
        )

        self.assertEqual([hit.chunk_id for hit in hits], [expanded_chunk_id])
        self.assertTrue(all(hit.payload["retrieval_source"] == "graph" for hit in hits))

    def test_generated_citations_must_exist_in_retrieved_context(self) -> None:
        parsed = ParsedAnswer(
            answer="Can cu Labor Code 2019, Dieu 99 thi...",
            legal_basis=("Labor Code 2019, Dieu 99",),
            evidence_quotes=(),
            insufficient_context=False,
            notes="",
            raw_content="",
        )

        validation = validate_grounded_answer(parsed, (make_context(),))

        self.assertFalse(validation.passed)
        self.assertFalse(validation.citations_allowed)
        self.assertEqual(validation.unretrieved_citations, ("Labor Code 2019, Dieu 99",))

    def test_out_of_corpus_query_returns_insufficient_context_decision(self) -> None:
        decision = assess_scope(
            "Tro cap thoi viec co tinh thue thu nhap ca nhan khong?",
            (make_context(),),
        )

        self.assertTrue(decision.out_of_scope)
        self.assertTrue(decision.reason)


if __name__ == "__main__":
    unittest.main()
