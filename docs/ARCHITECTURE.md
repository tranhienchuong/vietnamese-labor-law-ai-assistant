# Architecture

## Scope

Vietnamese Labor Law GraphRAG QA is a scoped legal information assistant over an indexed Vietnamese labor-law corpus. It retrieves and cites passages from the corpus; it does not provide legal advice and does not claim legal correctness outside the indexed sources.

The official corpus is the six-document thesis corpus:

| Document id | Source |
| --- | --- |
| `45-2019-qh14` | Labor Code 2019 |
| `92-2015-qh13-labor-only` | Civil Procedure Code 2015 labor-only scope |
| `nghi-dinh-135-2020-nd-cp` | Retirement age decree |
| `nghi-dinh-145-2020-nd-cp` | Labor Code implementation decree |
| `thong-tu-09-2020-tt-bldtbxh` | Child labor circular |
| `thong-tu-10-2020-tt-bldtbxh` | Labor mediation and arbitration circular |

## Retrieval And Answer Flow

```text
User query
  -> query routing and intent gating
  -> Qdrant hybrid retrieval
  -> Neo4j graph expansion
  -> coordinate/reference fallback
  -> rerank and context budget
  -> grounded generation
  -> citation validation
  -> final answer or refusal
```

The retrieval set used for generation is:

```text
C_final = Rerank(S_k ∪ C_graph ∪ C_fallback)
```

Where:

| Symbol | Meaning |
| --- | --- |
| `S_k` | Seed passages from Qdrant hybrid dense+sparse retrieval. |
| `C_graph` | Passages added by Neo4j expansion through article, document, reference, and hierarchy relationships. |
| `C_fallback` | Coordinate and reference fallback passages used when seed retrieval misses explicit article or document references. |
| `C_final` | Deduplicated, reranked, budgeted context passed to the generator. |

## Main Components

| Thesis component | Implementation path |
| --- | --- |
| Curated source validation | `scripts/validate_curated_legal_texts.py` |
| Hierarchical chunking | `scripts/build_legal_chunks.py` |
| Chunk enrichment | `scripts/enrich_legal_chunks.py` |
| Legal reference edge extraction | `scripts/build_reference_edges.py` |
| Qdrant hybrid retrieval index | `scripts/build_index.py`, `src/vn_labor_law_ai_assistant/retrieval/` |
| Query routing and scope guard | `src/vn_labor_law_ai_assistant/rules/routing_config.yaml`, `src/vn_labor_law_ai_assistant/core/scope_guard.py` |
| Neo4j legal graph build | `scripts/build_legal_graph.py`, `src/vn_labor_law_ai_assistant/graph/` |
| Graph expansion during retrieval | `src/vn_labor_law_ai_assistant/rag/graph_expansion.py` |
| Context rerank and budget | `src/vn_labor_law_ai_assistant/reranker.py`, `src/vn_labor_law_ai_assistant/query_answering.py` |
| Grounded generation | `src/vn_labor_law_ai_assistant/llm.py`, `src/vn_labor_law_ai_assistant/query_answering.py` |
| Citation validation | `src/vn_labor_law_ai_assistant/core/citation_validation.py` |
| Deterministic evaluation | `scripts/ablation_retrieval_100.py`, `scripts/evaluate_end_to_end_rag.py`, `scripts/compute_100_split_metrics.py` |

## Citation Contract

Generated answers must cite only passages present in the retrieved context. Citation validation checks that each cited legal location can be traced back to a retrieved chunk and that the answer refuses when the available context is insufficient.

The assistant should return an insufficient-context response when:

- the query is outside Vietnamese labor-law scope,
- the query asks for legal advice or a conclusion not grounded in the corpus,
- retrieval does not provide enough supporting passages, or
- generated citations cannot be validated against retrieved context.

## Official Artifacts

These artifacts represent the current thesis-aligned corpus and should not be replaced by archived exploratory outputs:

- `artifacts/chunks/legal_chunks_enriched.jsonl`
- `artifacts/index/current.json`
- `artifacts/graph/legal_graph_build_summary.json`
- `artifacts/evaluation/golden_benchmark_100_extended.jsonl`

Exploratory LLM-as-Judge and RAGAS runs are retained only as legacy material. They are not the final evaluation method for the thesis-aligned project.
