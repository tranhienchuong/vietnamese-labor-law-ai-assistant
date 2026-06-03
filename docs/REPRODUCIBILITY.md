# Reproducibility

This guide rebuilds the thesis-aligned Vietnamese Labor Law GraphRAG QA artifacts from the curated six-document corpus.

The commands assume they are run from the repository root:

```powershell
cd C:\Workspace\vietnamese-labor-law-ai-assistant
```

## Environment

Copy the example environment file and fill in only the services you need for the command being run:

```powershell
Copy-Item .env.example .env
```

Qdrant is required for index build and retrieval evaluation. Neo4j is required for graph build and graph-expansion integration. LLM keys are required only when running provider-backed answer generation; deterministic and extractive checks can run without them.

## 1. Preprocessing

Validate the curated legal text files:

```powershell
python scripts/validate_curated_legal_texts.py `
  --input-dir corpus/data/curated `
  --output-dir artifacts/validation
```

## 2. Chunking

Build hierarchical legal chunks:

```powershell
python scripts/build_legal_chunks.py `
  --input-dir corpus/data/curated `
  --output-dir artifacts/chunks `
  --max-chars 1200
```

## 3. Enrichment

Normalize coordinates, citation text, and article metadata:

```powershell
python scripts/enrich_legal_chunks.py `
  --input artifacts/chunks/legal_chunks.jsonl `
  --output-dir artifacts/chunks
```

## 4. Reference Edges

Extract legal cross-references used by graph expansion:

```powershell
python scripts/build_reference_edges.py `
  --input artifacts/chunks/legal_chunks_enriched.jsonl `
  --output-dir artifacts/graph
```

## 5. Qdrant Index

Build the hybrid retrieval index:

```powershell
python scripts/build_index.py `
  --chunks-dir artifacts/chunks `
  --chunk-file artifacts/chunks/legal_chunks_enriched.jsonl `
  --artifacts-dir artifacts/index `
  --dense-model keepitreal/vietnamese-sbert `
  --collection-name vietnamese_labor_law_chunks
```

The canonical index pointer is:

```text
artifacts/index/current.json
```

## 6. Neo4j Graph

Build the legal graph from the chunk and reference artifacts:

```powershell
python scripts/build_legal_graph.py `
  --index-path artifacts/index/current.json `
  --chunks-path artifacts/chunks/legal_chunks_enriched.jsonl `
  --reference-edges-path artifacts/graph/reference_edges.jsonl `
  --summary-path artifacts/graph/legal_graph_build_summary.json `
  --with-references `
  --with-normative-hierarchy
```

Use `--reset` only when intentionally replacing the Neo4j graph for this project.

## 7. Retrieval Ablation

Run deterministic retrieval ablation against the 100-query benchmark:

```powershell
python scripts/ablation_retrieval_100.py `
  --index-path artifacts/index/current.json `
  --benchmark-path artifacts/evaluation/golden_benchmark_100_extended.jsonl `
  --output-dir artifacts/evaluation `
  --output-prefix ablation_retrieval_100_final_candidate `
  --top-k 10 `
  --prefetch-limit 50
```

## 8. End-To-End Evaluation

Run end-to-end grounded answer evaluation. The default extractive provider avoids external LLM keys:

```powershell
python scripts/evaluate_end_to_end_rag.py `
  --index-path artifacts/index/current.json `
  --benchmark-path artifacts/evaluation/golden_benchmark_100_extended.jsonl `
  --output-dir artifacts/evaluation `
  --output-prefix end_to_end_100_final_candidate `
  --provider extractive `
  --top-k 10 `
  --prefetch-limit 50 `
  --max-answer-contexts 6
```

Compute the deterministic 100-query split metrics:

```powershell
python scripts/compute_100_split_metrics.py `
  --results-path artifacts/evaluation/end_to_end_100_final_candidate_results.json `
  --output-json artifacts/evaluation/benchmark_100_scope_guard_split_metrics.json `
  --failed-cases-csv artifacts/evaluation/benchmark_100_scope_guard_adjusted_failed_cases.csv
```

## Notes

The final evaluation path is deterministic and corpus-grounded. Legacy LLM-as-Judge, RAGAS, and exploratory reports are retained under `archive/legacy/` for provenance only and are not the final thesis evaluation method.
