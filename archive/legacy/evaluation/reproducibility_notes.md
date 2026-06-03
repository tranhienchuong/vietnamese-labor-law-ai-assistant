# Reproducibility Notes

Prepared: 2026-05-27

These notes describe how to reproduce the final thesis evaluation artifacts for the Vietnamese labor-law RAG assistant. The final reported metrics use the already built chunk, vector-index, and Neo4j graph artifacts. Rebuilding those artifacts is optional and should only be done when intentionally regenerating the retrieval infrastructure.

## Environment Assumptions

- Working directory: repository root, for example `c:\Workspace\vietnamese-labor-law-ai-assistant`.
- Shell: PowerShell on Windows.
- Python: repository virtual environment at `.venv`.
- Embedding provider used for evaluation: `sentence_transformers`.
- Dense embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Vector store: Qdrant collection `vietnamese_labor_law_chunks`.
- Neo4j connection settings are available through `.env` or command-line flags.
- Evaluation commands are run with `-X utf8` to avoid Windows console encoding issues.

## Required Inputs

Core evaluation inputs:

- `artifacts/evaluation/end_to_end_expanded_benchmark.jsonl`
- `artifacts/index/current.json`
- `artifacts/chunks/legal_chunks_enriched.jsonl`
- Existing Qdrant collection containing 1,556 indexed chunks.
- Existing Neo4j graph built from the enriched chunks and resolved reference edges.

Summary inputs used by the final thesis report:

- `artifacts/evaluation/retrieval_modes_expanded_summary.json`
- `artifacts/evaluation/retrieval_modes_expanded_report.md`
- `artifacts/evaluation/end_to_end_expanded_results.json`
- `artifacts/evaluation/end_to_end_expanded_report.md`
- `artifacts/chunks/legal_chunks_summary.md`
- `artifacts/chunks/legal_chunks_enriched_summary.md`
- `artifacts/graph/reference_edges_summary.md`
- `artifacts/graph/legal_graph_build_summary.md`
- `artifacts/index/vector_index_summary.md`

## Expected Outputs

Retrieval-mode evaluation outputs:

- `artifacts/evaluation/retrieval_modes_expanded_results.csv`
- `artifacts/evaluation/retrieval_modes_expanded_summary.json`
- `artifacts/evaluation/retrieval_modes_expanded_report.md`

End-to-end evaluation outputs:

- `artifacts/evaluation/end_to_end_expanded_results.json`
- `artifacts/evaluation/end_to_end_expanded_results.csv`
- `artifacts/evaluation/end_to_end_expanded_report.md`

Final thesis package outputs:

- `artifacts/evaluation/final_thesis_evaluation_report.md`
- `artifacts/evaluation/final_thesis_evaluation_summary.json`
- `artifacts/evaluation/reproducibility_notes.md`

## Expected Metric Values

Retrieval modes on the 69-query benchmark:

| Mode | Recall@10 | Required citation coverage | Forbidden citation violation |
| --- | ---: | ---: | ---: |
| Vector-only | 0.012 | 0.012 | 0.000 |
| Hybrid | 0.920 | 0.920 | 0.000 |
| Graph-augmented | 1.000 | 1.000 | 0.000 |

End-to-end graph-augmented evaluation:

| Metric | Expected value |
| --- | ---: |
| Benchmark queries | 69 |
| End-to-end passed | True |
| End-to-end pass rate | 1.000 |
| Retrieval pass rate | 1.000 |
| Answer pass rate | 1.000 |
| Citation pass rate | 1.000 |
| Quality pass rate | 1.000 |
| Average final quality score | 100.00 |
| Low-information quotes | 0 |
| Unsupported article numbers | None |
| Unretrieved citations | None |

## Rerun Retrieval Evaluation

From the repository root:

```powershell
.venv\Scripts\python.exe -X utf8 scripts/evaluate_retrieval_modes.py `
  --benchmark-path artifacts/evaluation/end_to_end_expanded_benchmark.jsonl `
  --output-dir artifacts/evaluation `
  --output-prefix retrieval_modes_expanded `
  --top-k 10 `
  --prefetch-limit 24 `
  --device cpu `
  --embedding-provider sentence_transformers
```

Expected outputs:

- `artifacts/evaluation/retrieval_modes_expanded_results.csv`
- `artifacts/evaluation/retrieval_modes_expanded_summary.json`
- `artifacts/evaluation/retrieval_modes_expanded_report.md`

The expected graph-augmented Recall@10 and required citation coverage are both 1.000.

## Rerun End-to-End Evaluation

Run this after the retrieval-mode summary exists:

```powershell
.venv\Scripts\python.exe -X utf8 scripts/evaluate_end_to_end_rag.py `
  --benchmark-path artifacts/evaluation/end_to_end_expanded_benchmark.jsonl `
  --output-dir artifacts/evaluation `
  --output-prefix end_to_end_expanded `
  --comparison-summary-path artifacts/evaluation/retrieval_modes_expanded_summary.json `
  --top-k 10 `
  --prefetch-limit 24 `
  --max-answer-contexts 8 `
  --provider extractive `
  --device cpu `
  --embedding-provider sentence_transformers
```

Expected outputs:

- `artifacts/evaluation/end_to_end_expanded_results.json`
- `artifacts/evaluation/end_to_end_expanded_results.csv`
- `artifacts/evaluation/end_to_end_expanded_report.md`

Expected result: `passed: true`, `benchmark_count: 69`, `end_to_end_pass_rate: 1.000`, and `citation_validation_pass_rate: 1.000`.

## Regenerate Vector Index

This step is not required to reproduce the final evaluation package if the current Qdrant index is already available. Use it only when intentionally rebuilding the vector index from the enriched chunk artifact.

```powershell
.venv\Scripts\python.exe scripts/build_index.py `
  --chunk-file artifacts/chunks/legal_chunks_enriched.jsonl `
  --artifacts-dir artifacts `
  --dense-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 `
  --collection-name vietnamese_labor_law_chunks `
  --device cpu
```

Expected validation summary:

- Chunk count: 1,556
- Document count: 6
- Vector dimension: 384
- All chunks indexed: True
- Duplicate chunk IDs: 0
- Missing retrieval text: 0
- Missing citation text: 0
- Missing document ID: 0
- Missing normative rank: 0
- Empty vector payloads: 0

## Rebuild Neo4j Graph

This step is not required to reproduce the final evaluation package if the current Neo4j graph is already available. Use it only when intentionally rebuilding the graph from the enriched chunks and reference edges.

```powershell
.venv\Scripts\python.exe scripts/build_legal_graph.py `
  --chunks-path artifacts/chunks/legal_chunks_enriched.jsonl `
  --reference-edges-path artifacts/graph/reference_edges.jsonl `
  --summary-path artifacts/graph/legal_graph_build_summary.md `
  --with-concepts `
  --with-references `
  --with-normative-hierarchy `
  --reset
```

If not using environment variables, provide Neo4j connection flags:

```powershell
  --neo4j-uri <NEO4J_URI> `
  --neo4j-user <NEO4J_USER> `
  --neo4j-password <NEO4J_PASSWORD> `
  --neo4j-database <NEO4J_DATABASE>
```

Expected validation summary:

- Documents: 6
- Articles: 411
- Clauses: 1,235
- Points: 774
- Appendices: 35
- Evidence chunks: 1,556
- Resolved reference edges loaded: 948
- Orphan evidence chunks: 0
- Unresolved reference edges loaded: 0
- Passed: True

## Optional Full Artifact Regeneration

Use these steps only if rebuilding from curated text inputs. They are not required for a normal evaluation rerun.

```powershell
.venv\Scripts\python.exe scripts/build_legal_chunks.py `
  --input-dir corpus/cleaned `
  --output-dir artifacts/chunks

.venv\Scripts\python.exe scripts/enrich_legal_chunks.py `
  --input artifacts/chunks/legal_chunks.jsonl `
  --output-dir artifacts/chunks

.venv\Scripts\python.exe scripts/build_reference_edges.py `
  --input artifacts/chunks/legal_chunks_enriched.jsonl `
  --output-dir artifacts/graph
```

After these optional steps, rebuild the vector index and Neo4j graph before rerunning evaluations.

## Careful Interpretation

- The 100% result is measured on the constructed 69-query benchmark.
- The benchmark is not a guarantee of universal legal correctness.
- The system is strongest for questions covered by the labor-law corpus and benchmark scope.
- External laws outside the indexed corpus may still require additional sources.
- Legal consultation should still be verified by qualified professionals for real cases.

## Limitations For Thesis

- The benchmark is manually and heuristically constructed.
- Answer quality is evaluated using deterministic rule-based validation, not human legal expert review or LLM-as-Judge evaluation.
- Answer generation is currently extractive and deterministic.
- The system depends on chunk quality and citation parsing.
- The vector-only baseline appears weak compared with hybrid and graph retrieval.
- Future work should include an expert-validated benchmark and LLM-as-judge or human evaluation.
