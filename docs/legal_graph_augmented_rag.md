# Legal Graph-Augmented RAG

The implementation keeps the current Hybrid RAG pipeline intact:

```text
User query
-> Query routing
-> Qdrant hybrid retrieval
-> Seed chunks
-> Neo4j graph expansion
-> Expanded evidence chunks
-> Merge and deduplicate
-> Heuristic rerank
-> Semantic rerank
-> Context assembly
-> LLM answer generation
```

Neo4j is only an evidence expansion layer. It runs after Qdrant retrieval and
before reranking, so graph-expanded chunks do not automatically outrank seed
chunks. They are merged with Qdrant hits, deduplicated by `chunk_id`, and then
passed through the existing heuristic and semantic rerankers.

Runtime switch:

```text
LEGAL_GRAPH_ENABLED=false
```

When this flag is false, the retriever does not create a Neo4j driver and the
previous behavior is preserved. When the flag is true, Neo4j must be reachable
or startup raises a clear runtime error.

Graph-expanded hit provenance is attached to the hit payload:

- `retrieval_source="neo4j_graph_expansion"`
- `graph_seed_chunk_ids`
- `graph_edge_path`
- `graph_node_path`
- `graph_depth`
- `graph_confidence`

Build output is written to:

```text
artifacts/graph/legal_graph_build_summary.json
```

Evaluation outputs are written to:

```text
artifacts/eval/baseline_hybrid_results.csv
artifacts/eval/neo4j_graph_results.csv
artifacts/eval/graph_comparison_summary.json
```
