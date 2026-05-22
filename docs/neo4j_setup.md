# Neo4j Setup

Start Neo4j locally:

```powershell
docker compose -f docker-compose.neo4j.yml up -d
```

Default connection:

```text
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
```

Build the legal graph from the current hybrid index:

```powershell
.venv\Scripts\python.exe scripts\build_legal_graph.py --index-path artifacts/index --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-password password --neo4j-database neo4j
```

Enable graph expansion in the RAG runtime:

```text
LEGAL_GRAPH_ENABLED=true
LEGAL_GRAPH_BACKEND=neo4j
LEGAL_GRAPH_EXPANSION_DEPTH=2
LEGAL_GRAPH_MAX_EXPANDED_CHUNKS=12
LEGAL_GRAPH_MIN_CONFIDENCE=0.60
LEGAL_GRAPH_COMPLEX_QUERY_ONLY=true
```

Open Neo4j Browser at `http://localhost:7474` and use the Cypher examples in
[cypher_examples.md](cypher_examples.md).
