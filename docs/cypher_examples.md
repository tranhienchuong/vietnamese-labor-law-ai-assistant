# Neo4j Cypher Examples

## Show Legal Hierarchy

```cypher
MATCH path = (d:Legal_Document)-[:HAS_ARTICLE]->(a:Legal_Article)-[:HAS_CLAUSE]->(c:Legal_Clause)
RETURN path
LIMIT 30;
```

## Show Article To Concept

```cypher
MATCH path = (a:Legal_Article)-[:MENTIONS_CONCEPT]->(concept:Legal_Concept)
RETURN path
LIMIT 30;
```

## Show References

```cypher
MATCH path = (u1:LegalNode)-[:REFERENCES]->(u2:LegalNode)
RETURN path
LIMIT 30;
```

## Show Expansion From Chunk

```cypher
MATCH path = (seed:Evidence_Chunk {chunk_id: $chunk_id})-[*1..2]-(n:LegalNode)
RETURN path
LIMIT 30;
```

## Count Node Types

```cypher
MATCH (n:LegalNode)
RETURN n.node_type AS node_type, count(*) AS count
ORDER BY node_type;
```

## Count Edge Types

```cypher
MATCH ()-[r]->()
RETURN type(r) AS edge_type, count(*) AS count
ORDER BY edge_type;
```
