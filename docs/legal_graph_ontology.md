# Legal Graph Ontology

The MVP legal graph uses Neo4j as the graph backend and keeps Qdrant as the
primary hybrid retrieval backend.

Every node has:

- `node_id`
- `node_type`
- `name`
- `normalized_name`
- `source_chunk_id`
- provenance properties: `citation_text`, `extraction_method`, `confidence`

`Legal_Document` nodes aggregate chunk provenance in `source_chunk_ids` and
`source_chunk_count` instead of relying on one representative chunk.

Every edge has:

- `edge_id`
- `edge_type`
- `source_chunk_id`
- `extraction_method`
- `confidence`
- provenance properties such as `citation_text`, `normalized_matched_text`,
  `normalized_source_span`, and, when available, `original_matched_text` and
  `raw_source_span`

Graph build summaries include `build_metadata` with the source `index_path`,
manifest hash, manifest build id, record count, and UTC build timestamp.

## Node Types

`Legal_Document`, `Legal_Chapter`, `Legal_Section`, `Legal_Article`,
`Legal_Clause`, `Legal_Point`, `Legal_Rule`, `Subject`, `Legal_Concept`,
`Action`, `Condition`, `Exception`, `Right`, `Obligation`, `Consequence`,
`Sanction`, `Deadline`, `Formula`, `Procedure`, `Evidence_Chunk`.

## Edge Types

`HAS_CHAPTER`, `HAS_SECTION`, `HAS_ARTICLE`, `HAS_CLAUSE`, `HAS_POINT`,
`HAS_SOURCE_CHUNK`, `SOURCE_OF`, `REFERENCES`, `GUIDED_BY`, `DETAILS`,
`AMENDS`, `REPLACES`, `APPLIES_TO`, `MENTIONS_CONCEPT`, `REGULATES_ACTION`,
`HAS_CONDITION`, `HAS_EXCEPTION`, `HAS_DEADLINE`, `HAS_FORMULA`,
`GRANTS_RIGHT`, `IMPOSES_OBLIGATION`, `TRIGGERS_CONSEQUENCE`, `PROHIBITS`,
`PERMITS`, `REQUIRES`.

## MVP Extraction

The graph builder does not use LLM extraction. It creates graph evidence from:

- indexed record metadata for document, article, clause, point and chunk nodes
- dictionary/rule concept linking
- regex cross-reference parsing

This keeps provenance deterministic and makes the graph auditable for thesis
experiments.
