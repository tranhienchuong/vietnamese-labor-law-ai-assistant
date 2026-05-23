from __future__ import annotations

import json
import logging
from typing import Any, Sequence

from ...heuristic_router import dedupe_preserve_order
from .models import GraphExpansionResult, LegalGraphEdge, LegalGraphNode
from .ontology import EdgeType, NodeType, edge_type_label, node_type_label


logger = logging.getLogger(__name__)


def _coerce_neo4j_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_coerce_neo4j_value(item) for item in value]
    return json.dumps(value, ensure_ascii=False)


def _coerce_properties(properties: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _coerce_neo4j_value(value) for key, value in properties.items()}


def _batched(values: Sequence[Any], batch_size: int = 500) -> Sequence[Sequence[Any]]:
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


class Neo4jLegalGraphStore:
    def __init__(
        self,
        *,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        verify_connectivity: bool = True,
        driver: Any | None = None,
    ) -> None:
        self.database = database
        self._driver = driver or self._build_driver(uri=uri, user=user, password=password)
        if verify_connectivity and driver is None:
            try:
                self._driver.verify_connectivity()
            except Exception as exc:
                raise RuntimeError(
                    "Neo4j legal graph is enabled but Neo4j is unavailable. "
                    f"Check NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD. Original error: {exc}"
                ) from exc

    @staticmethod
    def _build_driver(*, uri: str, user: str, password: str) -> Any:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError(
                "The 'neo4j' package is required when LEGAL_GRAPH_ENABLED=true."
            ) from exc
        return GraphDatabase.driver(uri, auth=(user, password))

    def _session(self):
        return self._driver.session(database=self.database)

    def setup_schema(self) -> None:
        statements = (
            "CREATE CONSTRAINT legal_node_node_id IF NOT EXISTS "
            "FOR (n:LegalNode) REQUIRE n.node_id IS UNIQUE",
            "CREATE INDEX legal_node_node_type IF NOT EXISTS "
            "FOR (n:LegalNode) ON (n.node_type)",
            "CREATE INDEX evidence_chunk_chunk_id IF NOT EXISTS "
            "FOR (n:Evidence_Chunk) ON (n.chunk_id)",
            "CREATE INDEX legal_node_source_chunk_id IF NOT EXISTS "
            "FOR (n:LegalNode) ON (n.source_chunk_id)",
            "CREATE INDEX legal_node_normalized_name IF NOT EXISTS "
            "FOR (n:LegalNode) ON (n.normalized_name)",
        )
        with self._session() as session:
            for statement in statements:
                session.run(statement)

    def close(self) -> None:
        self._driver.close()

    def reset_graph(self) -> None:
        with self._session() as session:
            session.run("MATCH (n:LegalNode) DETACH DELETE n")

    def upsert_nodes(self, nodes: Sequence[LegalGraphNode]) -> None:
        if not nodes:
            return
        grouped_rows: dict[str, list[dict[str, Any]]] = {}
        for node in nodes:
            label = node_type_label(node.node_type)
            properties = _coerce_properties(
                {
                    **node.properties,
                    "node_id": node.node_id,
                    "node_type": node.node_type.value,
                    "name": node.name,
                    "normalized_name": node.normalized_name,
                    "source_chunk_id": node.source_chunk_id,
                }
            )
            grouped_rows.setdefault(label, []).append(
                {
                    "node_id": node.node_id,
                    "properties": properties,
                }
            )

        with self._session() as session:
            for label, rows in grouped_rows.items():
                for batch in _batched(rows):
                    session.run(
                        f"""
                        UNWIND $rows AS row
                        MERGE (n:LegalNode {{node_id: row.node_id}})
                        SET n += row.properties
                        SET n:{label}
                        """,
                        rows=list(batch),
                    )

    def upsert_edges(self, edges: Sequence[LegalGraphEdge]) -> None:
        if not edges:
            return
        grouped_rows: dict[str, list[dict[str, Any]]] = {}
        for edge in edges:
            relationship_type = edge_type_label(edge.edge_type)
            properties = _coerce_properties(
                {
                    **edge.properties,
                    "edge_id": edge.edge_id,
                    "edge_type": edge.edge_type.value,
                    "confidence": edge.confidence,
                    "source_chunk_id": edge.source_chunk_id,
                    "extraction_method": edge.extraction_method,
                }
            )
            grouped_rows.setdefault(relationship_type, []).append(
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "edge_id": edge.edge_id,
                    "properties": properties,
                }
            )

        with self._session() as session:
            for relationship_type, rows in grouped_rows.items():
                for batch in _batched(rows):
                    session.run(
                        f"""
                        UNWIND $rows AS row
                        MATCH (source:LegalNode {{node_id: row.source_id}})
                        MATCH (target:LegalNode {{node_id: row.target_id}})
                        MERGE (source)-[r:{relationship_type} {{edge_id: row.edge_id}}]->(target)
                        SET r += row.properties
                        """,
                        rows=list(batch),
                    )

    @staticmethod
    def _node_from_record(record: Any) -> LegalGraphNode | None:
        data = dict(record["n"]) if "n" in record else dict(record)
        node_id = str(data.get("node_id") or "").strip()
        node_type_value = str(data.get("node_type") or "").strip()
        if not node_id or not node_type_value:
            return None
        try:
            node_type = NodeType(node_type_value)
        except ValueError:
            logger.warning("Skipping unsupported legal graph node type: %s", node_type_value)
            return None
        properties = dict(data)
        for key in ("node_id", "node_type", "name", "normalized_name", "source_chunk_id"):
            properties.pop(key, None)
        return LegalGraphNode(
            node_id=node_id,
            node_type=node_type,
            name=str(data.get("name") or ""),
            normalized_name=str(data.get("normalized_name") or ""),
            properties=properties,
            source_chunk_id=str(data.get("source_chunk_id") or ""),
        )

    def get_nodes_by_ids(self, node_ids: Sequence[str]) -> tuple[LegalGraphNode, ...]:
        ordered_ids = dedupe_preserve_order(tuple(str(value) for value in node_ids if value))
        if not ordered_ids:
            return ()
        with self._session() as session:
            result = session.run(
                """
                MATCH (n:LegalNode)
                WHERE n.node_id IN $node_ids
                RETURN n
                """,
                node_ids=list(ordered_ids),
            )
            nodes = [self._node_from_record(record) for record in result]
        return tuple(node for node in nodes if node is not None)

    def get_nodes_by_chunk_ids(self, chunk_ids: Sequence[str]) -> tuple[LegalGraphNode, ...]:
        ordered_ids = dedupe_preserve_order(tuple(str(value) for value in chunk_ids if value))
        if not ordered_ids:
            return ()
        with self._session() as session:
            result = session.run(
                """
                MATCH (n:LegalNode)
                WHERE n.source_chunk_id IN $chunk_ids
                   OR (n:Evidence_Chunk AND n.chunk_id IN $chunk_ids)
                RETURN n
                """,
                chunk_ids=list(ordered_ids),
            )
            nodes = [self._node_from_record(record) for record in result]
        return tuple(node for node in nodes if node is not None)

    def expand_from_chunk_ids(
        self,
        chunk_ids: Sequence[str],
        *,
        depth: int,
        limit: int,
        min_confidence: float,
        edge_types: Sequence[EdgeType],
    ) -> GraphExpansionResult:
        seed_chunk_ids = dedupe_preserve_order(tuple(str(value) for value in chunk_ids if value))
        if not seed_chunk_ids or limit <= 0:
            return GraphExpansionResult(seed_chunk_ids=seed_chunk_ids, expanded_chunk_ids=())

        relationship_depth = max(1, min(4, int(depth)))
        edge_type_values = [edge_type_label(edge_type) for edge_type in edge_types]
        allowed_edge_types = set(edge_type_values)
        if not allowed_edge_types:
            return GraphExpansionResult(seed_chunk_ids=seed_chunk_ids, expanded_chunk_ids=())

        def allows(*values: EdgeType) -> bool:
            return {value.value for value in values}.issubset(allowed_edge_types)

        query_limit = max(max(1, int(limit)) * 4, 20)
        expansion_queries: list[str] = []
        if relationship_depth >= 2 and allows(EdgeType.SOURCE_OF, EdgeType.HAS_SOURCE_CHUNK):
            expansion_queries.append(
                """
                MATCH path = (seed:Evidence_Chunk)-[r1:SOURCE_OF]->(:LegalNode)-[r2:HAS_SOURCE_CHUNK]->(e:Evidence_Chunk)
                WHERE seed.chunk_id IN $seed_chunk_ids
                  AND e.chunk_id IS NOT NULL
                  AND NOT e.chunk_id IN $seed_chunk_ids
                  AND all(rel IN relationships(path) WHERE coalesce(rel.confidence, 1.0) >= $min_confidence)
                RETURN e.chunk_id AS chunk_id,
                       length(path) AS graph_depth,
                       [rel IN relationships(path) | type(rel)] AS graph_edge_path,
                       [node IN nodes(path) | node.node_id] AS graph_node_path,
                       reduce(conf = 1.0, rel IN relationships(path) |
                           conf * coalesce(rel.confidence, 1.0)) AS graph_confidence
                ORDER BY graph_depth ASC, graph_confidence DESC, chunk_id
                LIMIT $query_limit
                """
            )
        if relationship_depth >= 2 and (
            allows(EdgeType.MENTIONS_CONCEPT)
            or allows(EdgeType.APPLIES_TO)
            or allows(EdgeType.REGULATES_ACTION)
        ):
            expansion_queries.append(
                """
                MATCH path = (seed:Evidence_Chunk)-[r1:MENTIONS_CONCEPT|APPLIES_TO|REGULATES_ACTION]->(:LegalNode)<-[r2:MENTIONS_CONCEPT|APPLIES_TO|REGULATES_ACTION]-(e:Evidence_Chunk)
                WHERE seed.chunk_id IN $seed_chunk_ids
                  AND e.chunk_id IS NOT NULL
                  AND NOT e.chunk_id IN $seed_chunk_ids
                  AND all(rel IN relationships(path)
                          WHERE type(rel) IN $edge_types
                            AND coalesce(rel.confidence, 1.0) >= $min_confidence)
                RETURN e.chunk_id AS chunk_id,
                       length(path) AS graph_depth,
                       [rel IN relationships(path) | type(rel)] AS graph_edge_path,
                       [node IN nodes(path) | node.node_id] AS graph_node_path,
                       reduce(conf = 1.0, rel IN relationships(path) |
                           conf * coalesce(rel.confidence, 1.0)) AS graph_confidence
                ORDER BY graph_depth ASC, graph_confidence DESC, chunk_id
                LIMIT $query_limit
                """
            )
        if relationship_depth >= 3 and allows(EdgeType.SOURCE_OF, EdgeType.HAS_SOURCE_CHUNK):
            expansion_queries.append(
                """
                MATCH path = (seed:Evidence_Chunk)-[r1:SOURCE_OF]->(:LegalNode)-[r2:REFERENCES|GUIDED_BY]-(:LegalNode)-[r3:HAS_SOURCE_CHUNK]->(e:Evidence_Chunk)
                WHERE seed.chunk_id IN $seed_chunk_ids
                  AND e.chunk_id IS NOT NULL
                  AND NOT e.chunk_id IN $seed_chunk_ids
                  AND type(r2) IN $edge_types
                  AND all(rel IN relationships(path) WHERE coalesce(rel.confidence, 1.0) >= $min_confidence)
                RETURN e.chunk_id AS chunk_id,
                       length(path) AS graph_depth,
                       [rel IN relationships(path) | type(rel)] AS graph_edge_path,
                       [node IN nodes(path) | node.node_id] AS graph_node_path,
                       reduce(conf = 1.0, rel IN relationships(path) |
                           conf * coalesce(rel.confidence, 1.0)) AS graph_confidence
                ORDER BY graph_depth ASC, graph_confidence DESC, chunk_id
                LIMIT $query_limit
                """
            )
        if relationship_depth >= 4 and allows(EdgeType.SOURCE_OF, EdgeType.HAS_SOURCE_CHUNK):
            expansion_queries.append(
                """
                MATCH path = (seed:Evidence_Chunk)-[r1:SOURCE_OF]->(:LegalNode)-[r2:MENTIONS_CONCEPT|APPLIES_TO|REGULATES_ACTION]->(:LegalNode)<-[r3:MENTIONS_CONCEPT|APPLIES_TO|REGULATES_ACTION]-(target:LegalNode)-[r4:HAS_SOURCE_CHUNK]->(e:Evidence_Chunk)
                WHERE seed.chunk_id IN $seed_chunk_ids
                  AND NOT target:Evidence_Chunk
                  AND e.chunk_id IS NOT NULL
                  AND NOT e.chunk_id IN $seed_chunk_ids
                  AND type(r2) IN $edge_types
                  AND type(r3) IN $edge_types
                  AND all(rel IN relationships(path) WHERE coalesce(rel.confidence, 1.0) >= $min_confidence)
                RETURN e.chunk_id AS chunk_id,
                       length(path) AS graph_depth,
                       [rel IN relationships(path) | type(rel)] AS graph_edge_path,
                       [node IN nodes(path) | node.node_id] AS graph_node_path,
                       reduce(conf = 1.0, rel IN relationships(path) |
                           conf * coalesce(rel.confidence, 1.0)) AS graph_confidence
                ORDER BY graph_depth ASC, graph_confidence DESC, chunk_id
                LIMIT $query_limit
                """
            )

        if not expansion_queries:
            return GraphExpansionResult(seed_chunk_ids=seed_chunk_ids, expanded_chunk_ids=())

        paths_by_chunk_id: dict[str, dict[str, Any]] = {}
        with self._session() as session:
            for query in expansion_queries:
                result = session.run(
                    query,
                    seed_chunk_ids=list(seed_chunk_ids),
                    edge_types=edge_type_values,
                    min_confidence=float(min_confidence),
                    query_limit=query_limit,
                )
                for record in result:
                    path = dict(record)
                    chunk_id = str(path.get("chunk_id") or "")
                    if not chunk_id:
                        continue
                    current = paths_by_chunk_id.get(chunk_id)
                    if current is None:
                        paths_by_chunk_id[chunk_id] = path
                        continue
                    current_key = (
                        int(current.get("graph_depth") or 10_000),
                        -float(current.get("graph_confidence") or 0.0),
                    )
                    next_key = (
                        int(path.get("graph_depth") or 10_000),
                        -float(path.get("graph_confidence") or 0.0),
                    )
                    if next_key < current_key:
                        paths_by_chunk_id[chunk_id] = path

        paths = tuple(
            sorted(
                paths_by_chunk_id.values(),
                key=lambda path: (
                    int(path.get("graph_depth") or 10_000),
                    -float(path.get("graph_confidence") or 0.0),
                    str(path.get("chunk_id") or ""),
                ),
            )[: max(1, int(limit))]
        )

        expanded_chunk_ids = dedupe_preserve_order(
            tuple(str(path.get("chunk_id") or "") for path in paths if path.get("chunk_id"))
        )
        return GraphExpansionResult(
            seed_chunk_ids=seed_chunk_ids,
            expanded_chunk_ids=expanded_chunk_ids,
            paths=paths,
        )

    def get_source_chunk_ids(self, node_ids: Sequence[str]) -> tuple[str, ...]:
        ordered_ids = dedupe_preserve_order(tuple(str(value) for value in node_ids if value))
        if not ordered_ids:
            return ()
        with self._session() as session:
            result = session.run(
                """
                MATCH (n:LegalNode)
                WHERE n.node_id IN $node_ids
                RETURN coalesce(n.chunk_id, n.source_chunk_id) AS chunk_id
                """,
                node_ids=list(ordered_ids),
            )
            chunk_ids = tuple(str(record["chunk_id"]) for record in result if record["chunk_id"])
        return dedupe_preserve_order(chunk_ids)


__all__ = ["Neo4jLegalGraphStore"]
