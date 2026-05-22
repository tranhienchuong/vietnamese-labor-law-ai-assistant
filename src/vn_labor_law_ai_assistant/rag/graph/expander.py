from __future__ import annotations

import logging
from typing import Sequence

from ...corpus_pipeline import normalize_for_matching
from ...heuristic_router import QueryIntent, dedupe_preserve_order
from ...indexing import make_qdrant_point_id
from ..retrieval.models import RetrievedRecord, SearchHit
from .config import LegalGraphConfig
from .models import GraphExpansionResult
from .ontology import GRAPH_EXPANSION_EDGE_TYPES
from .store import LegalGraphStore


logger = logging.getLogger(__name__)


RELATION_WEIGHTS: dict[str, float] = {
    "REFERENCES": 0.18,
    "GUIDED_BY": 0.14,
    "MENTIONS_CONCEPT": 0.10,
    "APPLIES_TO": 0.08,
    "HAS_SOURCE_CHUNK": 0.06,
    "SOURCE_OF": 0.06,
    "HAS_ARTICLE": 0.04,
    "HAS_CLAUSE": 0.04,
    "HAS_POINT": 0.04,
}


NATURAL_GRAPH_QUERY_HINTS: tuple[str, ...] = tuple(
    normalize_for_matching(value)
    for value in (
        "khi nào",
        "điều kiện",
        "trường hợp",
        "ngoại lệ",
        "có được không",
        "bồi thường",
        "trợ cấp",
        "đơn phương chấm dứt",
        "không cần báo trước",
        "cho nghỉ việc",
        "sa thải",
        "người lao động được gì",
        "công ty phải làm gì",
    )
)
MULTI_HOP_QUERY_HINTS: tuple[str, ...] = tuple(
    normalize_for_matching(value)
    for value in (
        "liên quan",
        "theo nghị định",
        "hướng dẫn",
        "quy định chi tiết",
        "trừ trường hợp",
        "ngoại lệ",
        "điều kiện",
        "hậu quả",
        "bồi thường",
        "trợ cấp",
        "so sánh",
        "áp dụng như thế nào",
    )
)


def dedupe_search_hits(hits: Sequence[SearchHit]) -> tuple[SearchHit, ...]:
    seen: set[str] = set()
    ordered: list[SearchHit] = []
    for hit in hits:
        if hit.chunk_id in seen:
            continue
        seen.add(hit.chunk_id)
        ordered.append(hit)
    return tuple(ordered)


class Neo4jLegalGraphExpander:
    def __init__(
        self,
        *,
        store: LegalGraphStore,
        config: LegalGraphConfig,
    ) -> None:
        self.store = store
        self.config = config

    @staticmethod
    def _contains_hint(normalized_query: str, hints: Sequence[str]) -> bool:
        return any(hint and hint in normalized_query for hint in hints)

    def _has_natural_graph_trigger(self, intent: QueryIntent) -> bool:
        return self._contains_hint(intent.normalized_query, NATURAL_GRAPH_QUERY_HINTS)

    def _is_multi_hop_query(self, intent: QueryIntent) -> bool:
        return bool(
            len(intent.all_article_numbers) > 1
            or len(intent.document_filters) > 1
            or self._contains_hint(intent.normalized_query, MULTI_HOP_QUERY_HINTS)
            or (
                self._has_natural_graph_trigger(intent)
                and (intent.issue_filters or intent.topic_filters)
            )
        )

    def _expansion_depth(self, intent: QueryIntent) -> int:
        configured_depth = max(1, min(4, int(self.config.expansion_depth)))
        if not self._is_multi_hop_query(intent):
            return min(configured_depth, 2)
        if len(intent.all_article_numbers) > 1 and self._has_natural_graph_trigger(intent):
            return max(configured_depth, 4)
        return max(configured_depth, 3)

    def _should_expand(self, intent: QueryIntent) -> bool:
        if self.config.max_expanded_chunks <= 0:
            return False
        if not self.config.complex_query_only:
            return True
        return bool(
            len(intent.all_article_numbers) > 1
            or intent.clause_refs
            or intent.point_refs
            or intent.issue_filters
            or intent.topic_filters
            or intent.forced_references
            or self._has_natural_graph_trigger(intent)
        )

    @staticmethod
    def _score_graph_hit(
        *,
        seed_scores: Sequence[float],
        path: dict[str, object],
    ) -> float:
        min_seed_score = min(seed_scores) if seed_scores else 0.0
        max_seed_score = max(seed_scores) if seed_scores else 0.0
        depth = max(1, int(path.get("graph_depth") or 1))
        confidence = float(path.get("graph_confidence") or 0.75)
        edge_path = tuple(str(value) for value in path.get("graph_edge_path") or [])
        relation_weight = max((RELATION_WEIGHTS.get(edge, 0.03) for edge in edge_path), default=0.03)
        depth_decay = 1.0 / (1.0 + (0.35 * (depth - 1)))
        graph_score = (min_seed_score * 0.82) + (relation_weight * depth_decay) + (0.04 * confidence)
        if max_seed_score > 0:
            graph_score = min(graph_score, max_seed_score - 1e-6)
        return max(0.0, graph_score)

    def _result_to_hits(
        self,
        result: GraphExpansionResult,
        *,
        seed_hits: Sequence[SearchHit],
    ) -> tuple[SearchHit, ...]:
        seed_scores = tuple(hit.score for hit in seed_hits)
        paths_by_chunk_id = {
            str(path.get("chunk_id")): path
            for path in result.paths
            if path.get("chunk_id")
        }
        hits: list[SearchHit] = []
        for chunk_id in result.expanded_chunk_ids:
            path = paths_by_chunk_id.get(chunk_id, {})
            payload = {
                "chunk_id": chunk_id,
                "qdrant_point_id": make_qdrant_point_id(chunk_id),
                "retrieval_source": "neo4j_graph_expansion",
                "graph_seed_chunk_ids": list(result.seed_chunk_ids),
                "graph_edge_path": list(path.get("graph_edge_path") or []),
                "graph_node_path": list(path.get("graph_node_path") or []),
                "graph_depth": int(path.get("graph_depth") or 1),
                "graph_confidence": float(path.get("graph_confidence") or 0.0),
            }
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    qdrant_point_id=str(payload["qdrant_point_id"]),
                    score=self._score_graph_hit(seed_scores=seed_scores, path=path),
                    citation_text="",
                    payload=payload,
                )
            )
        return tuple(hits)

    def expand_from_hits(
        self,
        hits: Sequence[SearchHit],
        direct_records: dict[str, RetrievedRecord],
        intent: QueryIntent,
    ) -> tuple[SearchHit, ...]:
        if not hits or not self._should_expand(intent):
            return ()

        seed_chunk_ids = dedupe_preserve_order(
            tuple(hit.chunk_id for hit in hits if hit.chunk_id in direct_records)
        )
        if not seed_chunk_ids:
            return ()

        result = self.store.expand_from_chunk_ids(
            seed_chunk_ids,
            depth=self._expansion_depth(intent),
            limit=self.config.max_expanded_chunks,
            min_confidence=self.config.min_confidence,
            edge_types=GRAPH_EXPANSION_EDGE_TYPES,
        )
        if self.config.trace:
            logger.info(
                "Neo4j graph expansion: seeds=%s expanded=%s",
                result.seed_chunk_ids,
                result.expanded_chunk_ids,
            )
        seed_hit_lookup = {hit.chunk_id: hit for hit in hits}
        seed_hits = tuple(
            seed_hit_lookup[chunk_id] for chunk_id in seed_chunk_ids if chunk_id in seed_hit_lookup
        )
        return self._result_to_hits(result, seed_hits=seed_hits)


__all__ = [
    "Neo4jLegalGraphExpander",
    "dedupe_search_hits",
]
