from __future__ import annotations

from typing import Protocol, Sequence

from .models import GraphExpansionResult, LegalGraphEdge, LegalGraphNode
from .ontology import EdgeType


class LegalGraphStore(Protocol):
    def setup_schema(self) -> None:
        ...

    def close(self) -> None:
        ...

    def upsert_nodes(self, nodes: Sequence[LegalGraphNode]) -> None:
        ...

    def upsert_edges(self, edges: Sequence[LegalGraphEdge]) -> None:
        ...

    def get_nodes_by_ids(self, node_ids: Sequence[str]) -> tuple[LegalGraphNode, ...]:
        ...

    def get_nodes_by_chunk_ids(self, chunk_ids: Sequence[str]) -> tuple[LegalGraphNode, ...]:
        ...

    def expand_from_chunk_ids(
        self,
        chunk_ids: Sequence[str],
        *,
        depth: int,
        limit: int,
        min_confidence: float,
        edge_types: Sequence[EdgeType],
    ) -> GraphExpansionResult:
        ...

    def get_source_chunk_ids(self, node_ids: Sequence[str]) -> tuple[str, ...]:
        ...


__all__ = ["LegalGraphStore"]
