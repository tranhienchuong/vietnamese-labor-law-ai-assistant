from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ontology import EdgeType, NodeType


@dataclass(frozen=True)
class LegalGraphNode:
    node_id: str
    node_type: NodeType
    name: str
    normalized_name: str
    properties: dict[str, Any] = field(default_factory=dict)
    source_chunk_id: str = ""


@dataclass(frozen=True)
class LegalGraphEdge:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    confidence: float
    source_chunk_id: str
    extraction_method: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphExpansionResult:
    seed_chunk_ids: tuple[str, ...]
    expanded_chunk_ids: tuple[str, ...]
    paths: tuple[dict[str, Any], ...] = ()


__all__ = [
    "GraphExpansionResult",
    "LegalGraphEdge",
    "LegalGraphNode",
]
