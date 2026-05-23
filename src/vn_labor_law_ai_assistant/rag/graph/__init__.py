"""Neo4j-backed legal graph expansion for the hybrid RAG pipeline."""

from typing import Any

from .config import LegalGraphConfig
from .models import GraphExpansionResult, LegalGraphEdge, LegalGraphNode
from .ontology import EdgeType, NodeType


def __getattr__(name: str) -> Any:
    if name in {"LegalGraphBuildResult", "LegalGraphBuilder"}:
        from .builder import LegalGraphBuildResult, LegalGraphBuilder

        return {
            "LegalGraphBuildResult": LegalGraphBuildResult,
            "LegalGraphBuilder": LegalGraphBuilder,
        }[name]
    if name in {"Neo4jLegalGraphExpander", "dedupe_search_hits"}:
        from .expander import Neo4jLegalGraphExpander, dedupe_search_hits

        return {
            "Neo4jLegalGraphExpander": Neo4jLegalGraphExpander,
            "dedupe_search_hits": dedupe_search_hits,
        }[name]
    if name == "Neo4jLegalGraphStore":
        from .neo4j_store import Neo4jLegalGraphStore

        return Neo4jLegalGraphStore
    raise AttributeError(name)

__all__ = [
    "EdgeType",
    "GraphExpansionResult",
    "LegalGraphBuildResult",
    "LegalGraphBuilder",
    "LegalGraphConfig",
    "LegalGraphEdge",
    "LegalGraphNode",
    "Neo4jLegalGraphExpander",
    "Neo4jLegalGraphStore",
    "NodeType",
    "dedupe_search_hits",
]
