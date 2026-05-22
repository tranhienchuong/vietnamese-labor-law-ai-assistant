from __future__ import annotations

from dataclasses import dataclass

from ...core.config import load_repo_env, load_settings


@dataclass(frozen=True)
class LegalGraphConfig:
    enabled: bool = False
    backend: str = "neo4j"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    expansion_depth: int = 2
    max_expanded_chunks: int = 12
    min_confidence: float = 0.60
    complex_query_only: bool = True
    trace: bool = False

    @classmethod
    def from_env(cls) -> "LegalGraphConfig":
        load_repo_env()
        settings = load_settings()
        return cls(
            enabled=bool(settings.legal_graph_enabled),
            backend=settings.legal_graph_backend.strip().lower() or "neo4j",
            neo4j_uri=settings.neo4j_uri.strip() or "bolt://localhost:7687",
            neo4j_user=settings.neo4j_user.strip() or "neo4j",
            neo4j_password=settings.optional_secret_value(settings.neo4j_password) or "password",
            neo4j_database=settings.neo4j_database.strip() or "neo4j",
            expansion_depth=max(1, int(settings.legal_graph_expansion_depth)),
            max_expanded_chunks=max(0, int(settings.legal_graph_max_expanded_chunks)),
            min_confidence=max(0.0, min(1.0, float(settings.legal_graph_min_confidence))),
            complex_query_only=bool(settings.legal_graph_complex_query_only),
            trace=bool(settings.legal_graph_trace),
        )


__all__ = ["LegalGraphConfig"]
