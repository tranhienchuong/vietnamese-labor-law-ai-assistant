from __future__ import annotations

from dataclasses import dataclass

from ...heuristic_router import QueryIntent


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    qdrant_point_id: str
    score: float
    citation_text: str
    payload: dict[str, object]


@dataclass(frozen=True)
class RetrievedRecord:
    chunk_id: str
    parent_chunk_id: str | None
    citation_text: str
    text: str
    dense_text: str
    sparse_text: str
    payload: dict[str, object]


@dataclass(frozen=True)
class RetrievalContext:
    chunk_id: str
    citation_text: str
    text: str
    payload: dict[str, object]
    score: float
    matched_chunk_ids: tuple[str, ...]
    matched_citations: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    intent: QueryIntent
    hits: tuple[SearchHit, ...]
    contexts: tuple[RetrievalContext, ...]


__all__ = [
    "RetrievedRecord",
    "RetrievalContext",
    "RetrievalResult",
    "SearchHit",
]

