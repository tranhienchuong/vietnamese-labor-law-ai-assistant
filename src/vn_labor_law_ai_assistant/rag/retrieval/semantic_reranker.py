from __future__ import annotations

from typing import Sequence

from ...indexing import require_cross_encoder
from .constants import DEFAULT_RERANKER_TOP_N, RRF_K
from .models import RetrievedRecord, SearchHit


class SemanticReranker:
    def __init__(
        self,
        *,
        model_name: str | None,
        device: str,
        top_n: int = DEFAULT_RERANKER_TOP_N,
    ) -> None:
        self.model_name = str(model_name or "").strip()
        self.device = device
        self.top_n = max(1, int(top_n))
        self._reranker = None

    @property
    def enabled(self) -> bool:
        return bool(self.model_name)

    def get_reranker(self):
        if not self.enabled:
            return None
        if self._reranker is None:
            cross_encoder_cls = require_cross_encoder()
            self._reranker = cross_encoder_cls(self.model_name, device=self.device)
        return self._reranker

    def predict_reranker_scores(
        self,
        query: str,
        hits: Sequence[SearchHit],
        direct_records: dict[str, RetrievedRecord],
    ) -> dict[str, float]:
        reranker = self.get_reranker()
        if reranker is None:
            return {}

        hit_records: list[tuple[SearchHit, RetrievedRecord]] = []
        for hit in hits[: self.top_n]:
            record = direct_records.get(hit.chunk_id)
            if record is None:
                continue
            hit_records.append((hit, record))

        if not hit_records:
            return {}

        pairs = [
            (
                query,
                record.dense_text.strip() or f"{record.citation_text}\n{record.text}".strip(),
            )
            for _, record in hit_records
        ]
        raw_scores = reranker.predict(pairs, show_progress_bar=False)
        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()

        def coerce_score(value: object) -> float:
            if isinstance(value, (list, tuple)):
                return float(value[0])
            if hasattr(value, "item"):
                return float(value.item())
            return float(value)

        return {
            hit.chunk_id: coerce_score(score)
            for (hit, _), score in zip(hit_records, raw_scores)
        }

    def semantic_rerank_hits(
        self,
        query: str,
        hits: Sequence[SearchHit],
        direct_records: dict[str, RetrievedRecord],
    ) -> tuple[SearchHit, ...]:
        if not self.enabled:
            return tuple(hits)

        candidate_hits = tuple(hits[: self.top_n])
        if not candidate_hits:
            return tuple(hits)

        semantic_scores = self.predict_reranker_scores(query, candidate_hits, direct_records)
        if not semantic_scores:
            return tuple(hits)

        heuristic_rank = {hit.chunk_id: rank for rank, hit in enumerate(candidate_hits, start=1)}
        semantic_rank = {
            hit.chunk_id: rank
            for rank, hit in enumerate(
                sorted(
                    candidate_hits,
                    key=lambda current_hit: (
                        -semantic_scores.get(current_hit.chunk_id, float("-inf")),
                        -current_hit.score,
                        current_hit.citation_text,
                    ),
                ),
                start=1,
            )
        }

        def fused_rrf_score(hit: SearchHit) -> float:
            return (
                1.0 / (RRF_K + heuristic_rank[hit.chunk_id])
                + 1.0 / (RRF_K + semantic_rank[hit.chunk_id])
            )

        fused_candidates = sorted(
            candidate_hits,
            key=lambda hit: (
                -fused_rrf_score(hit),
                -semantic_scores.get(hit.chunk_id, float("-inf")),
                -hit.score,
                hit.citation_text,
            ),
        )

        max_existing_score = max((hit.score for hit in hits), default=0.0)
        score_step = 1e-4
        reranked_candidates = tuple(
            SearchHit(
                chunk_id=hit.chunk_id,
                qdrant_point_id=hit.qdrant_point_id,
                score=max_existing_score + 1.0 - (rank * score_step),
                citation_text=hit.citation_text,
                payload=hit.payload,
            )
            for rank, hit in enumerate(fused_candidates)
        )
        remainder = tuple(hits[len(candidate_hits) :])
        return reranked_candidates + remainder


__all__ = ["SemanticReranker"]

