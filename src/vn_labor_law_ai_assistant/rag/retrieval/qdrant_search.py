from __future__ import annotations

from typing import Any, Sequence

from ...heuristic_router import QueryIntent, build_query_variants, prioritize_issue_filters
from ...indexing import make_qdrant_point_id
from .models import SearchHit
from .query_encoder import QueryEncoder


class QdrantSearcher:
    def __init__(
        self,
        *,
        qdrant_client: Any | None,
        qdrant_models: Any,
        collection_name: str = "",
        dense_vector_name: str = "",
        sparse_vector_name: str = "",
        query_encoder: QueryEncoder | None = None,
        rule_config: Any | None = None,
    ) -> None:
        self.qdrant = qdrant_client
        self.models = qdrant_models
        self.collection_name = collection_name
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self.query_encoder = query_encoder
        self.rule_config = rule_config

    def build_query_filter(self, intent: QueryIntent):
        must_conditions: list[object] = []
        models = self.models

        if intent.document_filters:
            must_conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=list(intent.document_filters)),
                )
            )

        for field_name, values in intent.explicit_legal_reference_filters:
            must_conditions.append(
                models.FieldCondition(
                    key=field_name,
                    match=models.MatchAny(any=list(values)),
                )
            )

        return models.Filter(must=must_conditions) if must_conditions else None

    def build_reference_boost_filter(self, intent: QueryIntent):
        if not intent.legal_reference_filters:
            return None

        models = self.models
        must_conditions: list[object] = []

        if intent.document_filters:
            must_conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=list(intent.document_filters)),
                )
            )

        for field_name, values in intent.legal_reference_filters:
            must_conditions.append(
                models.FieldCondition(
                    key=field_name,
                    match=models.MatchAny(any=list(values)),
                )
            )

        return models.Filter(must=must_conditions)

    def build_issue_focus_filter(self, intent: QueryIntent):
        prioritized_issue_filters = prioritize_issue_filters(intent.issue_filters)
        if not prioritized_issue_filters:
            return None

        models = self.models
        must_conditions: list[object] = []

        if intent.document_filters:
            must_conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=list(intent.document_filters)),
                )
            )

        must_conditions.append(
            models.FieldCondition(
                key="issue_type",
                match=models.MatchAny(any=list(prioritized_issue_filters)),
            )
        )
        return models.Filter(must=must_conditions)

    def search(
        self,
        *,
        intent: QueryIntent,
        top_k: int,
        prefetch_limit: int,
        query_variants: Sequence[str] | None = None,
    ) -> tuple[SearchHit, ...]:
        if self.qdrant is None or self.query_encoder is None:
            raise RuntimeError("QdrantSearcher is not fully configured.")

        query_filter = self.build_query_filter(intent)
        reference_boost_filter = self.build_reference_boost_filter(intent)
        issue_focus_filter = self.build_issue_focus_filter(intent)
        variants = tuple(query_variants or build_query_variants(intent))
        _, sparse_query = self.query_encoder.encode_sparse_query(intent)
        models = self.models

        prefetches = []
        for variant in variants:
            dense_query = self.query_encoder.encode_dense_query(variant)
            _, variant_sparse_query = self.query_encoder.encode_sparse_query(intent, query_text=variant)
            prefetches.extend(
                [
                    models.Prefetch(
                        query=dense_query,
                        using=self.dense_vector_name,
                        filter=query_filter,
                        limit=prefetch_limit,
                    ),
                    models.Prefetch(
                        query=variant_sparse_query,
                        using=self.sparse_vector_name,
                        filter=query_filter,
                        limit=prefetch_limit,
                    ),
                ]
            )

        if reference_boost_filter is not None:
            prefetches.append(
                models.Prefetch(
                    query=sparse_query,
                    using=self.sparse_vector_name,
                    filter=reference_boost_filter,
                    limit=max(8, prefetch_limit // 2),
                )
            )
        if issue_focus_filter is not None:
            prefetches.append(
                models.Prefetch(
                    query=sparse_query,
                    using=self.sparse_vector_name,
                    filter=issue_focus_filter,
                    limit=max(12, prefetch_limit),
                )
            )

        candidate_limit = max(top_k * 6, prefetch_limit * max(4, len(variants) * 2), 64)

        response = self.qdrant.query_points(
            collection_name=self.collection_name,
            prefetch=prefetches,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=candidate_limit,
            with_payload=True,
        )

        return tuple(
            SearchHit(
                chunk_id=str(point.payload["chunk_id"]),
                qdrant_point_id=str(
                    point.payload.get("qdrant_point_id")
                    or make_qdrant_point_id(str(point.payload["chunk_id"]))
                ),
                score=float(point.score),
                citation_text=str(point.payload.get("citation_text") or ""),
                payload=dict(point.payload),
            )
            for point in response.points
        )


__all__ = ["QdrantSearcher"]

