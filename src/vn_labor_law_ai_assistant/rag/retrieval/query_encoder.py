from __future__ import annotations

from typing import Any

from ...embeddings import embed_query_via_http, is_custom_http_embedding_provider
from ...heuristic_router import (
    QueryIntent,
    contains_normalized_phrase,
    filter_specific_actor_labels,
)
from ...indexing import (
    PyViWordSegmenter,
    SparseBM25Encoder,
    extract_legal_hint_tokens,
    require_sentence_transformers,
)
from .constants import RULE_CONFIG


def actor_labels_for_sparse(intent: QueryIntent, rule_config: Any = RULE_CONFIG) -> tuple[str, ...]:
    query_context = rule_config.QUERY_CONTEXT
    if (
        set(intent.query_types).intersection(query_context.get("full_actor_filter_query_types", ()))
        or contains_normalized_phrase(
            intent.normalized_query,
            query_context.get("full_actor_filter_query_phrases", ()),
        )
    ):
        return tuple(intent.actor_filters)
    return filter_specific_actor_labels(intent.actor_filters)


class QueryEncoder:
    def __init__(
        self,
        *,
        dense_model_name: str,
        device: str,
        sparse_encoder: SparseBM25Encoder,
        qdrant_models: Any,
        rule_config: Any = RULE_CONFIG,
        segmenter: PyViWordSegmenter | None = None,
    ) -> None:
        self.dense_model_name = dense_model_name
        self.device = device
        self.sparse_encoder = sparse_encoder
        self.qdrant_models = qdrant_models
        self.rule_config = rule_config
        self.segmenter = segmenter or PyViWordSegmenter()
        self._dense_model = None

    def get_dense_model(self):
        if self._dense_model is None:
            sentence_transformer_cls = require_sentence_transformers()
            self._dense_model = sentence_transformer_cls(self.dense_model_name, device=self.device)
        return self._dense_model

    def encode_dense_query(self, query: str) -> list[float]:
        if is_custom_http_embedding_provider():
            return embed_query_via_http(query)

        model = self.get_dense_model()
        vector = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return vector.tolist()

    def encode_sparse_query(
        self,
        intent: QueryIntent,
        query_text: str | None = None,
    ) -> tuple[list[str], object]:
        sparse_query_text = query_text or "\n".join(
            part for part in (intent.raw_query, *intent.query_expansions) if part
        )
        tokens = self.segmenter.segment(sparse_query_text)
        tokens.extend(extract_legal_hint_tokens(sparse_query_text))
        tokens.extend(f"dieu_{value}" for value in intent.all_article_numbers)
        tokens.extend(f"khoan_{value}" for value in intent.clause_refs)
        tokens.extend(f"diem_{value}" for value in intent.point_refs)
        tokens.extend(f"topic_{value}" for value in intent.topic_filters)
        tokens.extend(f"issue_{value}" for value in intent.issue_filters)
        tokens.extend(f"actor_{value}" for value in actor_labels_for_sparse(intent, self.rule_config))
        tokens.extend(f"qtype_{value}" for value in intent.query_types)
        sparse_query = self.sparse_encoder.encode_query(tokens)
        sparse_vector = self.qdrant_models.SparseVector(
            indices=sparse_query.indices,
            values=sparse_query.values,
        )
        return tokens, sparse_vector


__all__ = [
    "QueryEncoder",
    "actor_labels_for_sparse",
    "embed_query_via_http",
    "is_custom_http_embedding_provider",
]

