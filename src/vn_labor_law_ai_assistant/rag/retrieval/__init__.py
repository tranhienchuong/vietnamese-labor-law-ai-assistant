"""Retrieval components for the RAG pipeline."""

from .constants import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_RERANKER_TOP_N,
    FORCED_REFERENCE_SCORE_MARGIN,
    POINT_REF_ALPHABET,
    POINT_REF_ORDER,
    RECORD_SOURCE_QDRANT_PAYLOAD,
    RECORD_SOURCE_SQLITE,
    RRF_K,
    RULE_CONFIG,
    SUPPORTED_RECORD_SOURCES,
)
from .context_assembler import ContextAssembler
from .manifest import load_manifest
from .models import RetrievedRecord, RetrievalContext, RetrievalResult, SearchHit
from .qdrant_search import QdrantSearcher
from .query_encoder import QueryEncoder, actor_labels_for_sparse
from .record_store import QdrantPayloadRecordStore, RecordStore, SQLiteRecordStore
from .reference_expander import ReferenceExpander
from .retriever import HybridRetriever
from .scoring import RelevanceScorer
from .semantic_reranker import SemanticReranker
from .utils import (
    build_context_block,
    build_expanded_context_text,
    context_article_key,
    context_looks_like_enumeration_parent,
    dedupe_records_by_chunk_id,
    diversify_contexts_by_article,
    env_flag,
    estimate_token_count,
    extend_context_with_records,
    format_context_for_prompt,
    point_ref_sort_key,
    record_from_qdrant_payload,
    record_reference_sort_key,
    resolve_record_source,
    select_contexts_for_prompt,
)

__all__ = [
    "ContextAssembler",
    "DEFAULT_MAX_CONTEXT_CHARS",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "DEFAULT_RERANKER_TOP_N",
    "FORCED_REFERENCE_SCORE_MARGIN",
    "HybridRetriever",
    "POINT_REF_ALPHABET",
    "POINT_REF_ORDER",
    "QdrantPayloadRecordStore",
    "QdrantSearcher",
    "QueryEncoder",
    "RECORD_SOURCE_QDRANT_PAYLOAD",
    "RECORD_SOURCE_SQLITE",
    "RRF_K",
    "RULE_CONFIG",
    "RecordStore",
    "ReferenceExpander",
    "RelevanceScorer",
    "RetrievedRecord",
    "RetrievalContext",
    "RetrievalResult",
    "SQLiteRecordStore",
    "SUPPORTED_RECORD_SOURCES",
    "SearchHit",
    "SemanticReranker",
    "actor_labels_for_sparse",
    "build_context_block",
    "build_expanded_context_text",
    "context_article_key",
    "context_looks_like_enumeration_parent",
    "dedupe_records_by_chunk_id",
    "diversify_contexts_by_article",
    "env_flag",
    "estimate_token_count",
    "extend_context_with_records",
    "format_context_for_prompt",
    "load_manifest",
    "point_ref_sort_key",
    "record_from_qdrant_payload",
    "record_reference_sort_key",
    "resolve_record_source",
    "select_contexts_for_prompt",
]

