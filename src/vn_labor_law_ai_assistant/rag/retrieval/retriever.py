from __future__ import annotations

from pathlib import Path
import sqlite3
import sys
from typing import Any, Sequence

from ...embeddings import embed_query_via_http, is_custom_http_embedding_provider
from ...heuristic_router import (
    LegalReference,
    QueryIntent,
    build_query_variants,
    contains_normalized_phrase,
    dedupe_preserve_order,
    route_query_heuristic,
)
from ...indexing import (
    PyViWordSegmenter,
    build_qdrant_client,
    load_sparse_encoder,
    make_qdrant_point_id,
    require_cross_encoder,
    require_qdrant,
    require_sentence_transformers,
)
from ...query_router import route_query_with_llm
from ...core.config import load_settings
from .constants import (
    DEFAULT_RERANKER_TOP_N,
    FORCED_REFERENCE_SCORE_MARGIN,
    RECORD_SOURCE_QDRANT_PAYLOAD,
    RECORD_SOURCE_SQLITE,
    RRF_K,
    RULE_CONFIG,
)
from .context_assembler import ContextAssembler
from .manifest import load_manifest
from .models import RetrievedRecord, RetrievalContext, RetrievalResult, SearchHit
from .qdrant_search import QdrantSearcher
from .query_encoder import QueryEncoder
from .record_store import QdrantPayloadRecordStore, RecordStore, SQLiteRecordStore
from .reference_expander import ReferenceExpander
from .scoring import RelevanceScorer
from .semantic_reranker import SemanticReranker
from .utils import (
    build_expanded_context_text,
    context_article_key,
    context_looks_like_enumeration_parent,
    dedupe_records_by_chunk_id,
    env_flag,
    extend_context_with_records,
    record_from_qdrant_payload,
    record_reference_sort_key,
    resolve_record_source,
)


_RULE_CONFIG_EXPORTS = frozenset(("LEGAL_ISSUE_ARTICLE_MAP", "TERMINATION_ARTICLE_MAP"))


def __getattr__(name: str) -> object:
    if name in _RULE_CONFIG_EXPORTS:
        return getattr(RULE_CONFIG, name)
    raise AttributeError(name)


def _compat_symbol(name: str, default: Any) -> Any:
    module = sys.modules.get("vn_labor_law_ai_assistant.retriever")
    return getattr(module, name, default) if module is not None else default


class _HybridRecordStoreAdapter(RecordStore):
    def __init__(self, retriever: "HybridRetriever") -> None:
        self.retriever = retriever

    def fetch_records(self, chunk_ids: Sequence[str]) -> dict[str, RetrievedRecord]:
        return self.retriever._fetch_records(chunk_ids)

    def fetch_records_from_hits(self, hits: Sequence[SearchHit]) -> dict[str, RetrievedRecord]:
        return self.retriever._fetch_records_from_hits(hits)

    def fetch_records_by_reference(
        self,
        *,
        document_ids: Sequence[str] = (),
        article_numbers: Sequence[str] = (),
        clause_refs: Sequence[str] = (),
        point_refs: Sequence[str] = (),
        exclude_chunk_ids: Sequence[str] = (),
        limit: int = 12,
    ) -> tuple[RetrievedRecord, ...]:
        return self.retriever._fetch_records_by_reference(
            document_ids=document_ids,
            article_numbers=article_numbers,
            clause_refs=clause_refs,
            point_refs=point_refs,
            exclude_chunk_ids=exclude_chunk_ids,
            limit=limit,
        )

    def fetch_article_siblings(
        self,
        *,
        document_id: str,
        article_number: str,
        exclude_chunk_ids: Sequence[str] = (),
        limit: int = 6,
    ) -> tuple[RetrievedRecord, ...]:
        return self.retriever._fetch_article_siblings(
            document_id=document_id,
            article_number=article_number,
            exclude_chunk_ids=exclude_chunk_ids,
            limit=limit,
        )


class HybridRetriever:
    def __init__(
        self,
        *,
        index_path: Path = Path("artifacts/index"),
        device: str | None = None,
        reranker_model: str | None = None,
        reranker_top_n: int = DEFAULT_RERANKER_TOP_N,
        query_router_enabled: bool | None = None,
    ) -> None:
        self._manifest = load_manifest(index_path)
        self._device = device or str(self._manifest.get("device") or "cpu")
        self._record_source = resolve_record_source(self._manifest)
        records_db_path = str(self._manifest.get("records_db_path") or "").strip()
        qdrant_path = str(self._manifest.get("qdrant_path") or "").strip()
        self._records_db_path = Path(records_db_path) if records_db_path else None
        self._qdrant_path = Path(qdrant_path) if qdrant_path else None
        settings = load_settings()
        collection_override = (
            settings.qdrant_collection.strip()
            if settings.qdrant_collection_was_configured()
            else ""
        )
        self._collection_name = collection_override or str(self._manifest["collection_name"])
        self._dense_model_name = str(self._manifest["dense_model_name"])
        self._dense_vector_name = str(self._manifest["dense_vector_name"])
        self._sparse_vector_name = str(self._manifest["sparse_vector_name"])
        self._segmenter = PyViWordSegmenter()
        self._sparse_encoder = load_sparse_encoder(Path(str(self._manifest["sparse_encoder_path"])))
        qdrant_client_cls, self._qdrant_models = require_qdrant()
        self._qdrant = build_qdrant_client(qdrant_client_cls, self._qdrant_path)

        self._query_encoder = QueryEncoder(
            dense_model_name=self._dense_model_name,
            device=self._device,
            sparse_encoder=self._sparse_encoder,
            qdrant_models=self._qdrant_models,
            rule_config=RULE_CONFIG,
            segmenter=self._segmenter,
        )
        self._dense_model = None

        if self._record_source == RECORD_SOURCE_SQLITE:
            if self._records_db_path is None:
                raise ValueError("records_db_path is required when record_source is sqlite.")
            self._record_store: RecordStore = SQLiteRecordStore(self._records_db_path)
            self._sqlite = self._record_store.sqlite
        else:
            self._record_store = QdrantPayloadRecordStore(
                qdrant_client=self._qdrant,
                qdrant_models=self._qdrant_models,
                collection_name=self._collection_name,
            )
            self._sqlite = None

        self._qdrant_searcher = QdrantSearcher(
            qdrant_client=self._qdrant,
            qdrant_models=self._qdrant_models,
            collection_name=self._collection_name,
            dense_vector_name=self._dense_vector_name,
            sparse_vector_name=self._sparse_vector_name,
            query_encoder=self._query_encoder,
            rule_config=RULE_CONFIG,
        )
        self._reference_expander = ReferenceExpander(
            record_store=self._record_store,
            rule_config=RULE_CONFIG,
        )
        self._scorer = RelevanceScorer(rule_config=RULE_CONFIG)
        self._reranker_model_name = str(reranker_model or "").strip()
        self._reranker_top_n = max(1, int(reranker_top_n))
        self._semantic_reranker = SemanticReranker(
            model_name=self._reranker_model_name,
            device=self._device,
            top_n=self._reranker_top_n,
        )
        self._reranker = None
        self._context_assembler = ContextAssembler(
            record_store=self._record_store,
            rule_config=RULE_CONFIG,
        )

        self._query_router_enabled = (
            env_flag("QUERY_ROUTER_ENABLED", True)
            if query_router_enabled is None
            else bool(query_router_enabled)
        )
        self._query_router_provider = settings.query_router_provider.strip() or None
        self._query_router_model = settings.query_router_model.strip() or None
        self._query_router_fallback_to_heuristic = env_flag(
            "QUERY_ROUTER_FALLBACK_TO_HEURISTIC",
            True,
        )

    @property
    def manifest(self) -> dict[str, object]:
        return dict(self._manifest)

    @property
    def reranker_model_name(self) -> str:
        return self._reranker_model_name

    @property
    def reranker_enabled(self) -> bool:
        return bool(self._reranker_model_name)

    @property
    def query_router_enabled(self) -> bool:
        return self._query_router_enabled

    def close(self) -> None:
        self._qdrant.close()
        self._record_store.close()

    def _get_dense_model(self):
        if hasattr(self, "_query_encoder"):
            model = self._query_encoder.get_dense_model()
            self._dense_model = model
            return model
        if getattr(self, "_dense_model", None) is None:
            sentence_transformer_cls = require_sentence_transformers()
            self._dense_model = sentence_transformer_cls(self._dense_model_name, device=self._device)
        return self._dense_model

    def _get_reranker(self):
        if not self.reranker_enabled:
            return None
        if hasattr(self, "_semantic_reranker"):
            reranker = self._semantic_reranker.get_reranker()
            self._reranker = reranker
            return reranker
        if getattr(self, "_reranker", None) is None:
            cross_encoder_cls = require_cross_encoder()
            self._reranker = cross_encoder_cls(self._reranker_model_name, device=self._device)
        return self._reranker

    def _route_query(self, query: str) -> QueryIntent:
        if not self._query_router_enabled:
            return route_query_heuristic(query, RULE_CONFIG)

        route_with_llm = _compat_symbol("route_query_with_llm", route_query_with_llm)
        try:
            return route_with_llm(
                query,
                provider=self._query_router_provider,
                model=self._query_router_model,
            )
        except Exception:
            if not self._query_router_fallback_to_heuristic:
                raise
            return route_query_heuristic(query, RULE_CONFIG)

    def _encode_dense_query(self, query: str) -> list[float]:
        is_custom_http = _compat_symbol(
            "is_custom_http_embedding_provider",
            is_custom_http_embedding_provider,
        )
        if is_custom_http():
            embed_http = _compat_symbol("embed_query_via_http", embed_query_via_http)
            return embed_http(query)

        model = self._get_dense_model()
        vector = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return vector.tolist()

    def _encode_sparse_query(
        self,
        intent: QueryIntent,
        query_text: str | None = None,
    ) -> tuple[list[str], object]:
        if hasattr(self, "_query_encoder"):
            return self._query_encoder.encode_sparse_query(intent, query_text=query_text)
        encoder = QueryEncoder(
            dense_model_name=getattr(self, "_dense_model_name", ""),
            device=getattr(self, "_device", "cpu"),
            sparse_encoder=self._sparse_encoder,
            qdrant_models=self._qdrant_models,
            rule_config=RULE_CONFIG,
            segmenter=self._segmenter,
        )
        return encoder.encode_sparse_query(intent, query_text=query_text)

    def _qdrant_searcher_for_compat(self) -> QdrantSearcher:
        if hasattr(self, "_qdrant_searcher"):
            return self._qdrant_searcher
        return QdrantSearcher(
            qdrant_client=getattr(self, "_qdrant", None),
            qdrant_models=self._qdrant_models,
            collection_name=getattr(self, "_collection_name", ""),
            dense_vector_name=getattr(self, "_dense_vector_name", ""),
            sparse_vector_name=getattr(self, "_sparse_vector_name", ""),
            query_encoder=getattr(self, "_query_encoder", None),
            rule_config=RULE_CONFIG,
        )

    def _build_query_filter(self, intent: QueryIntent):
        return self._qdrant_searcher_for_compat().build_query_filter(intent)

    def _build_reference_boost_filter(self, intent: QueryIntent):
        return self._qdrant_searcher_for_compat().build_reference_boost_filter(intent)

    def _build_issue_focus_filter(self, intent: QueryIntent):
        return self._qdrant_searcher_for_compat().build_issue_focus_filter(intent)

    def _records_from_rows(self, rows: Sequence[sqlite3.Row]) -> dict[str, RetrievedRecord]:
        return SQLiteRecordStore.records_from_rows(rows)

    def _uses_qdrant_payload_records(self) -> bool:
        return (
            getattr(self, "_record_source", RECORD_SOURCE_SQLITE)
            == RECORD_SOURCE_QDRANT_PAYLOAD
        )

    def _records_from_qdrant_points(self, points: Sequence[object]) -> dict[str, RetrievedRecord]:
        return QdrantPayloadRecordStore.records_from_qdrant_points(points)

    def _fetch_records_from_qdrant_ids(self, chunk_ids: Sequence[str]) -> dict[str, RetrievedRecord]:
        ordered_ids = dedupe_preserve_order(chunk_ids)
        if not ordered_ids:
            return {}

        points = self._qdrant.retrieve(
            collection_name=self._collection_name,
            ids=[make_qdrant_point_id(chunk_id) for chunk_id in ordered_ids],
            with_payload=True,
            with_vectors=False,
        )
        return self._records_from_qdrant_points(points)

    def _fetch_records_from_hits(self, hits: Sequence[SearchHit]) -> dict[str, RetrievedRecord]:
        if hasattr(self, "_record_store"):
            return self._record_store.fetch_records_from_hits(hits)

        if not self._uses_qdrant_payload_records():
            return self._fetch_records([hit.chunk_id for hit in hits])

        records: dict[str, RetrievedRecord] = {}
        missing_chunk_ids: list[str] = []
        for hit in hits:
            try:
                record = record_from_qdrant_payload(hit.payload)
            except ValueError:
                missing_chunk_ids.append(hit.chunk_id)
                continue
            records[record.chunk_id] = record

        if missing_chunk_ids:
            records.update(self._fetch_records_from_qdrant_ids(missing_chunk_ids))
        return records

    def _fetch_records(self, chunk_ids: Sequence[str]) -> dict[str, RetrievedRecord]:
        if hasattr(self, "_record_store"):
            return self._record_store.fetch_records(chunk_ids)

        ordered_ids = dedupe_preserve_order(chunk_ids)
        if not ordered_ids:
            return {}

        if self._uses_qdrant_payload_records():
            return self._fetch_records_from_qdrant_ids(ordered_ids)

        if getattr(self, "_sqlite", None) is None:
            raise RuntimeError("SQLite record store is not open.")

        placeholders = ", ".join("?" for _ in ordered_ids)
        rows = self._sqlite.execute(
            f"""
            SELECT chunk_id, parent_chunk_id, citation_text, text, dense_text, sparse_text, payload_json
            FROM records
            WHERE chunk_id IN ({placeholders})
            """,
            ordered_ids,
        ).fetchall()

        return self._records_from_rows(rows)

    def _qdrant_payload_store_for_compat(self) -> QdrantPayloadRecordStore:
        return QdrantPayloadRecordStore(
            qdrant_client=getattr(self, "_qdrant", None),
            qdrant_models=self._qdrant_models,
            collection_name=getattr(self, "_collection_name", ""),
        )

    def _build_reference_payload_filter(
        self,
        *,
        document_ids: Sequence[str] = (),
        article_numbers: Sequence[str] = (),
        clause_refs: Sequence[str] = (),
        point_refs: Sequence[str] = (),
        exclude_chunk_ids: Sequence[str] = (),
    ):
        if isinstance(getattr(self, "_record_store", None), QdrantPayloadRecordStore):
            return self._record_store.build_reference_payload_filter(
                document_ids=document_ids,
                article_numbers=article_numbers,
                clause_refs=clause_refs,
                point_refs=point_refs,
                exclude_chunk_ids=exclude_chunk_ids,
            )
        return self._qdrant_payload_store_for_compat().build_reference_payload_filter(
            document_ids=document_ids,
            article_numbers=article_numbers,
            clause_refs=clause_refs,
            point_refs=point_refs,
            exclude_chunk_ids=exclude_chunk_ids,
        )

    def _fetch_records_by_reference_from_qdrant(
        self,
        *,
        document_ids: Sequence[str] = (),
        article_numbers: Sequence[str] = (),
        clause_refs: Sequence[str] = (),
        point_refs: Sequence[str] = (),
        exclude_chunk_ids: Sequence[str] = (),
        limit: int = 12,
    ) -> tuple[RetrievedRecord, ...]:
        return self._qdrant_payload_store_for_compat().fetch_records_by_reference(
            document_ids=document_ids,
            article_numbers=article_numbers,
            clause_refs=clause_refs,
            point_refs=point_refs,
            exclude_chunk_ids=exclude_chunk_ids,
            limit=limit,
        )

    def _fetch_records_by_reference(
        self,
        *,
        document_ids: Sequence[str] = (),
        article_numbers: Sequence[str] = (),
        clause_refs: Sequence[str] = (),
        point_refs: Sequence[str] = (),
        exclude_chunk_ids: Sequence[str] = (),
        limit: int = 12,
    ) -> tuple[RetrievedRecord, ...]:
        if hasattr(self, "_record_store"):
            return self._record_store.fetch_records_by_reference(
                document_ids=document_ids,
                article_numbers=article_numbers,
                clause_refs=clause_refs,
                point_refs=point_refs,
                exclude_chunk_ids=exclude_chunk_ids,
                limit=limit,
            )

        if self._uses_qdrant_payload_records():
            return self._fetch_records_by_reference_from_qdrant(
                document_ids=document_ids,
                article_numbers=article_numbers,
                clause_refs=clause_refs,
                point_refs=point_refs,
                exclude_chunk_ids=exclude_chunk_ids,
                limit=limit,
            )

        if getattr(self, "_sqlite", None) is None:
            raise RuntimeError("SQLite record store is not open.")

        where_parts: list[str] = []
        params: list[object] = []

        def add_in_filter(field_name: str, values: Sequence[str]) -> None:
            ordered_values = dedupe_preserve_order(tuple(value for value in values if value))
            if not ordered_values:
                return
            placeholders = ", ".join("?" for _ in ordered_values)
            where_parts.append(f"{field_name} IN ({placeholders})")
            params.extend(ordered_values)

        add_in_filter("document_id", document_ids)
        add_in_filter("article_number", article_numbers)
        add_in_filter("clause_ref", clause_refs)
        ordered_point_refs = dedupe_preserve_order(tuple(value for value in point_refs if value))
        if ordered_point_refs:
            placeholders = ", ".join("?" for _ in ordered_point_refs)
            point_conditions = [f"point_ref IN ({placeholders})"]
            params.extend(ordered_point_refs)
            for point_ref in ordered_point_refs:
                point_conditions.append("point_refs LIKE ?")
                params.append(f"%|{point_ref}|%")
            where_parts.append("(" + " OR ".join(point_conditions) + ")")

        excluded_ids = dedupe_preserve_order(tuple(value for value in exclude_chunk_ids if value))
        if excluded_ids:
            placeholders = ", ".join("?" for _ in excluded_ids)
            where_parts.append(f"chunk_id NOT IN ({placeholders})")
            params.extend(excluded_ids)

        if not where_parts:
            return ()

        params.append(max(1, int(limit)))
        rows = self._sqlite.execute(
            f"""
            SELECT chunk_id, parent_chunk_id, citation_text, text, dense_text, sparse_text, payload_json
            FROM records
            WHERE {" AND ".join(where_parts)}
            ORDER BY
                CASE level
                    WHEN 'article' THEN 0
                    WHEN 'clause' THEN 1
                    WHEN 'point' THEN 2
                    ELSE 3
                END,
                CAST(NULLIF(clause_ref, '') AS INTEGER),
                clause_ref,
                point_ref,
                point_refs,
                chunk_id
            LIMIT ?
            """,
            params,
        ).fetchall()
        return tuple(self._records_from_rows(rows).values())

    def _fetch_article_siblings(
        self,
        *,
        document_id: str,
        article_number: str,
        exclude_chunk_ids: Sequence[str] = (),
        limit: int = 6,
    ) -> tuple[RetrievedRecord, ...]:
        if hasattr(self, "_record_store"):
            return self._record_store.fetch_article_siblings(
                document_id=document_id,
                article_number=article_number,
                exclude_chunk_ids=exclude_chunk_ids,
                limit=limit,
            )
        return self._fetch_records_by_reference(
            document_ids=(document_id,),
            article_numbers=(article_number,),
            exclude_chunk_ids=exclude_chunk_ids,
            limit=limit,
        )

    @staticmethod
    def _record_to_search_hit(record: RetrievedRecord, score: float) -> SearchHit:
        return ReferenceExpander.record_to_search_hit(record, score)

    @staticmethod
    def _reference_label(reference: LegalReference) -> str:
        return ReferenceExpander.reference_label(reference)

    @staticmethod
    def _payload_matches_reference(payload: dict[str, object], reference: LegalReference) -> bool:
        return ReferenceExpander.payload_matches_reference(payload, reference)

    def _reference_expander_for_compat(self) -> ReferenceExpander:
        if hasattr(self, "_reference_expander"):
            return self._reference_expander
        return ReferenceExpander(
            record_store=_HybridRecordStoreAdapter(self),
            rule_config=RULE_CONFIG,
        )

    def _hit_matches_forced_reference(self, hit: SearchHit, intent: QueryIntent) -> bool:
        return self._reference_expander_for_compat().hit_matches_forced_reference(hit, intent)

    def _forced_reference_records(
        self,
        intent: QueryIntent,
        *,
        limit: int,
    ) -> tuple[tuple[LegalReference, RetrievedRecord], ...]:
        return self._reference_expander_for_compat().forced_reference_records(intent, limit=limit)

    def _append_forced_reference_hits(
        self,
        hits: Sequence[SearchHit],
        intent: QueryIntent,
        *,
        limit: int,
    ) -> tuple[SearchHit, ...]:
        return self._reference_expander_for_compat().append_forced_reference_hits(
            hits,
            intent,
            limit=limit,
        )

    def _append_reference_fallback_hits(
        self,
        hits: Sequence[SearchHit],
        intent: QueryIntent,
        *,
        limit: int,
    ) -> tuple[SearchHit, ...]:
        return self._reference_expander_for_compat().append_reference_fallback_hits(
            hits,
            intent,
            limit=limit,
        )

    def _pin_forced_reference_hits(
        self,
        hits: Sequence[SearchHit],
        intent: QueryIntent,
    ) -> tuple[SearchHit, ...]:
        return self._reference_expander_for_compat().pin_forced_reference_hits(hits, intent)

    @staticmethod
    def _boost_values(value: object) -> tuple[str, ...]:
        return RelevanceScorer.boost_values(value)

    @staticmethod
    def _hint_values(name: object) -> tuple[str, ...]:
        return RelevanceScorer(rule_config=RULE_CONFIG).hint_values(name)

    @staticmethod
    def _has_any(text: str, phrases: object) -> bool:
        return RelevanceScorer(rule_config=RULE_CONFIG).has_any(text, phrases)

    @staticmethod
    def _has_all(text: str, phrases: object) -> bool:
        return RelevanceScorer(rule_config=RULE_CONFIG).has_all(text, phrases)

    def _scorer_for_compat(self) -> RelevanceScorer:
        if hasattr(self, "_scorer"):
            return self._scorer
        return RelevanceScorer(rule_config=RULE_CONFIG)

    def _boost_flags(self, intent: QueryIntent) -> dict[str, bool]:
        return self._scorer_for_compat().boost_flags(intent)

    def _boost_condition_matches(
        self,
        condition: dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        return self._scorer_for_compat().boost_condition_matches(condition, **kwargs)

    def _score_hit_relevance(
        self,
        hit: SearchHit,
        record: RetrievedRecord,
        intent: QueryIntent,
    ) -> float:
        return self._scorer_for_compat().score_hit_relevance(hit, record, intent)

    def _rerank_hits(
        self,
        hits: Sequence[SearchHit],
        intent: QueryIntent,
        direct_records: dict[str, RetrievedRecord],
    ) -> tuple[SearchHit, ...]:
        return self._scorer_for_compat().rerank_hits(hits, intent, direct_records)

    def _predict_reranker_scores(
        self,
        query: str,
        hits: Sequence[SearchHit],
        direct_records: dict[str, RetrievedRecord],
    ) -> dict[str, float]:
        if hasattr(self, "_semantic_reranker"):
            return self._semantic_reranker.predict_reranker_scores(query, hits, direct_records)
        reranker = self._get_reranker()
        if reranker is None:
            return {}

        top_n = getattr(self, "_reranker_top_n", DEFAULT_RERANKER_TOP_N)
        hit_records: list[tuple[SearchHit, RetrievedRecord]] = []
        for hit in hits[:top_n]:
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

    def _semantic_rerank_hits(
        self,
        query: str,
        hits: Sequence[SearchHit],
        direct_records: dict[str, RetrievedRecord],
    ) -> tuple[SearchHit, ...]:
        if not self.reranker_enabled:
            return tuple(hits)

        top_n = getattr(self, "_reranker_top_n", DEFAULT_RERANKER_TOP_N)
        candidate_hits = tuple(hits[:top_n])
        if not candidate_hits:
            return tuple(hits)

        semantic_scores = self._predict_reranker_scores(query, candidate_hits, direct_records)
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

    def _context_assembler_for_compat(self) -> ContextAssembler:
        if hasattr(self, "_context_assembler"):
            return self._context_assembler
        return ContextAssembler(
            record_store=_HybridRecordStoreAdapter(self),
            rule_config=RULE_CONFIG,
        )

    def _query_needs_article_siblings(self, intent: QueryIntent) -> bool:
        return self._context_assembler_for_compat().query_needs_article_siblings(intent)

    def _add_article_sibling_contexts(
        self,
        contexts: Sequence[RetrievalContext],
        *,
        intent: QueryIntent,
        direct_records: dict[str, RetrievedRecord],
    ) -> tuple[RetrievalContext, ...]:
        return self._context_assembler_for_compat().add_article_sibling_contexts(
            contexts,
            intent=intent,
            direct_records=direct_records,
        )

    def _assemble_contexts(
        self,
        hits: Sequence[SearchHit],
        intent: QueryIntent | None = None,
    ) -> tuple[RetrievalContext, ...]:
        return self._context_assembler_for_compat().assemble_contexts(hits, intent=intent)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 8,
        prefetch_limit: int = 24,
    ) -> RetrievalResult:
        intent = self._route_query(query)
        query_variants = build_query_variants(intent)
        hits = self._qdrant_searcher.search(
            intent=intent,
            top_k=top_k,
            prefetch_limit=prefetch_limit,
            query_variants=query_variants,
        )
        hits = self._reference_expander.append_forced_reference_hits(hits, intent, limit=max(4, top_k))
        hits = self._reference_expander.append_reference_fallback_hits(hits, intent, limit=max(4, top_k))
        direct_records = self._record_store.fetch_records_from_hits(hits)
        hits = self._scorer.rerank_hits(hits, intent, direct_records)
        hits = self._semantic_reranker.semantic_rerank_hits(query, hits, direct_records)
        hits = self._reference_expander.pin_forced_reference_hits(hits, intent)[:top_k]
        contexts = self._context_assembler.assemble_contexts(hits, intent=intent)
        return RetrievalResult(
            query=query,
            intent=intent,
            hits=hits,
            contexts=contexts,
        )


__all__ = [
    "FORCED_REFERENCE_SCORE_MARGIN",
    "HybridRetriever",
    "RRF_K",
    "RULE_CONFIG",
    "RetrievedRecord",
    "RetrievalContext",
    "RetrievalResult",
    "SearchHit",
    "build_expanded_context_text",
    "contains_normalized_phrase",
    "context_article_key",
    "context_looks_like_enumeration_parent",
    "dedupe_records_by_chunk_id",
    "embed_query_via_http",
    "env_flag",
    "extend_context_with_records",
    "is_custom_http_embedding_provider",
    "load_manifest",
    "record_from_qdrant_payload",
    "record_reference_sort_key",
    "resolve_record_source",
    "route_query_with_llm",
]

