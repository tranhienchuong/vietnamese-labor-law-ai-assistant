from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
import re
from typing import Sequence

from .corpus_pipeline import normalize_for_matching
from .embeddings import embed_query_via_http, is_custom_http_embedding_provider
from .indexing import (
    PyViWordSegmenter,
    SparseBM25Encoder,
    build_qdrant_client,
    extract_legal_hint_tokens,
    load_sparse_encoder,
    make_qdrant_point_id,
    require_cross_encoder,
    require_qdrant,
    require_sentence_transformers,
)
from .heuristic_router import (
    ARTICLE_REF_RE,
    BENEFIT_COMPUTATION_QUERY_HINTS,
    CALCULATION_CONTEXT_HINTS,
    CALCULATION_QUERY_HINTS,
    DELEGATION_CONTEXT_HINTS,
    ENUMERATION_PARENT_CONTEXT_HINTS,
    IMPLEMENTATION_DETAIL_HINTS,
    LEGAL_ISSUE_ARTICLE_MAP,
    MATERNITY_CONTEXT_HINTS,
    MAX_ENUMERATION_CONTEXT_RECORDS,
    NO_NOTICE_QUERY_HINTS,
    QueryIntent,
    RETIREMENT_CONTEXT_HINTS,
    TERMINATION_ARTICLE_MAP,
    TERMINATION_BENEFIT_CONTEXT_HINTS,
    TERMINATION_QUERY_HINTS,
    TERMINATION_SECTION_HINTS,
    YEAR_COUNT_RE,
    build_query_variants,
    contains_normalized_phrase,
    dedupe_preserve_order,
    filter_specific_actor_labels,
    format_intent_summary,
    infer_employee_notice_period_reference,
    parse_reference_values,
    prioritize_issue_filters,
    query_asks_for_enumeration,
    query_asks_without_notice,
    route_query,
    route_query_heuristic,
)
from .query_router import query_intent_from_metadata, route_query_with_llm

DEFAULT_MAX_CONTEXT_CHARS = 8000
DEFAULT_MAX_CONTEXT_TOKENS = 1400
TOKEN_ESTIMATE_RE = re.compile(r"\S+")
DEFAULT_RERANKER_TOP_N = 24
RECORD_SOURCE_SQLITE = "sqlite"
RECORD_SOURCE_QDRANT_PAYLOAD = "qdrant_payload"
SUPPORTED_RECORD_SOURCES = {RECORD_SOURCE_SQLITE, RECORD_SOURCE_QDRANT_PAYLOAD}
RRF_K = 60.0
TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
FALSE_ENV_VALUES = frozenset({"0", "false", "no", "off"})
POINT_REF_ALPHABET = (
    "a",
    "b",
    "c",
    "d",
    "đ",
    "e",
    "g",
    "h",
    "i",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "x",
    "y",
)
POINT_REF_ORDER = {value: index for index, value in enumerate(POINT_REF_ALPHABET)}


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    if value in TRUE_ENV_VALUES:
        return True
    if value in FALSE_ENV_VALUES:
        return False
    return default


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


def resolve_record_source(manifest: dict[str, object]) -> str:
    configured = (
        os.getenv("RETRIEVER_RECORD_SOURCE", "").strip()
        or str(manifest.get("record_source") or "").strip()
        or RECORD_SOURCE_SQLITE
    )
    if configured not in SUPPORTED_RECORD_SOURCES:
        raise ValueError(
            "RETRIEVER_RECORD_SOURCE must be one of: "
            + ", ".join(sorted(SUPPORTED_RECORD_SOURCES))
        )
    return configured


def record_from_qdrant_payload(payload: dict[str, object]) -> RetrievedRecord:
    chunk_id = str(payload.get("chunk_id") or "").strip()
    if not chunk_id:
        raise ValueError("Qdrant payload is missing chunk_id.")

    return RetrievedRecord(
        chunk_id=chunk_id,
        parent_chunk_id=(
            str(payload.get("parent_chunk_id"))
            if payload.get("parent_chunk_id")
            else None
        ),
        citation_text=str(payload.get("citation_text") or ""),
        text=str(payload.get("text") or ""),
        dense_text=str(payload.get("dense_text") or ""),
        sparse_text=str(payload.get("sparse_text") or ""),
        payload=dict(payload),
    )


def point_ref_sort_key(point_ref: str) -> tuple[int, tuple[tuple[int, int | str], ...], str]:
    if not point_ref:
        return (10_000, (), "")

    normalized = point_ref.strip().lower().replace("Ä‘", "đ")
    base, *suffix_parts = normalized.split(".")
    base_order = POINT_REF_ORDER.get(base, 1_000 + ord(base[:1] or "~"))
    suffix_key: list[tuple[int, int | str]] = []
    for suffix in suffix_parts:
        suffix_key.append((0, int(suffix)) if suffix.isdigit() else (1, suffix))
    return (base_order, tuple(suffix_key), normalized)


def record_reference_sort_key(
    record: RetrievedRecord,
) -> tuple[int, int, str, tuple[int, tuple[tuple[int, int | str], ...], str], str]:
    level_order = {"article": 0, "clause": 1, "point": 2}
    level = str(record.payload.get("level") or "")
    clause_ref = str(record.payload.get("clause_ref") or "")
    point_ref = str(record.payload.get("point_ref") or "")
    try:
        clause_order = int(clause_ref)
    except ValueError:
        clause_order = 10_000
    return (
        level_order.get(level, 3),
        clause_order,
        clause_ref,
        point_ref_sort_key(point_ref),
        record.chunk_id,
    )


def build_expanded_context_text(
    context_record: RetrievedRecord,
    matched_records: Sequence[RetrievedRecord],
) -> str:
    parts: list[str] = []
    seen_surfaces: list[str] = []

    def add_text(text: str) -> None:
        stripped = text.strip()
        normalized = normalize_for_matching(stripped)
        if not normalized:
            return
        if any(normalized == seen or normalized in seen for seen in seen_surfaces):
            return
        seen_surfaces.append(normalized)
        parts.append(stripped)

    add_text(context_record.text)
    for record in matched_records:
        if record.chunk_id == context_record.chunk_id:
            continue
        add_text(record.text)

    return "\n\n".join(parts).strip()


def extend_context_with_records(
    context: RetrievalContext,
    records: Sequence[RetrievedRecord],
    *,
    force_include: bool = False,
    replace_text: bool = False,
) -> RetrievalContext:
    if not records:
        return context

    parts: list[str] = []
    seen_surfaces: list[str] = []

    def add_text(text: str) -> None:
        stripped = text.strip()
        normalized = normalize_for_matching(stripped)
        if not normalized:
            return
        if any(normalized == seen or normalized in seen for seen in seen_surfaces):
            return
        seen_surfaces.append(normalized)
        parts.append(stripped)

    if not replace_text:
        add_text(context.text)
    for record in records:
        add_text(record.text)

    payload = dict(context.payload)
    if force_include:
        payload["retrieval_force_include"] = True

    matched_chunk_ids = (
        (context.chunk_id, *(record.chunk_id for record in records))
        if replace_text
        else (*context.matched_chunk_ids, *(record.chunk_id for record in records))
    )
    matched_citations = (
        (context.citation_text, *(record.citation_text for record in records))
        if replace_text
        else (*context.matched_citations, *(record.citation_text for record in records))
    )

    return RetrievalContext(
        chunk_id=context.chunk_id,
        citation_text=context.citation_text,
        text="\n\n".join(parts).strip(),
        payload=payload,
        score=context.score,
        matched_chunk_ids=dedupe_preserve_order(matched_chunk_ids),
        matched_citations=dedupe_preserve_order(matched_citations),
    )


def context_looks_like_enumeration_parent(context: RetrievalContext) -> bool:
    level = str(context.payload.get("level") or "")
    if level not in {"article", "clause"}:
        return False
    return contains_normalized_phrase(
        normalize_for_matching(context.text),
        ENUMERATION_PARENT_CONTEXT_HINTS,
    )


def load_manifest(index_path: Path) -> dict[str, object]:
    if index_path.is_dir():
        manifest_path = index_path / "current.json"
    else:
        manifest_path = index_path
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def build_context_block(context: RetrievalContext, index: int) -> str:
    lines = [
        f"[NGU CANH {index}]",
        f"Co so phap ly: {context.citation_text}",
    ]

    unique_matched_citations = dedupe_preserve_order(context.matched_citations)
    if unique_matched_citations and unique_matched_citations != (context.citation_text,):
        lines.append("Match goc:")
        lines.extend(f"- {citation}" for citation in unique_matched_citations)

    lines.extend(
        [
            "Noi dung:",
            context.text.strip(),
        ]
    )
    return "\n".join(lines).strip()


def estimate_token_count(text: str) -> int:
    return len(TOKEN_ESTIMATE_RE.findall(text))


def context_article_key(context: RetrievalContext) -> tuple[str, str] | None:
    document_id = str(context.payload.get("document_id") or "")
    article_number = str(context.payload.get("article_number") or "")
    if not document_id or not article_number:
        return None
    return document_id, article_number


def diversify_contexts_by_article(contexts: Sequence[RetrievalContext]) -> tuple[RetrievalContext, ...]:
    first_per_article: list[RetrievalContext] = []
    remaining: list[RetrievalContext] = []
    seen_article_keys: set[tuple[str, str]] = set()

    for context in contexts:
        key = context_article_key(context)
        force_include = bool(context.payload.get("retrieval_force_include"))
        if key is None:
            first_per_article.append(context)
            continue
        if key not in seen_article_keys:
            seen_article_keys.add(key)
            first_per_article.append(context)
        elif force_include:
            first_per_article.append(context)
        else:
            remaining.append(context)

    return tuple((*first_per_article, *remaining))


def select_contexts_for_prompt(
    contexts: Sequence[RetrievalContext],
    *,
    max_contexts: int | None = None,
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_tokens: int | None = DEFAULT_MAX_CONTEXT_TOKENS,
) -> tuple[RetrievalContext, ...]:
    ranked_contexts = diversify_contexts_by_article(contexts)
    limited_contexts = ranked_contexts[:max_contexts] if max_contexts is not None else ranked_contexts
    selected: list[RetrievalContext] = []
    current_len = 0
    current_tokens = 0

    for context in limited_contexts:
        block = build_context_block(context, len(selected) + 1)
        block_tokens = estimate_token_count(block)
        separator_len = 2 if selected else 0
        next_len = current_len + separator_len + len(block)
        next_tokens = current_tokens + block_tokens
        exceeds_char_budget = max_chars > 0 and next_len > max_chars
        exceeds_token_budget = max_tokens is not None and max_tokens > 0 and next_tokens > max_tokens

        if selected and (exceeds_char_budget or exceeds_token_budget):
            break

        if not exceeds_char_budget and not exceeds_token_budget:
            selected.append(context)
            current_len = next_len
            current_tokens = next_tokens
            continue

        # Preserve the highest-ranked block intact instead of truncating legal text mid-sentence.
        if not selected:
            selected.append(context)
            current_len = next_len
            current_tokens = next_tokens
        break

    return tuple(selected)


def actor_labels_for_sparse(intent: QueryIntent) -> tuple[str, ...]:
    if (
        "definition" in intent.query_types
        or "quyen va nghia vu" in intent.normalized_query
        or "quyen nghia vu" in intent.normalized_query
    ):
        return tuple(intent.actor_filters)
    return filter_specific_actor_labels(intent.actor_filters)


def dedupe_records_by_chunk_id(records: Sequence[RetrievedRecord]) -> tuple[RetrievedRecord, ...]:
    seen: set[str] = set()
    ordered: list[RetrievedRecord] = []
    for record in records:
        if record.chunk_id in seen:
            continue
        seen.add(record.chunk_id)
        ordered.append(record)
    return tuple(ordered)


def format_context_for_prompt(
    contexts: Sequence[RetrievalContext],
    *,
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_tokens: int | None = DEFAULT_MAX_CONTEXT_TOKENS,
) -> str:
    selected_contexts = select_contexts_for_prompt(
        contexts,
        max_chars=max_chars,
        max_tokens=max_tokens,
    )
    blocks = [
        build_context_block(context, index)
        for index, context in enumerate(selected_contexts, start=1)
    ]

    return "\n\n".join(blocks).strip()


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
        self._collection_name = (
            os.getenv("QDRANT_COLLECTION", "").strip()
            or str(self._manifest["collection_name"])
        )
        self._dense_model_name = str(self._manifest["dense_model_name"])
        self._dense_vector_name = str(self._manifest["dense_vector_name"])
        self._sparse_vector_name = str(self._manifest["sparse_vector_name"])
        self._segmenter = PyViWordSegmenter()
        self._sparse_encoder = load_sparse_encoder(Path(str(self._manifest["sparse_encoder_path"])))
        self._sqlite: sqlite3.Connection | None = None
        if self._record_source == RECORD_SOURCE_SQLITE:
            if self._records_db_path is None:
                raise ValueError("records_db_path is required when record_source is sqlite.")
            self._sqlite = sqlite3.connect(self._records_db_path)
            self._sqlite.row_factory = sqlite3.Row
        qdrant_client_cls, self._qdrant_models = require_qdrant()
        self._qdrant = build_qdrant_client(qdrant_client_cls, self._qdrant_path)
        self._dense_model = None
        self._reranker_model_name = str(reranker_model or "").strip()
        self._reranker_top_n = max(1, int(reranker_top_n))
        self._reranker = None
        self._query_router_enabled = (
            env_flag("QUERY_ROUTER_ENABLED", True)
            if query_router_enabled is None
            else bool(query_router_enabled)
        )
        self._query_router_provider = os.getenv("QUERY_ROUTER_PROVIDER", "").strip() or None
        self._query_router_model = os.getenv("QUERY_ROUTER_MODEL", "").strip() or None
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
        if self._sqlite is not None:
            self._sqlite.close()

    def _get_dense_model(self):
        if self._dense_model is None:
            sentence_transformer_cls = require_sentence_transformers()
            self._dense_model = sentence_transformer_cls(self._dense_model_name, device=self._device)
        return self._dense_model

    def _get_reranker(self):
        if not self.reranker_enabled:
            return None
        if self._reranker is None:
            cross_encoder_cls = require_cross_encoder()
            self._reranker = cross_encoder_cls(self._reranker_model_name, device=self._device)
        return self._reranker

    def _route_query(self, query: str) -> QueryIntent:
        if not self._query_router_enabled:
            return route_query_heuristic(query)

        try:
            return route_query_with_llm(
                query,
                provider=self._query_router_provider,
                model=self._query_router_model,
            )
        except Exception:
            if not self._query_router_fallback_to_heuristic:
                raise
            return route_query_heuristic(query)

    def _encode_dense_query(self, query: str) -> list[float]:
        if is_custom_http_embedding_provider():
            return embed_query_via_http(query)

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
        sparse_query_text = query_text or "\n".join(
            part for part in (intent.raw_query, *intent.query_expansions) if part
        )
        tokens = self._segmenter.segment(sparse_query_text)
        tokens.extend(extract_legal_hint_tokens(sparse_query_text))
        tokens.extend(f"dieu_{value}" for value in intent.all_article_numbers)
        tokens.extend(f"khoan_{value}" for value in intent.clause_refs)
        tokens.extend(f"diem_{value}" for value in intent.point_refs)
        tokens.extend(f"topic_{value}" for value in intent.topic_filters)
        tokens.extend(f"issue_{value}" for value in intent.issue_filters)
        tokens.extend(f"actor_{value}" for value in actor_labels_for_sparse(intent))
        tokens.extend(f"qtype_{value}" for value in intent.query_types)
        sparse_query = self._sparse_encoder.encode_query(tokens)
        sparse_vector = self._qdrant_models.SparseVector(
            indices=sparse_query.indices,
            values=sparse_query.values,
        )
        return tokens, sparse_vector

    def _build_query_filter(self, intent: QueryIntent):
        must_conditions: list[object] = []
        models = self._qdrant_models

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

    def _build_reference_boost_filter(self, intent: QueryIntent):
        if not intent.legal_reference_filters:
            return None

        models = self._qdrant_models
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

    def _build_issue_focus_filter(self, intent: QueryIntent):
        prioritized_issue_filters = prioritize_issue_filters(intent.issue_filters)
        if not prioritized_issue_filters:
            return None

        models = self._qdrant_models
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

    def _records_from_rows(self, rows: Sequence[sqlite3.Row]) -> dict[str, RetrievedRecord]:
        records: dict[str, RetrievedRecord] = {}
        for row in rows:
            payload = json.loads(row["payload_json"])
            records[str(row["chunk_id"])] = RetrievedRecord(
                chunk_id=str(row["chunk_id"]),
                parent_chunk_id=str(row["parent_chunk_id"]) if row["parent_chunk_id"] else None,
                citation_text=str(row["citation_text"]),
                text=str(row["text"]),
                dense_text=str(row["dense_text"]),
                sparse_text=str(row["sparse_text"]),
                payload=payload,
            )
        return records

    def _uses_qdrant_payload_records(self) -> bool:
        return (
            getattr(self, "_record_source", RECORD_SOURCE_SQLITE)
            == RECORD_SOURCE_QDRANT_PAYLOAD
        )

    def _records_from_qdrant_points(self, points: Sequence[object]) -> dict[str, RetrievedRecord]:
        records: dict[str, RetrievedRecord] = {}
        for point in points:
            payload = getattr(point, "payload", None)
            if not isinstance(payload, dict):
                continue
            try:
                record = record_from_qdrant_payload(payload)
            except ValueError:
                continue
            records[record.chunk_id] = record
        return records

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
        ordered_ids = dedupe_preserve_order(chunk_ids)
        if not ordered_ids:
            return {}

        if self._uses_qdrant_payload_records():
            return self._fetch_records_from_qdrant_ids(ordered_ids)

        if self._sqlite is None:
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

    def _build_reference_payload_filter(
        self,
        *,
        document_ids: Sequence[str] = (),
        article_numbers: Sequence[str] = (),
        clause_refs: Sequence[str] = (),
        point_refs: Sequence[str] = (),
        exclude_chunk_ids: Sequence[str] = (),
    ):
        models = self._qdrant_models
        must_conditions: list[object] = []
        must_not_conditions: list[object] = []

        def add_match_any(field_name: str, values: Sequence[str]) -> None:
            ordered_values = dedupe_preserve_order(tuple(value for value in values if value))
            if not ordered_values:
                return
            must_conditions.append(
                models.FieldCondition(
                    key=field_name,
                    match=models.MatchAny(any=list(ordered_values)),
                )
            )

        add_match_any("document_id", document_ids)
        add_match_any("article_number", article_numbers)
        add_match_any("clause_ref", clause_refs)
        add_match_any("point_ref", point_refs)

        excluded_ids = dedupe_preserve_order(tuple(value for value in exclude_chunk_ids if value))
        if excluded_ids:
            must_not_conditions.append(
                models.FieldCondition(
                    key="chunk_id",
                    match=models.MatchAny(any=list(excluded_ids)),
                )
            )

        if not must_conditions and not must_not_conditions:
            return None
        return models.Filter(
            must=must_conditions or None,
            must_not=must_not_conditions or None,
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
        query_filter = self._build_reference_payload_filter(
            document_ids=document_ids,
            article_numbers=article_numbers,
            clause_refs=clause_refs,
            point_refs=point_refs,
            exclude_chunk_ids=exclude_chunk_ids,
        )
        if query_filter is None:
            return ()

        points, _ = self._qdrant.scroll(
            collection_name=self._collection_name,
            scroll_filter=query_filter,
            limit=max(1, int(limit)) * 4,
            with_payload=True,
            with_vectors=False,
        )
        records = tuple(self._records_from_qdrant_points(points).values())
        return tuple(sorted(records, key=record_reference_sort_key)[: max(1, int(limit))])

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
        if self._uses_qdrant_payload_records():
            return self._fetch_records_by_reference_from_qdrant(
                document_ids=document_ids,
                article_numbers=article_numbers,
                clause_refs=clause_refs,
                point_refs=point_refs,
                exclude_chunk_ids=exclude_chunk_ids,
                limit=limit,
            )

        if self._sqlite is None:
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
        add_in_filter("point_ref", point_refs)

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
        return self._fetch_records_by_reference(
            document_ids=(document_id,),
            article_numbers=(article_number,),
            exclude_chunk_ids=exclude_chunk_ids,
            limit=limit,
        )

    @staticmethod
    def _record_to_search_hit(record: RetrievedRecord, score: float) -> SearchHit:
        return SearchHit(
            chunk_id=record.chunk_id,
            qdrant_point_id=str(record.payload.get("qdrant_point_id") or record.chunk_id),
            score=score,
            citation_text=record.citation_text,
            payload=record.payload,
        )

    def _append_reference_fallback_hits(
        self,
        hits: Sequence[SearchHit],
        intent: QueryIntent,
        *,
        limit: int,
    ) -> tuple[SearchHit, ...]:
        fallback_kind = "explicit" if intent.article_numbers else "high_confidence"
        article_candidates = intent.article_numbers
        if not article_candidates:
            article_candidates = tuple(
                article
                for article in intent.force_reference_article_numbers
                if article not in intent.article_numbers
            )
            if len(article_candidates) > 4:
                return tuple(hits)

        if not article_candidates:
            return tuple(hits)

        existing_records = self._fetch_records_from_hits(hits)
        existing_articles = {
            str(record.payload.get("article_number") or "")
            for record in existing_records.values()
        }
        missing_articles = tuple(
            article for article in article_candidates if article not in existing_articles
        )
        if fallback_kind == "explicit":
            article_numbers_to_fetch = missing_articles or article_candidates
            if not missing_articles and not intent.clause_refs and not intent.point_refs:
                return tuple(hits)
        else:
            article_numbers_to_fetch = missing_articles
            if not article_numbers_to_fetch or intent.clause_refs or intent.point_refs:
                return tuple(hits)

        if article_numbers_to_fetch and not intent.clause_refs and not intent.point_refs:
            fallback_records_list: list[RetrievedRecord] = []
            excluded_chunk_ids: set[str] = set()
            primary_article_limit = max(
                limit,
                8
                if (
                    "definition" in intent.query_types
                    or "giai_thich_tu_ngu" in intent.issue_filters
                    or "loai_hop_dong" in intent.issue_filters
                )
                else 4,
            )
            secondary_article_limit = max(2, min(4, limit))
            for index, article_number in enumerate(article_numbers_to_fetch):
                fetched_records = self._fetch_records_by_reference(
                    document_ids=intent.document_filters,
                    article_numbers=(article_number,),
                    exclude_chunk_ids=tuple(excluded_chunk_ids),
                    limit=primary_article_limit if index == 0 else secondary_article_limit,
                )
                fallback_records_list.extend(fetched_records)
                excluded_chunk_ids.update(record.chunk_id for record in fetched_records)
            fallback_records = tuple(fallback_records_list)
        else:
            fallback_records = self._fetch_records_by_reference(
                document_ids=intent.document_filters,
                article_numbers=article_numbers_to_fetch,
                clause_refs=intent.clause_refs,
                point_refs=intent.point_refs,
                limit=limit,
            )
        if not fallback_records:
            return tuple(hits)

        max_existing_score = max((hit.score for hit in hits), default=0.0)
        fallback_score = (
            max_existing_score + 1e-3
            if fallback_kind == "explicit"
            else max(max_existing_score - 1e-3, 0.0)
        )
        fallback_hits_by_chunk_id = {
            record.chunk_id: self._record_to_search_hit(record, fallback_score - (rank * 1e-4))
            for rank, record in enumerate(dedupe_records_by_chunk_id(fallback_records))
        }
        promoted_hits: list[SearchHit] = []
        used_fallback_chunk_ids: set[str] = set()
        for hit in hits:
            fallback_hit = fallback_hits_by_chunk_id.get(hit.chunk_id)
            if fallback_hit is None:
                promoted_hits.append(hit)
                continue
            promoted_hits.append(fallback_hit if fallback_kind == "explicit" else hit)
            used_fallback_chunk_ids.add(hit.chunk_id)
        promoted_hits.extend(
            fallback_hit
            for chunk_id, fallback_hit in fallback_hits_by_chunk_id.items()
            if chunk_id not in used_fallback_chunk_ids
        )
        return tuple(promoted_hits)

    def _score_hit_relevance(
        self,
        hit: SearchHit,
        record: RetrievedRecord,
        intent: QueryIntent,
    ) -> float:
        boost = 0.0
        prioritized_issue_filters = prioritize_issue_filters(intent.issue_filters)
        article_number = str(record.payload.get("article_number") or "").lower()
        clause_ref = str(record.payload.get("clause_ref") or "").lower()
        point_ref = str(record.payload.get("point_ref") or "").lower()
        document_id = str(record.payload.get("document_id") or "")
        level = str(record.payload.get("level") or "")
        actor_values = {str(value) for value in (record.payload.get("actor") or [])}
        topic_values = {str(value) for value in (record.payload.get("topic") or [])}
        issue_values = {str(value) for value in (record.payload.get("issue_type") or [])}
        section_heading = normalize_for_matching(str(record.payload.get("section_heading") or ""))
        heading_text = normalize_for_matching(
            " ".join(
                part
                for part in [
                    str(record.payload.get("article_title") or ""),
                    str(record.payload.get("heading") or ""),
                    record.citation_text,
                ]
                if part
            )
        )
        normalized_text = normalize_for_matching(f" {record.citation_text} {record.text} ")
        is_calculation_query = (
            contains_normalized_phrase(intent.normalized_query, CALCULATION_QUERY_HINTS)
            or (
                "tro_cap_thoi_viec" in intent.issue_filters
                and (
                    contains_normalized_phrase(intent.normalized_query, BENEFIT_COMPUTATION_QUERY_HINTS)
                    or YEAR_COUNT_RE.search(intent.normalized_query) is not None
                )
            )
        )
        wants_implementation_detail = contains_normalized_phrase(
            intent.normalized_query,
            IMPLEMENTATION_DETAIL_HINTS,
        )
        is_termination_query = (
            "cham_dut_hop_dong_lao_dong" in intent.topic_filters
            or "tro_cap_thoi_viec" in intent.issue_filters
            or "tro_cap_mat_viec" in intent.issue_filters
            or "nghia_vu_khi_cham_dut" in intent.issue_filters
            or contains_normalized_phrase(intent.normalized_query, TERMINATION_QUERY_HINTS)
        )
        is_termination_benefit_query = (
            is_termination_query
            and (
                "tro_cap" in intent.topic_filters
                or "tro_cap_thoi_viec" in intent.issue_filters
                or "tro_cap_mat_viec" in intent.issue_filters
            )
        )
        query_has_maternity_hint = contains_normalized_phrase(intent.normalized_query, MATERNITY_CONTEXT_HINTS)
        query_has_retirement_hint = contains_normalized_phrase(intent.normalized_query, RETIREMENT_CONTEXT_HINTS)
        prefers_primary_law = (
            is_termination_query
            and not wants_implementation_detail
            and "nghi dinh" not in intent.normalized_query
        )

        if intent.all_article_numbers and article_number in intent.all_article_numbers:
            boost += 0.2
        if intent.clause_refs and clause_ref in intent.clause_refs:
            boost += 0.25
        if article_number == "20" and "loai_hop_dong" in intent.issue_filters:
            if point_ref == "a" and "khong xac dinh thoi han" in intent.normalized_query:
                boost += 0.75
            elif (
                point_ref == "b"
                and "xac dinh thoi han" in intent.normalized_query
                and "khong xac dinh thoi han" not in intent.normalized_query
            ):
                boost += 0.75
        if article_number == "17" and {
            "giu_giay_to_goc",
            "dat_coc_bao_dam",
            "hanh_vi_cam_khi_giao_ket",
        }.intersection(intent.issue_filters):
            boost += 0.9
        if "phan_biet_doi_xu" in intent.issue_filters:
            if article_number == "3" and clause_ref == "8":
                boost += 1.0
            if article_number in {"8", "11", "135", "16"}:
                boost += 0.7
            if article_number == "8" and document_id != "45-2019-qh14":
                boost -= 1.2
        if "bao_ve_thai_san" in intent.issue_filters and article_number == "137":
            boost += 0.8
        if "lam_ban_dem" in intent.issue_filters and article_number == "106":
            boost += 0.8
        if "quay_roi_tinh_duc" in intent.issue_filters and article_number in {"3", "8", "35", "118"}:
            boost += 0.65
        if "xu_ly_ky_luat_lao_dong" in intent.issue_filters and article_number == "127":
            boost += 0.65
        if "ep_nghi_viec" in intent.issue_filters and article_number in {"7", "15", "34", "36", "39", "41"}:
            boost += 0.45
        if "han_che_viec_lam_sau_nghi" in intent.issue_filters and article_number in {"10", "21", "15"}:
            boost += 0.45
        if "bao_mat_bi_mat_kinh_doanh" in intent.issue_filters and article_number in {"21", "10", "15"}:
            boost += 0.35
        if "dieu_khoan_bat_cong" in intent.issue_filters and article_number in {"15", "49", "51"}:
            boost += 0.35
        if (
            {"du_lieu_ca_nhan", "thong_tin_suc_khoe"}.intersection(intent.issue_filters)
            and article_number in {"16", "21", "6"}
        ):
            boost += 0.25
        if "quyen_nghia_vu_nguoi_lao_dong" in intent.issue_filters and article_number == "5":
            boost += 0.9
        if "quyen_nghia_vu_nguoi_su_dung_lao_dong" in intent.issue_filters and article_number == "6":
            boost += 0.9
        if prioritized_issue_filters and issue_values.intersection(prioritized_issue_filters):
            boost += 0.15
        if intent.topic_filters and topic_values.intersection(intent.topic_filters):
            boost += 0.05
        specific_actor_filters = actor_labels_for_sparse(intent)
        if specific_actor_filters and actor_values.intersection(specific_actor_filters):
            boost += 0.04
        if "time_limit" in intent.query_types and contains_normalized_phrase(
            normalized_text,
            ("ngay", "thang", "nam", "thoi han", "khong qua", "it nhat"),
        ):
            boost += 0.08
        if "money_percentage" in intent.query_types and contains_normalized_phrase(
            normalized_text,
            ("%", "phan tram", "tien luong", "it nhat", "bang", "muc luong"),
        ):
            boost += 0.08
        if "procedure" in intent.query_types and contains_normalized_phrase(
            normalized_text,
            ("thong bao", "tham khao y kien", "bien ban", "thoi han", "to chuc dai dien"),
        ):
            boost += 0.08
        if "classification" in intent.query_types and contains_normalized_phrase(
            normalized_text,
            ("hinh thuc", "don phuong cham dut", "cac truong hop", "ap dung"),
        ):
            boost += 0.08
        if query_asks_without_notice(intent):
            if article_number == "35" and clause_ref == "2":
                boost += 0.75
            if article_number in {"37", "45"}:
                boost -= 0.45

        if is_termination_query:
            if contains_normalized_phrase(section_heading, TERMINATION_SECTION_HINTS):
                boost += 0.25
            if "cham_dut_hop_dong_lao_dong" in topic_values:
                boost += 0.15
            if "nghia_vu_khi_cham_dut" in issue_values:
                boost += 0.1

        if is_termination_benefit_query:
            if issue_values.intersection({"tro_cap_thoi_viec", "tro_cap_mat_viec"}):
                boost += 0.35
            if contains_normalized_phrase(normalized_text, TERMINATION_BENEFIT_CONTEXT_HINTS):
                boost += 0.15
            if "bao hiem that nghiep" in intent.normalized_query and "bao hiem that nghiep" in normalized_text:
                boost += 0.25
            if contains_normalized_phrase(heading_text, MATERNITY_CONTEXT_HINTS) and not query_has_maternity_hint:
                boost -= 0.7
            if contains_normalized_phrase(heading_text, RETIREMENT_CONTEXT_HINTS) and not query_has_retirement_hint:
                boost -= 0.45
            if "tro_cap_thoi_viec" in prioritized_issue_filters:
                if "tro cap thoi viec" in heading_text:
                    boost += 0.4
                if "tro cap mat viec" in heading_text and "tro cap thoi viec" not in heading_text:
                    boost -= 0.2
                if "nghia vu" in heading_text or "trach nhiem" in heading_text:
                    boost -= 0.15
            if "tro_cap_mat_viec" in prioritized_issue_filters:
                if "tro cap mat viec" in heading_text:
                    boost += 0.4
                if "tro cap thoi viec" in heading_text and "tro cap mat viec" not in heading_text:
                    boost -= 0.2

        if not wants_implementation_detail and not intent.point_refs and not intent.clause_refs:
            if level == "clause":
                boost += 0.12
            elif level == "point":
                boost -= 0.35

        if prefers_primary_law:
            if document_id == "45-2019-qh14":
                boost += 0.22
            elif document_id == "nghi-dinh-145-2020-nd-cp":
                boost -= 0.08
                if level == "point":
                    boost -= 0.18

        if is_calculation_query:
            if contains_normalized_phrase(normalized_text, CALCULATION_CONTEXT_HINTS):
                boost += 0.25
            if (
                contains_normalized_phrase(normalized_text, DELEGATION_CONTEXT_HINTS)
                and not wants_implementation_detail
            ):
                boost -= 0.9

        return hit.score + boost

    def _rerank_hits(
        self,
        hits: Sequence[SearchHit],
        intent: QueryIntent,
        direct_records: dict[str, RetrievedRecord],
    ) -> tuple[SearchHit, ...]:
        scored_hits: list[tuple[float, SearchHit]] = []
        for hit in hits:
            record = direct_records.get(hit.chunk_id)
            adjusted_score = self._score_hit_relevance(hit, record, intent) if record is not None else hit.score
            scored_hits.append((adjusted_score, hit))

        ordered = sorted(
            scored_hits,
            key=lambda item: (-item[0], -item[1].score, item[1].citation_text),
        )
        return tuple(
            SearchHit(
                chunk_id=hit.chunk_id,
                qdrant_point_id=hit.qdrant_point_id,
                score=adjusted_score,
                citation_text=hit.citation_text,
                payload=hit.payload,
            )
            for adjusted_score, hit in ordered
        )

    def _predict_reranker_scores(
        self,
        query: str,
        hits: Sequence[SearchHit],
        direct_records: dict[str, RetrievedRecord],
    ) -> dict[str, float]:
        reranker = self._get_reranker()
        if reranker is None:
            return {}

        hit_records: list[tuple[SearchHit, RetrievedRecord]] = []
        for hit in hits[: self._reranker_top_n]:
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

        candidate_hits = tuple(hits[: self._reranker_top_n])
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

    def _query_needs_article_siblings(self, intent: QueryIntent) -> bool:
        if query_asks_for_enumeration(intent):
            return True
        if intent.clause_refs or intent.point_refs:
            return True
        if set(intent.query_types).intersection(
            {
                "procedure",
                "remedy",
                "definition",
                "classification",
                "enumeration",
                "money_percentage",
                "time_limit",
            }
        ):
            return True
        return bool(
            set(intent.issue_filters).intersection(
                {
                    "can_cu_cham_dut",
                    "boi_thuong",
                    "tro_cap_thoi_viec",
                    "tro_cap_mat_viec",
                    "nghia_vu_khi_cham_dut",
                    "quyen_don_phuong_cham_dut",
                    "trai_phap_luat",
                    "thoi_han_bao_truoc",
                }
            )
        )

    def _add_article_sibling_contexts(
        self,
        contexts: Sequence[RetrievalContext],
        *,
        intent: QueryIntent,
        direct_records: dict[str, RetrievedRecord],
    ) -> tuple[RetrievalContext, ...]:
        if not self._query_needs_article_siblings(intent):
            return tuple(contexts)

        seen_chunk_ids = {context.chunk_id for context in contexts}
        seen_chunk_ids.update(
            chunk_id
            for context in contexts
            for chunk_id in context.matched_chunk_ids
        )
        context_article_levels = {
            context_article_key(context): str(context.payload.get("level") or "")
            for context in contexts
            if context_article_key(context) is not None
        }
        records_by_article: dict[tuple[str, str], RetrievedRecord] = {}
        for record in direct_records.values():
            document_id = str(record.payload.get("document_id") or "")
            article_number = str(record.payload.get("article_number") or "")
            if not document_id or not article_number:
                continue
            key = (document_id, article_number)
            article_context_needs_children = any(
                context_article_key(context) == key
                and context_looks_like_enumeration_parent(context)
                for context in contexts
            )
            if (
                context_article_levels.get(key) == "article"
                and not query_asks_for_enumeration(intent)
                and not article_context_needs_children
            ):
                continue
            records_by_article.setdefault(key, record)

        if not records_by_article:
            return tuple(contexts)

        expanded_contexts: list[RetrievalContext] = []
        added_sibling_ids: set[str] = set()
        for context in contexts:
            key = context_article_key(context)
            if key not in records_by_article:
                expanded_contexts.append(context)
                continue
            document_id, article_number = key
            expanded_context = context
            merge_enumeration_records = (
                query_asks_for_enumeration(intent)
                or context_looks_like_enumeration_parent(context)
            )
            notice_clause_refs, notice_point_refs = infer_employee_notice_period_reference(intent)
            if article_number == "35" and notice_clause_refs:
                notice_limit = 4 if notice_point_refs else MAX_ENUMERATION_CONTEXT_RECORDS
                siblings = self._fetch_records_by_reference(
                    document_ids=(document_id,),
                    article_numbers=(article_number,),
                    clause_refs=notice_clause_refs,
                    point_refs=notice_point_refs,
                    exclude_chunk_ids=tuple(
                        (seen_chunk_ids if notice_point_refs else set()) | added_sibling_ids
                    ),
                    limit=notice_limit,
                )
                force_include_siblings = True
                if not notice_point_refs:
                    expanded_context = extend_context_with_records(
                        context,
                        siblings,
                        force_include=True,
                        replace_text=True,
                    )
                    added_sibling_ids.update(sibling.chunk_id for sibling in siblings)
                    expanded_contexts.append(expanded_context)
                    continue
            elif merge_enumeration_records:
                context_level = str(context.payload.get("level") or "")
                clause_ref = str(context.payload.get("clause_ref") or "")
                if context_level == "clause" and context_looks_like_enumeration_parent(context) and clause_ref:
                    siblings = self._fetch_records_by_reference(
                        document_ids=(document_id,),
                        article_numbers=(article_number,),
                        clause_refs=(clause_ref,),
                        exclude_chunk_ids=tuple(added_sibling_ids),
                        limit=MAX_ENUMERATION_CONTEXT_RECORDS,
                    )
                else:
                    siblings = self._fetch_article_siblings(
                        document_id=document_id,
                        article_number=article_number,
                        exclude_chunk_ids=tuple(added_sibling_ids),
                        limit=MAX_ENUMERATION_CONTEXT_RECORDS,
                    )
                expanded_context = extend_context_with_records(
                    context,
                    siblings,
                    force_include=True,
                    replace_text=True,
                )
                added_sibling_ids.update(sibling.chunk_id for sibling in siblings)
                expanded_contexts.append(expanded_context)
                continue
            else:
                siblings = self._fetch_article_siblings(
                    document_id=document_id,
                    article_number=article_number,
                    exclude_chunk_ids=tuple(seen_chunk_ids | added_sibling_ids),
                    limit=3,
                )
                force_include_siblings = False
            expanded_contexts.append(expanded_context)
            for rank, sibling in enumerate(siblings, start=1):
                added_sibling_ids.add(sibling.chunk_id)
                payload = dict(sibling.payload)
                if force_include_siblings:
                    payload["retrieval_force_include"] = True
                expanded_contexts.append(
                    RetrievalContext(
                        chunk_id=sibling.chunk_id,
                        citation_text=sibling.citation_text,
                        text=sibling.text,
                        payload=payload,
                        score=max(context.score - (rank * 1e-3), 0.0),
                        matched_chunk_ids=(sibling.chunk_id,),
                        matched_citations=(sibling.citation_text,),
                    )
                )

        return tuple(expanded_contexts)

    def _assemble_contexts(
        self,
        hits: Sequence[SearchHit],
        intent: QueryIntent | None = None,
    ) -> tuple[RetrievalContext, ...]:
        direct_records = self._fetch_records_from_hits(hits)
        parent_ids = [
            record.parent_chunk_id
            for record in direct_records.values()
            if record.parent_chunk_id and record.parent_chunk_id not in direct_records
        ]
        parent_records = self._fetch_records(parent_ids)
        record_map = {**direct_records, **parent_records}

        grouped: dict[str, dict[str, object]] = {}
        for rank, hit in enumerate(hits):
            matched_record = direct_records.get(hit.chunk_id)
            if matched_record is None:
                continue

            context_id = matched_record.parent_chunk_id or matched_record.chunk_id
            context_record = record_map.get(context_id, matched_record)
            group = grouped.setdefault(
                context_record.chunk_id,
                {
                    "record": context_record,
                    "score": hit.score,
                    "rank": rank,
                    "matched_chunk_ids": [],
                    "matched_citations": [],
                    "matched_records": [],
                },
            )
            group["score"] = max(float(group["score"]), hit.score)
            group["rank"] = min(int(group["rank"]), rank)
            group["matched_chunk_ids"].append(hit.chunk_id)
            group["matched_citations"].append(hit.citation_text)
            group["matched_records"].append(matched_record)

        ordered_groups = sorted(
            grouped.values(),
            key=lambda item: (-float(item["score"]), int(item["rank"])),
        )

        contexts: list[RetrievalContext] = []
        for item in ordered_groups:
            record = item["record"]
            assert isinstance(record, RetrievedRecord)
            matched_chunk_ids = dedupe_preserve_order(item["matched_chunk_ids"])
            matched_citations = dedupe_preserve_order(item["matched_citations"])
            matched_records = tuple(
                record
                for record in item["matched_records"]
                if isinstance(record, RetrievedRecord)
            )
            contexts.append(
                RetrievalContext(
                    chunk_id=record.chunk_id,
                    citation_text=record.citation_text,
                    text=build_expanded_context_text(record, matched_records),
                    payload=record.payload,
                    score=float(item["score"]),
                    matched_chunk_ids=matched_chunk_ids,
                    matched_citations=matched_citations,
                )
            )
        if intent is None:
            return tuple(contexts)
        return self._add_article_sibling_contexts(
            contexts,
            intent=intent,
            direct_records=direct_records,
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 8,
        prefetch_limit: int = 24,
    ) -> RetrievalResult:
        intent = self._route_query(query)
        query_filter = self._build_query_filter(intent)
        reference_boost_filter = self._build_reference_boost_filter(intent)
        issue_focus_filter = self._build_issue_focus_filter(intent)
        query_variants = build_query_variants(intent)
        _, sparse_query = self._encode_sparse_query(intent)
        models = self._qdrant_models

        prefetches = []
        for variant in query_variants:
            dense_query = self._encode_dense_query(variant)
            _, variant_sparse_query = self._encode_sparse_query(intent, query_text=variant)
            prefetches.extend(
                [
                    models.Prefetch(
                        query=dense_query,
                        using=self._dense_vector_name,
                        filter=query_filter,
                        limit=prefetch_limit,
                    ),
                    models.Prefetch(
                        query=variant_sparse_query,
                        using=self._sparse_vector_name,
                        filter=query_filter,
                        limit=prefetch_limit,
                    ),
                ]
            )

        if reference_boost_filter is not None:
            prefetches.append(
                models.Prefetch(
                    query=sparse_query,
                    using=self._sparse_vector_name,
                    filter=reference_boost_filter,
                    limit=max(8, prefetch_limit // 2),
                )
            )
        if issue_focus_filter is not None:
            prefetches.append(
                models.Prefetch(
                    query=sparse_query,
                    using=self._sparse_vector_name,
                    filter=issue_focus_filter,
                    limit=max(12, prefetch_limit),
                )
            )

        candidate_limit = max(top_k * 6, prefetch_limit * max(4, len(query_variants) * 2), 64)

        response = self._qdrant.query_points(
            collection_name=self._collection_name,
            prefetch=prefetches,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=candidate_limit,
            with_payload=True,
        )

        hits = tuple(
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
        hits = self._append_reference_fallback_hits(hits, intent, limit=max(4, top_k))
        direct_records = self._fetch_records_from_hits(hits)
        hits = self._rerank_hits(hits, intent, direct_records)
        hits = self._semantic_rerank_hits(query, hits, direct_records)[:top_k]
        contexts = self._assemble_contexts(hits, intent=intent)
        return RetrievalResult(
            query=query,
            intent=intent,
            hits=hits,
            contexts=contexts,
        )


__all__ = [
    "HybridRetriever",
    "QueryIntent",
    "RetrievalContext",
    "RetrievalResult",
    "LEGAL_ISSUE_ARTICLE_MAP",
    "TERMINATION_ARTICLE_MAP",
    "SearchHit",
    "RetrievedRecord",
    "build_context_block",
    "build_query_variants",
    "dedupe_preserve_order",
    "DEFAULT_MAX_CONTEXT_CHARS",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "DEFAULT_RERANKER_TOP_N",
    "RECORD_SOURCE_QDRANT_PAYLOAD",
    "RECORD_SOURCE_SQLITE",
    "diversify_contexts_by_article",
    "estimate_token_count",
    "format_context_for_prompt",
    "format_intent_summary",
    "load_manifest",
    "parse_reference_values",
    "record_from_qdrant_payload",
    "resolve_record_source",
    "route_query_with_llm",
    "route_query",
    "query_intent_from_metadata",
    "select_contexts_for_prompt",
]
