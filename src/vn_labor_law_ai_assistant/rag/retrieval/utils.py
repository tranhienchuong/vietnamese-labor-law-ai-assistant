from __future__ import annotations

from typing import Sequence

from ...core.config import load_settings
from ...corpus_pipeline import normalize_for_matching
from ...heuristic_router import contains_normalized_phrase, dedupe_preserve_order
from .constants import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_CONTEXT_TOKENS,
    POINT_REF_ORDER,
    RECORD_SOURCE_SQLITE,
    RULE_CONFIG,
    SUPPORTED_RECORD_SOURCES,
    TOKEN_ESTIMATE_RE,
)
from .models import RetrievedRecord, RetrievalContext


def env_flag(name: str, default: bool) -> bool:
    settings = load_settings()
    if name == "QUERY_ROUTER_ENABLED":
        return settings.query_router_enabled
    if name == "QUERY_ROUTER_FALLBACK_TO_HEURISTIC":
        return settings.query_router_fallback_to_heuristic
    if name == "ENABLE_ARTICLE_SIBLING_CONTEXTS":
        return settings.enable_article_sibling_contexts
    return default


def resolve_record_source(manifest: dict[str, object]) -> str:
    settings = load_settings()
    configured = (
        settings.retriever_record_source.strip()
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

    normalized = point_ref.strip().lower().replace("\u00c4\u2018", "\u0111")
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
    point_refs = tuple(str(value) for value in record.payload.get("point_refs") or [])
    point_ref = str(record.payload.get("point_ref") or (point_refs[0] if point_refs else ""))
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
        RULE_CONFIG.ENUMERATION_PARENT_CONTEXT_HINTS,
    )


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


__all__ = [
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
    "point_ref_sort_key",
    "record_from_qdrant_payload",
    "record_reference_sort_key",
    "resolve_record_source",
    "select_contexts_for_prompt",
]
