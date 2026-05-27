from __future__ import annotations

import re
from typing import Sequence

from ...corpus_pipeline import normalize_for_matching
from ...indexing import extract_legal_hint_tokens
from ...retriever import RetrievalContext, dedupe_preserve_order
from .prompt import build_allowed_citations
from .schema import EvidenceQuote

PARENTHETICAL_CITATION_RE = re.compile(r"\([^)]*\)")
EVIDENCE_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:])\s+|\n+")


def answer_mentions_any_citation(answer: str, legal_basis: Sequence[str]) -> bool:
    normalized_answer = normalize_for_matching(answer)
    return any(
        normalize_for_matching(citation) in normalized_answer
        for citation in legal_basis
        if str(citation or "").strip()
    )


def normalize_citation_surface(text: str) -> str:
    without_parenthetical = PARENTHETICAL_CITATION_RE.sub(" ", text)
    normalized = normalize_for_matching(without_parenthetical)
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def citation_overlap_matches(allowed_citation: str, requested_citation: str) -> bool:
    allowed_normalized = normalize_for_matching(allowed_citation)
    requested_normalized = normalize_for_matching(requested_citation)
    allowed_surface = normalize_citation_surface(allowed_citation)
    requested_surface = normalize_citation_surface(requested_citation)
    if not allowed_normalized or not requested_normalized:
        return False
    if allowed_normalized == requested_normalized:
        return True
    if allowed_surface and allowed_surface == requested_surface:
        return True

    allowed_tokens = set(extract_legal_hint_tokens(allowed_citation))
    requested_tokens = set(extract_legal_hint_tokens(requested_citation))
    if allowed_tokens and requested_tokens:
        if not (
            allowed_tokens.issubset(requested_tokens)
            or requested_tokens.issubset(allowed_tokens)
        ):
            return False
        return (
            allowed_normalized in requested_normalized
            or requested_normalized in allowed_normalized
            or (allowed_surface and allowed_surface in requested_surface)
            or (requested_surface and requested_surface in allowed_surface)
        )

    return (
        allowed_normalized in requested_normalized
        or requested_normalized in allowed_normalized
        or (allowed_surface and allowed_surface in requested_surface)
        or (requested_surface and requested_surface in allowed_surface)
    )


def canonicalize_citation(
    requested_citation: str,
    allowed_citations: Sequence[str],
) -> str | None:
    requested_normalized = normalize_for_matching(requested_citation)
    requested_tokens = tuple(extract_legal_hint_tokens(requested_citation))

    for allowed_citation in allowed_citations:
        if normalize_for_matching(allowed_citation) == requested_normalized:
            return allowed_citation

    fuzzy_matches: list[tuple[int, int, int, str]] = []
    for index, allowed_citation in enumerate(allowed_citations):
        if not citation_overlap_matches(allowed_citation, requested_citation):
            continue

        allowed_tokens = tuple(extract_legal_hint_tokens(allowed_citation))
        fuzzy_matches.append(
            (
                abs(len(allowed_tokens) - len(requested_tokens)),
                abs(len(normalize_for_matching(allowed_citation)) - len(requested_normalized)),
                index,
                allowed_citation,
            )
        )

    if not fuzzy_matches:
        return None

    fuzzy_matches.sort()
    return fuzzy_matches[0][3]


def sanitize_legal_basis(
    legal_basis: Sequence[str] | str | None,
    contexts: Sequence[RetrievalContext],
) -> tuple[str, ...]:
    if not legal_basis:
        return ()

    allowed_citations = build_allowed_citations(contexts)

    if isinstance(legal_basis, str):
        requested = [legal_basis]
    else:
        requested = [str(item) for item in legal_basis]

    filtered: list[str] = []
    for citation in requested:
        canonical_citation = canonicalize_citation(citation, allowed_citations)
        if canonical_citation is None:
            continue
        filtered.append(canonical_citation)
    return dedupe_preserve_order(filtered)


def normalize_quote_surface(text: str) -> str:
    normalized = normalize_for_matching(text)
    return re.sub(r"[^a-z0-9%]+", " ", normalized).strip()


def context_matches_citation(context: RetrievalContext, citation: str) -> bool:
    if citation_overlap_matches(context.citation_text, citation):
        return True
    return any(
        citation_overlap_matches(matched_citation, citation)
        for matched_citation in context.matched_citations
    )


def quote_appears_in_text(quote: str, text: str) -> bool:
    normalized_quote = normalize_quote_surface(quote)
    if len(normalized_quote) < 8:
        return False
    normalized_text = normalize_quote_surface(text)
    if normalized_quote in normalized_text:
        return True

    segments = [
        normalize_quote_surface(segment)
        for segment in EVIDENCE_SENTENCE_SPLIT_RE.split(quote)
        if segment.strip()
    ]
    meaningful_segments = [segment for segment in segments if len(segment) >= 8]
    return bool(meaningful_segments) and all(
        segment in normalized_text for segment in meaningful_segments
    )


def quote_supported_by_context(
    *,
    citation: str,
    quote: str,
    contexts: Sequence[RetrievalContext],
) -> bool:
    def context_supports_quote(context: RetrievalContext) -> bool:
        retrieval_text = str(context.payload.get("retrieval_text") or "")
        return quote_appears_in_text(quote, context.text) or (
            bool(retrieval_text) and quote_appears_in_text(quote, retrieval_text)
        )

    matching_contexts = [
        context for context in contexts if context_matches_citation(context, citation)
    ]
    if matching_contexts:
        return any(context_supports_quote(context) for context in matching_contexts)
    return any(context_supports_quote(context) for context in contexts)


def sanitize_evidence_quotes(
    evidence_quotes: object,
    contexts: Sequence[RetrievalContext],
) -> tuple[EvidenceQuote, ...]:
    if not isinstance(evidence_quotes, list):
        return ()

    allowed_citations = build_allowed_citations(contexts)
    sanitized: list[EvidenceQuote] = []
    seen: set[tuple[str, str]] = set()
    for item in evidence_quotes:
        if not isinstance(item, dict):
            continue
        raw_citation = str(item.get("citation") or "").strip()
        quote = str(item.get("quote") or "").strip()
        if not raw_citation or not quote:
            continue
        citation = canonicalize_citation(raw_citation, allowed_citations)
        if citation is None:
            continue
        if not quote_supported_by_context(citation=citation, quote=quote, contexts=contexts):
            continue
        key = (citation, normalize_quote_surface(quote))
        if key in seen:
            continue
        seen.add(key)
        sanitized.append(EvidenceQuote(citation=citation, quote=quote))
    return tuple(sanitized)


def combined_context_text(contexts: Sequence[RetrievalContext]) -> str:
    return "\n".join(context.text for context in contexts)


def contains_percent_value(text: str, value: int) -> bool:
    return re.search(rf"(?<!\d){value}\s*%", text) is not None


def first_context_with_terms(
    contexts: Sequence[RetrievalContext],
    required_terms: Sequence[str],
) -> RetrievalContext | None:
    for context in contexts:
        normalized = normalize_for_matching(context.text)
        if all(term in normalized for term in required_terms):
            return context
    return None


def citation_for_context(
    context: RetrievalContext,
    preferred_terms: Sequence[str] = (),
) -> str:
    citations = dedupe_preserve_order((*context.matched_citations, context.citation_text))
    normalized_terms = tuple(normalize_for_matching(term) for term in preferred_terms)

    for citation in citations:
        normalized_citation = normalize_for_matching(citation)
        if normalized_terms and all(term in normalized_citation for term in normalized_terms):
            return citation

    for citation in citations:
        if citation:
            return citation

    return context.citation_text


def extract_evidence_sentence(text: str, preferred_terms: Sequence[str]) -> str:
    normalized_terms = tuple(normalize_for_matching(term) for term in preferred_terms)
    candidates = [
        sentence.strip()
        for sentence in EVIDENCE_SENTENCE_SPLIT_RE.split(text)
        if sentence.strip()
    ]
    for sentence in candidates:
        normalized_sentence = normalize_for_matching(sentence)
        if all(term in normalized_sentence for term in normalized_terms):
            return sentence[:500]
    for sentence in candidates:
        normalized_sentence = normalize_for_matching(sentence)
        if any(term in normalized_sentence for term in normalized_terms):
            return sentence[:500]
    return text.strip()[:500]


__all__ = [
    "EVIDENCE_SENTENCE_SPLIT_RE",
    "PARENTHETICAL_CITATION_RE",
    "answer_mentions_any_citation",
    "canonicalize_citation",
    "citation_for_context",
    "citation_overlap_matches",
    "combined_context_text",
    "contains_percent_value",
    "context_matches_citation",
    "extract_evidence_sentence",
    "first_context_with_terms",
    "normalize_citation_surface",
    "normalize_quote_surface",
    "quote_appears_in_text",
    "quote_supported_by_context",
    "sanitize_evidence_quotes",
    "sanitize_legal_basis",
]
