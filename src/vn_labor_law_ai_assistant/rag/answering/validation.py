from __future__ import annotations

import re
from typing import Sequence

from ...corpus_pipeline import normalize_for_matching
from ...retriever import RetrievalContext, dedupe_preserve_order
from .citation_guard import (
    canonicalize_citation,
    normalize_citation_surface,
)
from .prompt import build_allowed_citations
from .schema import AnswerValidationResult, ParsedAnswer


ARTICLE_REF_RE = re.compile(r"\b(?:dieu|điều)\s+(?P<article>\d+[a-z]?)", re.IGNORECASE)
INSUFFICIENT_CONTEXT_HINTS = (
    "khong du can cu trong du lieu hien co",
    "chua du can cu",
    "khong du can cu",
)


def extract_article_numbers(text: str) -> tuple[str, ...]:
    normalized = normalize_for_matching(text)
    return dedupe_preserve_order(
        tuple(match.group("article").lower() for match in ARTICLE_REF_RE.finditer(normalized))
    )


def article_numbers_from_contexts(contexts: Sequence[RetrievalContext]) -> tuple[str, ...]:
    article_numbers: list[str] = []
    for context in contexts:
        payload_article = str(context.payload.get("article_number") or "").strip().lower()
        if payload_article:
            article_numbers.append(payload_article)
        article_numbers.extend(extract_article_numbers(context.citation_text))
        article_numbers.extend(extract_article_numbers(context.text))
        retrieval_text = str(context.payload.get("retrieval_text") or "")
        if retrieval_text:
            article_numbers.extend(extract_article_numbers(retrieval_text))
        for citation in context.matched_citations:
            article_numbers.extend(extract_article_numbers(citation))
    return dedupe_preserve_order(tuple(article_numbers))


def citation_rank_lookup(contexts: Sequence[RetrievalContext]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for context in contexts:
        try:
            rank = int(context.payload.get("normative_rank") or 0)
        except (TypeError, ValueError):
            rank = 0
        if rank <= 0:
            continue
        for citation in dedupe_preserve_order((*context.matched_citations, context.citation_text)):
            if citation:
                lookup[citation] = rank
    return lookup


def answer_mentions_citation(answer: str, citation: str) -> bool:
    normalized_answer = normalize_for_matching(answer)
    normalized_citation = normalize_for_matching(citation)
    if normalized_citation and normalized_citation in normalized_answer:
        return True
    citation_surface = normalize_citation_surface(citation)
    return bool(citation_surface and citation_surface in normalize_citation_surface(answer))


def validate_grounded_answer(
    parsed: ParsedAnswer,
    contexts: Sequence[RetrievalContext],
) -> AnswerValidationResult:
    allowed_citations = build_allowed_citations(contexts)
    citation_ranks = citation_rank_lookup(contexts)
    warnings: list[str] = []

    has_required_citation = True
    if contexts and not parsed.insufficient_context:
        has_required_citation = bool(parsed.legal_basis) and any(
            answer_mentions_citation(parsed.answer, citation)
            for citation in parsed.legal_basis
        )
        if not has_required_citation:
            warnings.append("Answer does not include a validated inline citation.")

    unretrieved_citations = tuple(
        citation
        for citation in parsed.legal_basis
        if canonicalize_citation(citation, allowed_citations) is None
    )
    citations_allowed = not unretrieved_citations
    if unretrieved_citations:
        warnings.append("Answer cites legal bases that were not retrieved.")

    allowed_articles = set(article_numbers_from_contexts(contexts))
    answer_articles = set(extract_article_numbers(parsed.answer))
    basis_articles = {
        article
        for citation in parsed.legal_basis
        for article in extract_article_numbers(citation)
    }
    unsupported_article_numbers = tuple(
        sorted((answer_articles | basis_articles) - allowed_articles, key=lambda value: (len(value), value))
    )
    if unsupported_article_numbers:
        warnings.append("Answer mentions article numbers not present in retrieved contexts.")

    available_ranks = [
        int(context.payload.get("normative_rank") or 0)
        for context in contexts
        if str(context.payload.get("normative_rank") or "").isdigit()
    ]
    cited_ranks = [
        citation_ranks[citation]
        for citation in parsed.legal_basis
        if citation in citation_ranks
    ]
    ignores_higher_rank_context = False
    if available_ranks and cited_ranks:
        highest_available = min(available_ranks)
        highest_cited = min(cited_ranks)
        ignores_higher_rank_context = highest_cited > highest_available
        if ignores_higher_rank_context:
            warnings.append("Answer cites lower-rank guidance without citing available higher-rank law first.")

    has_uncertainty_when_insufficient = True
    if parsed.insufficient_context:
        normalized_answer = normalize_for_matching(parsed.answer)
        has_uncertainty_when_insufficient = any(
            hint in normalized_answer for hint in INSUFFICIENT_CONTEXT_HINTS
        )
        if not has_uncertainty_when_insufficient:
            warnings.append("Insufficient-context answer does not say that available data is insufficient.")

    passed = (
        has_required_citation
        and citations_allowed
        and not unsupported_article_numbers
        and not ignores_higher_rank_context
        and has_uncertainty_when_insufficient
    )
    return AnswerValidationResult(
        passed=passed,
        has_required_citation=has_required_citation,
        citations_allowed=citations_allowed,
        unsupported_article_numbers=unsupported_article_numbers,
        unretrieved_citations=unretrieved_citations,
        ignores_higher_rank_context=ignores_higher_rank_context,
        has_uncertainty_when_insufficient=has_uncertainty_when_insufficient,
        warnings=tuple(warnings),
    )


__all__ = [
    "ARTICLE_REF_RE",
    "INSUFFICIENT_CONTEXT_HINTS",
    "answer_mentions_citation",
    "article_numbers_from_contexts",
    "extract_article_numbers",
    "validate_grounded_answer",
]
