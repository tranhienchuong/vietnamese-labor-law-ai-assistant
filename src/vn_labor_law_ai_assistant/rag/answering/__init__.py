from __future__ import annotations

from .citation_guard import (
    EVIDENCE_SENTENCE_SPLIT_RE,
    PARENTHETICAL_CITATION_RE,
    answer_mentions_any_citation,
    canonicalize_citation,
    citation_for_context,
    citation_overlap_matches,
    combined_context_text,
    contains_percent_value,
    context_matches_citation,
    extract_evidence_sentence,
    first_context_with_terms,
    normalize_citation_surface,
    normalize_quote_surface,
    quote_appears_in_text,
    quote_supported_by_context,
    sanitize_evidence_quotes,
    sanitize_legal_basis,
)
from .formatter import LEADING_QUESTION_SECTION_RE, format_answer_for_user
from .overrides import ContextualAnswerOverride, contextual_answer_override
from .parser import extract_json_candidate, parse_answer_payload
from .prompt import ANSWER_FEW_SHOT_PROMPT, SYSTEM_PROMPT, build_allowed_citations, build_messages
from .schema import ANSWER_JSON_SCHEMA, EvidenceQuote, ParsedAnswer

__all__ = [
    "ANSWER_FEW_SHOT_PROMPT",
    "ANSWER_JSON_SCHEMA",
    "ContextualAnswerOverride",
    "EvidenceQuote",
    "EVIDENCE_SENTENCE_SPLIT_RE",
    "LEADING_QUESTION_SECTION_RE",
    "PARENTHETICAL_CITATION_RE",
    "ParsedAnswer",
    "SYSTEM_PROMPT",
    "answer_mentions_any_citation",
    "build_allowed_citations",
    "build_messages",
    "canonicalize_citation",
    "citation_for_context",
    "citation_overlap_matches",
    "combined_context_text",
    "contains_percent_value",
    "context_matches_citation",
    "contextual_answer_override",
    "extract_evidence_sentence",
    "extract_json_candidate",
    "first_context_with_terms",
    "format_answer_for_user",
    "normalize_citation_surface",
    "normalize_quote_surface",
    "parse_answer_payload",
    "quote_appears_in_text",
    "quote_supported_by_context",
    "sanitize_evidence_quotes",
    "sanitize_legal_basis",
]
