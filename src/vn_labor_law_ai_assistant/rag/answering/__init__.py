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
from .generation import (
    DEFAULT_ANSWER_CONTEXTS,
    GroundedAnswerResult,
    build_extractive_answer_payload,
    generate_grounded_answer,
)
from .overrides import ContextualAnswerOverride, contextual_answer_override
from .parser import extract_json_candidate, parse_answer_payload
from .quality import is_low_information_quote, validate_answer_quality
from .prompt import (
    ANSWER_FEW_SHOT_PROMPT,
    SYSTEM_PROMPT,
    build_allowed_citations,
    build_answer_context_block,
    build_messages,
    format_answer_context_for_prompt,
    order_contexts_for_answer,
)
from .schema import ANSWER_JSON_SCHEMA, AnswerQualityValidationResult, AnswerValidationResult, EvidenceQuote, ParsedAnswer
from .synthesis import classify_answer_intent, select_contexts_for_grounded_generation
from .validation import (
    answer_mentions_citation,
    article_numbers_from_contexts,
    extract_article_numbers,
    validate_grounded_answer,
)

__all__ = [
    "ANSWER_FEW_SHOT_PROMPT",
    "ANSWER_JSON_SCHEMA",
    "AnswerQualityValidationResult",
    "AnswerValidationResult",
    "ContextualAnswerOverride",
    "DEFAULT_ANSWER_CONTEXTS",
    "EvidenceQuote",
    "EVIDENCE_SENTENCE_SPLIT_RE",
    "GroundedAnswerResult",
    "LEADING_QUESTION_SECTION_RE",
    "PARENTHETICAL_CITATION_RE",
    "ParsedAnswer",
    "SYSTEM_PROMPT",
    "answer_mentions_citation",
    "answer_mentions_any_citation",
    "article_numbers_from_contexts",
    "build_allowed_citations",
    "build_answer_context_block",
    "build_extractive_answer_payload",
    "build_messages",
    "canonicalize_citation",
    "citation_for_context",
    "citation_overlap_matches",
    "classify_answer_intent",
    "combined_context_text",
    "contains_percent_value",
    "context_matches_citation",
    "contextual_answer_override",
    "extract_evidence_sentence",
    "extract_article_numbers",
    "extract_json_candidate",
    "first_context_with_terms",
    "format_answer_context_for_prompt",
    "format_answer_for_user",
    "generate_grounded_answer",
    "is_low_information_quote",
    "normalize_citation_surface",
    "normalize_quote_surface",
    "order_contexts_for_answer",
    "parse_answer_payload",
    "quote_appears_in_text",
    "quote_supported_by_context",
    "sanitize_evidence_quotes",
    "sanitize_legal_basis",
    "select_contexts_for_grounded_generation",
    "validate_answer_quality",
    "validate_grounded_answer",
]
