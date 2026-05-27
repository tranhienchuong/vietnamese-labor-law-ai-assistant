from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

QualityCheckStatus = bool | Literal["not_applicable"]

ANSWER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "legal_basis": {
            "type": "array",
            "items": {"type": "string"},
        },
        "evidence_quotes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "citation": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["citation", "quote"],
                "additionalProperties": False,
            },
        },
        "insufficient_context": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": ["answer", "legal_basis", "evidence_quotes", "insufficient_context", "notes"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class EvidenceQuote:
    citation: str
    quote: str


@dataclass(frozen=True)
class ParsedAnswer:
    answer: str
    legal_basis: tuple[str, ...]
    evidence_quotes: tuple[EvidenceQuote, ...]
    insufficient_context: bool
    notes: str
    raw_content: str


@dataclass(frozen=True)
class AnswerValidationResult:
    passed: bool
    has_required_citation: bool
    citations_allowed: bool
    unsupported_article_numbers: tuple[str, ...]
    unretrieved_citations: tuple[str, ...]
    ignores_higher_rank_context: bool
    has_uncertainty_when_insufficient: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class AnswerQualityValidationResult:
    passed: bool
    applied_answer_intent: str
    direct_answer_present: QualityCheckStatus
    required_legal_rule_present: QualityCheckStatus
    low_information_quotes_count: int
    low_information_quotes: tuple[str, ...]
    numeric_answer_present: QualityCheckStatus
    yes_no_answer_present: QualityCheckStatus
    conditions_listed: QualityCheckStatus
    exception_answer_present: QualityCheckStatus
    no_article_title_only_answer: QualityCheckStatus
    all_legal_claims_have_citations: QualityCheckStatus
    warnings: tuple[str, ...]


__all__ = [
    "AnswerQualityValidationResult",
    "AnswerValidationResult",
    "ANSWER_JSON_SCHEMA",
    "EvidenceQuote",
    "ParsedAnswer",
    "QualityCheckStatus",
]
