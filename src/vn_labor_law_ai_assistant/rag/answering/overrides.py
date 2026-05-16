from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ...corpus_pipeline import normalize_for_matching
from ...rule_loader import AnswerOverrideRule, load_answer_override_rules
from ...retriever import RetrievalContext
from .citation_guard import (
    citation_for_context,
    combined_context_text,
    contains_percent_value,
    extract_evidence_sentence,
    first_context_with_terms,
)
from .schema import EvidenceQuote

@dataclass(frozen=True)
class ContextualAnswerOverride:
    answer: str
    evidence_quote: EvidenceQuote
    notes: str


def _contains_all(text: str, terms: Sequence[str]) -> bool:
    return all(term in text for term in terms)


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    return not terms or any(term in text for term in terms)


def _contains_none(text: str, terms: Sequence[str]) -> bool:
    return not any(term in text for term in terms)


def _rule_matches(
    rule: AnswerOverrideRule,
    *,
    normalized_question: str,
    normalized_context: str,
    raw_context: str,
) -> bool:
    if not rule.enabled:
        return False
    if not _contains_all(normalized_question, rule.question_all):
        return False
    if not _contains_any(normalized_question, rule.question_any):
        return False
    if not _contains_none(normalized_question, rule.question_not):
        return False
    if not _contains_all(normalized_context, rule.context_all):
        return False
    if not _contains_any(normalized_context, rule.context_any):
        return False
    if not _contains_none(normalized_context, rule.context_not):
        return False
    if rule.context_percent_any and not any(
        contains_percent_value(raw_context, value)
        for value in rule.context_percent_any
    ):
        return False
    return True


def _answer_for_rule(rule: AnswerOverrideRule, normalized_context: str) -> str:
    answer = rule.answer
    if rule.append_text and _contains_all(normalized_context, rule.append_if_context_all):
        answer += rule.append_text
    return answer


def contextual_answer_override(
    question: str,
    contexts: Sequence[RetrievalContext],
) -> ContextualAnswerOverride | None:
    normalized_question = normalize_for_matching(question)
    raw_context = combined_context_text(contexts)
    normalized_context = normalize_for_matching(raw_context)

    for rule in load_answer_override_rules():
        if not _rule_matches(
            rule,
            normalized_question=normalized_question,
            normalized_context=normalized_context,
            raw_context=raw_context,
        ):
            continue

        context = first_context_with_terms(contexts, rule.quote_terms)
        if context is None and contexts:
            context = contexts[0]
        if context is None:
            return None

        quote = extract_evidence_sentence(context.text, rule.quote_terms)
        return ContextualAnswerOverride(
            answer=_answer_for_rule(rule, normalized_context),
            evidence_quote=EvidenceQuote(
                citation=citation_for_context(context, rule.citation_terms),
                quote=quote,
            ),
            notes=rule.notes,
        )
    return None


__all__ = [
    "ContextualAnswerOverride",
    "contextual_answer_override",
]
