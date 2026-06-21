from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re

from ..corpus_pipeline import normalize_for_matching
from ..rule_loader import DEFAULT_RULE_CONFIG, RuleConfig
from ..retriever import RetrievalContext


LEGAL_REFERENCE_RE = re.compile(
    r"\b(?:dieu|bo luat|nghi dinh|thong tu|nd-cp|qh\d*)\b|\b\d+\s*/\s*\d{4}\s*/\s*qh\d+\b"
)
TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class UnsupportedTopicRule:
    label: str
    query_all: tuple[str, ...] = ()
    query_any: tuple[str, ...] = ()
    query_not: tuple[str, ...] = ()
    context_all: tuple[str, ...] = ()
    context_any: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScopeGuardDecision:
    out_of_scope: bool
    topic_label: str
    reason: str
    top_score: float
    topic_context_matches: int
    citation_topic_matches: int


@dataclass(frozen=True)
class DomainGuardDecision:
    out_of_domain: bool
    reason: str
    matched_signals: tuple[str, ...]
    meaningful_token_count: int
    refusal_answer: str


UNSUPPORTED_TOPIC_RULES: tuple[UnsupportedTopicRule, ...] = (
    UnsupportedTopicRule(
        label="luong toi thieu vung",
        query_all=("luong toi thieu",),
        query_any=("vung", "vung i", "vung ii", "vung iii", "vung iv"),
        context_all=("luong toi thieu vung",),
        context_any=("vung i", "vung ii", "vung iii", "vung iv", "nam 2026"),
    ),
    UnsupportedTopicRule(
        label="ty le dong bao hiem",
        query_any=(
            "ty le dong bhxh",
            "muc dong bhxh",
            "bhxh bhyt bhtn",
            "bao hiem xa hoi bao hiem y te bao hiem that nghiep",
        ),
        context_any=("ty le dong", "muc dong", "phan tram", "%"),
        context_all=("bao hiem",),
    ),
    UnsupportedTopicRule(
        label="xu phat hanh chinh",
        query_any=(
            "phat hanh chinh",
            "xu phat hanh chinh",
            "muc phat",
            "bi phat bao nhieu",
        ),
        context_any=("xu phat vi pham hanh chinh", "phat tien", "muc phat"),
        context_all=("hanh chinh",),
    ),
    UnsupportedTopicRule(
        label="giay phep lao dong nguoi nuoc ngoai",
        query_all=("giay phep lao dong",),
        query_any=("nguoi nuoc ngoai", "lao dong nuoc ngoai", "foreign worker"),
        context_all=("ho so", "giay phep lao dong"),
        context_any=("nguoi nuoc ngoai", "lao dong nuoc ngoai", "cap giay phep", "gia han", "cap lai"),
    ),
    UnsupportedTopicRule(
        label="kinh phi cong doan",
        query_any=("kinh phi cong doan", "doan phi cong doan", "cong doan"),
        query_all=("ty le",),
        context_any=("kinh phi cong doan", "doan phi cong doan"),
        context_all=("ty le",),
    ),
    UnsupportedTopicRule(
        label="thue thu nhap ca nhan",
        query_any=("thue thu nhap ca nhan", "thue tncn", "tinh thue", "chiu thue"),
        context_any=("thue thu nhap ca nhan", "thue tncn", "thu nhap chiu thue"),
    ),
)
LOW_RETRIEVAL_CONFIDENCE_SCORE = 0.75


REFUSAL_ANSWER_TEMPLATE = """Cau tra loi:
Khong du can cu trong tap van ban da lap chi muc de tra loi cau hoi nay. Cac ngu canh duoc truy xuat khong co co so phap ly truc tiep ve {topic_label}, nen toi khong the dua ra ket luan, muc tien, ty le hoac danh muc ho so cu the.

Tom lai:
- Corpus hien tai khong co du can cu phap ly truc tiep cho noi dung duoc hoi.
- Toi khong dua ra citation khong lien quan tu corpus hien co.
- Can bo sung van ban dung pham vi truoc khi ket luan."""
DEFAULT_OUT_OF_DOMAIN_ANSWER = (
    "Cau hoi nay nam ngoai pham vi tro ly phap luat lao dong Viet Nam. "
    "Vui long hoi ve quan he lao dong, hop dong lao dong, tien luong, "
    "thoi gio lam viec, nghi viec, sa thai, tranh chap lao dong hoac cac van de lien quan."
)


def _domain_guard_config(rule_config: RuleConfig) -> Mapping[str, object]:
    config = getattr(rule_config, "DOMAIN_GUARD", {}) or {}
    return config if isinstance(config, Mapping) else {}


def _configured_keyword_map(rule_config: RuleConfig, section_name: str) -> Mapping[str, Sequence[str]]:
    if section_name == "topics":
        return getattr(rule_config, "TOPIC_KEYWORDS", {})
    if section_name == "issues":
        return getattr(rule_config, "ISSUE_KEYWORDS", {})
    if section_name == "documents":
        return getattr(rule_config, "DOCUMENT_KEYWORDS", {})
    if section_name == "actors":
        return getattr(rule_config, "ACTOR_KEYWORDS", {})
    return {}


def _config_string_sequence(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return ()


def _domain_signal_phrases(rule_config: RuleConfig, config: Mapping[str, object]) -> tuple[str, ...]:
    minimum_chars = max(1, int(config.get("min_domain_phrase_chars") or 3))
    excluded = {
        normalized
        for term in _config_string_sequence(config.get("excluded_domain_terms"))
        if (normalized := normalize_for_matching(term))
    }
    phrases: list[str] = []

    for section_name in _config_string_sequence(
        config.get("keyword_sections") or ("topics", "issues", "documents")
    ):
        for keywords in _configured_keyword_map(rule_config, section_name).values():
            for keyword in keywords:
                normalized = normalize_for_matching(str(keyword))
                if len(normalized) >= minimum_chars and normalized not in excluded:
                    phrases.append(normalized)

    for term in _config_string_sequence(config.get("additional_domain_terms")):
        normalized = normalize_for_matching(term)
        if len(normalized) >= minimum_chars and normalized not in excluded:
            phrases.append(normalized)

    return tuple(dict.fromkeys(phrases))


def _meaningful_tokens(normalized_query: str, *, min_token_length: int) -> tuple[str, ...]:
    return tuple(
        token
        for token in TOKEN_RE.findall(normalized_query)
        if len(token) >= min_token_length
    )


def assess_question_domain(
    question: str,
    *,
    rule_config: RuleConfig = DEFAULT_RULE_CONFIG,
) -> DomainGuardDecision:
    config = _domain_guard_config(rule_config)
    refusal_answer = str(config.get("refusal_answer") or DEFAULT_OUT_OF_DOMAIN_ANSWER)
    if config.get("enabled", True) is False:
        return DomainGuardDecision(
            out_of_domain=False,
            reason="domain_guard_disabled",
            matched_signals=(),
            meaningful_token_count=0,
            refusal_answer=refusal_answer,
        )

    normalized_query = normalize_for_matching(question)
    min_token_length = max(1, int(config.get("min_token_length") or 2))
    min_meaningful_tokens = max(1, int(config.get("min_meaningful_tokens") or 2))
    tokens = _meaningful_tokens(normalized_query, min_token_length=min_token_length)

    if bool(config.get("direct_legal_reference_signals", True)) and LEGAL_REFERENCE_RE.search(
        normalized_query
    ):
        return DomainGuardDecision(
            out_of_domain=False,
            reason="direct_legal_reference_signal",
            matched_signals=("legal_reference",),
            meaningful_token_count=len(tokens),
            refusal_answer=refusal_answer,
        )

    signals = tuple(
        phrase for phrase in _domain_signal_phrases(rule_config, config) if phrase in normalized_query
    )
    min_domain_signals = max(1, int(config.get("min_domain_signals") or 1))
    if len(signals) >= min_domain_signals:
        return DomainGuardDecision(
            out_of_domain=False,
            reason="domain_signal_matched",
            matched_signals=signals,
            meaningful_token_count=len(tokens),
            refusal_answer=refusal_answer,
        )

    reason = (
        "too_few_meaningful_tokens"
        if len(tokens) < min_meaningful_tokens
        else "no_domain_signal"
    )
    return DomainGuardDecision(
        out_of_domain=True,
        reason=reason,
        matched_signals=signals,
        meaningful_token_count=len(tokens),
        refusal_answer=refusal_answer,
    )


def _contains_all(text: str, terms: Sequence[str]) -> bool:
    return all(term in text for term in terms)


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    return not terms or any(term in text for term in terms)


def _contains_none(text: str, terms: Sequence[str]) -> bool:
    return not any(term in text for term in terms)


def _rule_matches_query(rule: UnsupportedTopicRule, normalized_query: str) -> bool:
    return (
        _contains_all(normalized_query, rule.query_all)
        and _contains_any(normalized_query, rule.query_any)
        and _contains_none(normalized_query, rule.query_not)
    )


def _context_text(context: RetrievalContext) -> str:
    payload = context.payload
    topic_values = " ".join(str(value) for value in payload.get("topic") or ())
    issue_values = " ".join(str(value) for value in payload.get("issue_type") or ())
    return normalize_for_matching(
        " ".join(
            str(value or "")
            for value in (
                context.citation_text,
                context.text,
                payload.get("retrieval_text"),
                payload.get("article_title"),
                payload.get("heading"),
                payload.get("section_heading"),
                topic_values,
                issue_values,
            )
        )
    )


def _citation_text(context: RetrievalContext) -> str:
    return normalize_for_matching(
        " ".join(
            str(value or "")
            for value in (
                context.citation_text,
                context.payload.get("article_title"),
                context.payload.get("heading"),
            )
        )
    )


def _context_supports_rule(rule: UnsupportedTopicRule, context: RetrievalContext) -> bool:
    text = _context_text(context)
    return _contains_all(text, rule.context_all) and _contains_any(text, rule.context_any)


def _citation_matches_rule_topic(rule: UnsupportedTopicRule, context: RetrievalContext) -> bool:
    text = _citation_text(context)
    topic_terms = (*rule.context_all, *rule.context_any)
    return any(term and term in text for term in topic_terms)


def _top_score(contexts: Sequence[RetrievalContext]) -> float:
    scores: list[float] = []
    for context in contexts:
        try:
            scores.append(float(context.payload.get("final_score") or context.score))
        except (TypeError, ValueError):
            scores.append(float(context.score))
    return max(scores, default=0.0)


def assess_scope(
    question: str,
    contexts: Sequence[RetrievalContext],
) -> ScopeGuardDecision:
    normalized_query = normalize_for_matching(question)
    top_score = _top_score(contexts)
    matched_rules = [
        rule for rule in UNSUPPORTED_TOPIC_RULES if _rule_matches_query(rule, normalized_query)
    ]
    if not matched_rules:
        return ScopeGuardDecision(
            out_of_scope=False,
            topic_label="",
            reason="no_unsupported_topic_rule_matched",
            top_score=top_score,
            topic_context_matches=0,
            citation_topic_matches=0,
        )

    topic_context_matches = 0
    citation_topic_matches = 0
    for rule in matched_rules:
        topic_context_matches += sum(1 for context in contexts if _context_supports_rule(rule, context))
        citation_topic_matches += sum(
            1 for context in contexts if _citation_matches_rule_topic(rule, context)
        )

    if topic_context_matches > 0 and (
        citation_topic_matches > 0 or top_score >= LOW_RETRIEVAL_CONFIDENCE_SCORE
    ):
        return ScopeGuardDecision(
            out_of_scope=False,
            topic_label=matched_rules[0].label,
            reason="retrieved_context_contains_direct_topic_evidence",
            top_score=top_score,
            topic_context_matches=topic_context_matches,
            citation_topic_matches=citation_topic_matches,
        )

    return ScopeGuardDecision(
        out_of_scope=True,
        topic_label=matched_rules[0].label,
        reason=(
            "unsupported_topic_low_retrieval_confidence"
            if topic_context_matches
            else "unsupported_topic_without_direct_retrieved_evidence"
        ),
        top_score=top_score,
        topic_context_matches=topic_context_matches,
        citation_topic_matches=citation_topic_matches,
    )


def build_scope_refusal_payload(decision: ScopeGuardDecision) -> dict[str, object]:
    topic_label = decision.topic_label or "chu de phap ly duoc hoi"
    return {
        "answer": REFUSAL_ANSWER_TEMPLATE.format(topic_label=topic_label),
        "legal_basis": [],
        "evidence_quotes": [],
        "insufficient_context": True,
        "notes": "",
    }


__all__ = [
    "DEFAULT_OUT_OF_DOMAIN_ANSWER",
    "REFUSAL_ANSWER_TEMPLATE",
    "LOW_RETRIEVAL_CONFIDENCE_SCORE",
    "DomainGuardDecision",
    "ScopeGuardDecision",
    "UNSUPPORTED_TOPIC_RULES",
    "UnsupportedTopicRule",
    "assess_question_domain",
    "assess_scope",
    "build_scope_refusal_payload",
]
