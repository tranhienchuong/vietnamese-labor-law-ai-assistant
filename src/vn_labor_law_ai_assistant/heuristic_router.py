from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

from .corpus_pipeline import normalize_for_matching
from .rule_loader import (
    DEFAULT_RULE_CONFIG,
    DirectReferenceRule,
    LegalReference,
    RuleBasedQueryExpansion,
    RuleConfig,
)


ARTICLE_REF_RE = re.compile(r"\bdieu\s+(?P<value>\d+[a-z]?)")
CLAUSE_REF_RE = re.compile(r"\bkhoan\s+(?P<value>\d+)")
POINT_REF_RE = re.compile(r"\bdiem\s+(?P<value>[a-z](?:\.\d+)?)")
YEAR_COUNT_RE = re.compile(r"\b\d+\s+nam\b")

RULE_CONFIG = DEFAULT_RULE_CONFIG

_RULE_CONFIG_EXPORTS = frozenset(
    """
    ACTOR_KEYWORDS BENEFIT_COMPUTATION_QUERY_HINTS CALCULATION_CONTEXT_HINTS
    CALCULATION_QUERY_HINTS DELEGATION_CONTEXT_HINTS DOCUMENT_KEYWORDS
    DIRECT_REFERENCE_RULES
    ENUMERATION_PARENT_CONTEXT_HINTS ENUMERATION_QUERY_HINTS GENERIC_ACTOR_FILTERS
    IMPLEMENTATION_DETAIL_HINTS ISSUE_KEYWORDS LEGAL_HIGH_PRECISION_QUERY_RULES
    LEGAL_ISSUE_ARTICLE_MAP LEGAL_ISSUE_QUERY_HINTS LEGAL_SOFT_HINT_QUERY_RULES
    LEGAL_TOPIC_ARTICLE_MAP MATERNITY_CONTEXT_HINTS MAX_ENUMERATION_CONTEXT_RECORDS
    NO_NOTICE_QUERY_HINTS QUERY_TYPE_KEYWORDS RETIREMENT_CONTEXT_HINTS
    TERMINATION_ARTICLE_EXCLUDED_HINTS TERMINATION_ARTICLE_ISSUE_HINTS
    TERMINATION_ARTICLE_MAP TERMINATION_ARTICLE_QUERY_RULES TERMINATION_ARTICLE_TOPIC_HINTS
    TERMINATION_BENEFIT_CONTEXT_HINTS TERMINATION_QUERY_HINTS TERMINATION_SECTION_HINTS
    TOPIC_KEYWORDS
    """.split()
)


def __getattr__(name: str) -> object:
    if name in _RULE_CONFIG_EXPORTS:
        return getattr(RULE_CONFIG, name)
    raise AttributeError(name)


@dataclass(frozen=True)
class RuleBasedRoutingResult:
    inferred_articles: tuple[str, ...]
    force_reference_articles: tuple[str, ...]
    forced_references: tuple[LegalReference, ...]
    direct_reference_rule_names: tuple[str, ...]
    topics: tuple[str, ...]
    issues: tuple[str, ...]
    expansions: tuple[str, ...]


@dataclass(frozen=True)
class QueryIntent:
    raw_query: str
    normalized_query: str
    actor_filters: tuple[str, ...]
    topic_filters: tuple[str, ...]
    issue_filters: tuple[str, ...]
    document_filters: tuple[str, ...]
    article_numbers: tuple[str, ...] = ()
    inferred_article_numbers: tuple[str, ...] = ()
    force_reference_article_numbers: tuple[str, ...] = ()
    forced_references: tuple[LegalReference, ...] = ()
    matched_direct_reference_rules: tuple[str, ...] = ()
    clause_refs: tuple[str, ...] = ()
    point_refs: tuple[str, ...] = ()
    query_expansions: tuple[str, ...] = ()
    query_types: tuple[str, ...] = ()

    @property
    def all_article_numbers(self) -> tuple[str, ...]:
        return dedupe_preserve_order(
            (
                *self.article_numbers,
                *self.force_reference_article_numbers,
                *self.inferred_article_numbers,
            )
        )

    @property
    def article_number(self) -> str | None:
        article_numbers = self.all_article_numbers
        return article_numbers[0] if article_numbers else None

    @property
    def clause_ref(self) -> str | None:
        return self.clause_refs[0] if self.clause_refs else None

    @property
    def point_ref(self) -> str | None:
        return self.point_refs[0] if self.point_refs else None

    @property
    def legal_reference_filters(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        filters: list[tuple[str, tuple[str, ...]]] = []
        article_numbers = self.all_article_numbers
        if article_numbers:
            filters.append(("article_number", article_numbers))
        if self.clause_refs:
            filters.append(("clause_ref", self.clause_refs))
        if self.point_refs:
            filters.append(("point_ref", self.point_refs))
        return tuple(filters)

    @property
    def explicit_legal_reference_filters(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        filters: list[tuple[str, tuple[str, ...]]] = []
        if self.article_numbers:
            filters.append(("article_number", self.article_numbers))
        if self.clause_refs:
            filters.append(("clause_ref", self.clause_refs))
        if self.point_refs:
            filters.append(("point_ref", self.point_refs))
        return tuple(filters)


def dedupe_preserve_order(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def parse_reference_values(pattern: re.Pattern[str], normalized_query: str) -> tuple[str, ...]:
    return dedupe_preserve_order(
        tuple(match.group("value").lower() for match in pattern.finditer(normalized_query))
    )


def collect_keyword_matches(
    normalized_query: str,
    mapping: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    matches = [
        label
        for label, keywords in mapping.items()
        if any(keyword in normalized_query for keyword in keywords)
    ]
    return tuple(matches)


def contains_normalized_phrase(normalized_text: str, phrases: Sequence[str]) -> bool:
    return any(phrase in normalized_text for phrase in phrases)


def rule_matches_normalized_query(
    normalized_query: str,
    rule: RuleBasedQueryExpansion,
) -> bool:
    if not contains_normalized_phrase(normalized_query, rule.phrases):
        return False
    if rule.context_phrases and not contains_normalized_phrase(
        normalized_query,
        rule.context_phrases,
    ):
        return False
    if rule.excluded_phrases and contains_normalized_phrase(
        normalized_query,
        rule.excluded_phrases,
    ):
        return False
    return True


def direct_reference_rule_matches(
    normalized_query: str,
    rule: DirectReferenceRule,
    *,
    query_types: Sequence[str] = (),
    document_filters: Sequence[str] = (),
) -> bool:
    if not rule.references:
        return False
    if rule.excluded_phrases and contains_normalized_phrase(
        normalized_query,
        rule.excluded_phrases,
    ):
        return False
    if rule.document_scope and document_filters and not set(rule.document_scope).intersection(document_filters):
        return False

    markers = (*rule.markers, *rule.definition_markers)
    phrase_match = bool(rule.phrases) and contains_normalized_phrase(normalized_query, rule.phrases)
    canonical_marker_match = bool(rule.canonical_term) and rule.canonical_term in normalized_query and (
        not markers or contains_normalized_phrase(normalized_query, markers)
    )
    query_type_match = not rule.query_types or bool(set(rule.query_types).intersection(query_types))

    if rule.query_types and not query_type_match and not canonical_marker_match:
        return False
    return phrase_match or canonical_marker_match


def collect_direct_reference_matches(
    normalized_query: str,
    *,
    query_types: Sequence[str] = (),
    document_filters: Sequence[str] = (),
    rule_config: RuleConfig = RULE_CONFIG,
) -> tuple[DirectReferenceRule, ...]:
    return tuple(
        rule
        for rule in rule_config.DIRECT_REFERENCE_RULES
        if direct_reference_rule_matches(
            normalized_query,
            rule,
            query_types=query_types,
            document_filters=document_filters,
        )
    )


def query_asks_for_enumeration(intent: QueryIntent) -> bool:
    query_context = RULE_CONFIG.QUERY_CONTEXT
    if query_context.get("enumeration_query_type", "enumeration") in intent.query_types:
        return True
    if contains_normalized_phrase(intent.normalized_query, RULE_CONFIG.ENUMERATION_QUERY_HINTS):
        return True
    enumeration_issue = query_context.get("enumeration_issue")
    return bool(enumeration_issue and enumeration_issue in intent.issue_filters) and contains_normalized_phrase(
        intent.normalized_query,
        query_context.get("enumeration_context_hints", ()),
    )


def query_asks_without_notice(intent: QueryIntent) -> bool:
    return contains_normalized_phrase(intent.normalized_query, RULE_CONFIG.NO_NOTICE_QUERY_HINTS)


def infer_employee_notice_period_reference(
    intent: QueryIntent,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    config = RULE_CONFIG.QUERY_CONTEXT.get("employee_notice_period_reference", {})
    if not config or config.get("article") not in intent.all_article_numbers:
        return (), ()

    if not (
        set(config.get("required_issues", ())).intersection(intent.issue_filters)
        or set(config.get("required_topics", ())).intersection(intent.topic_filters)
        or set(config.get("required_query_types", ())).intersection(intent.query_types)
    ):
        return (), ()

    query = intent.normalized_query
    if query_asks_without_notice(intent):
        without_notice = config.get("without_notice", {})
        return tuple(without_notice.get("clause_refs", ())), tuple(without_notice.get("point_refs", ()))

    for case in config.get("cases", ()):
        if case.get("query_contains") and not contains_normalized_phrase(query, case["query_contains"]):
            continue
        if case.get("query_contains_all") and not all(phrase in query for phrase in case["query_contains_all"]):
            continue
        return tuple(case.get("clause_refs", ())), tuple(case.get("point_refs", ()))

    default = config.get("default", {})
    return tuple(default.get("clause_refs", ())), tuple(default.get("point_refs", ()))


def collect_rule_based_routing(
    normalized_query: str,
    rule_config: RuleConfig = RULE_CONFIG,
    *,
    query_types: Sequence[str] = (),
    document_filters: Sequence[str] = (),
) -> RuleBasedRoutingResult:
    inferred_articles: list[str] = []
    force_reference_articles: list[str] = []
    forced_references: list[LegalReference] = []
    direct_reference_rule_names: list[str] = []
    topics: list[str] = []
    issues: list[str] = []
    expansions: list[str] = []

    for rule in collect_direct_reference_matches(
        normalized_query,
        query_types=query_types,
        document_filters=document_filters,
        rule_config=rule_config,
    ):
        inferred_articles.extend(reference.article for reference in rule.references if reference.article)
        if rule.confidence == "high":
            forced_references.extend(rule.references)
            force_reference_articles.extend(reference.article for reference in rule.references if reference.article)
            direct_reference_rule_names.append(rule.name)
        topics.extend(rule.topics)
        issues.extend(rule.issues)
        expansions.extend(rule.expansions)

    for rule in (
        *rule_config.LEGAL_HIGH_PRECISION_QUERY_RULES,
        *rule_config.LEGAL_SOFT_HINT_QUERY_RULES,
        *rule_config.TERMINATION_ARTICLE_QUERY_RULES,
    ):
        if not rule_matches_normalized_query(normalized_query, rule):
            continue
        inferred_articles.extend(rule.articles)
        if rule.confidence == "high":
            force_reference_articles.extend(rule.articles)
        topics.extend(rule.topics)
        issues.extend(rule.issues)
        expansions.extend(rule.expansions)

    return RuleBasedRoutingResult(
        inferred_articles=dedupe_preserve_order(inferred_articles),
        force_reference_articles=dedupe_preserve_order(force_reference_articles),
        forced_references=tuple(dict.fromkeys(forced_references)),
        direct_reference_rule_names=dedupe_preserve_order(direct_reference_rule_names),
        topics=dedupe_preserve_order(topics),
        issues=dedupe_preserve_order(issues),
        expansions=dedupe_preserve_order(expansions),
    )


def collect_rule_based_query_expansions(
    normalized_query: str,
    rule_config: RuleConfig = RULE_CONFIG,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    routing = collect_rule_based_routing(normalized_query, rule_config)
    return (
        routing.inferred_articles,
        routing.topics,
        routing.issues,
        routing.expansions,
    )


def collect_mapped_article_expansions(
    *,
    topic_filters: Sequence[str],
    issue_filters: Sequence[str],
    rule_config: RuleConfig = RULE_CONFIG,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    articles: list[str] = []
    expansions: list[str] = []
    for issue in issue_filters:
        articles.extend(rule_config.LEGAL_ISSUE_ARTICLE_MAP.get(issue, ()))
        expansions.extend(rule_config.LEGAL_ISSUE_QUERY_HINTS.get(issue, ()))
    for topic in topic_filters:
        articles.extend(rule_config.LEGAL_TOPIC_ARTICLE_MAP.get(topic, ()))
    expansions.extend(f"Dieu {article}" for article in articles)
    return dedupe_preserve_order(articles), dedupe_preserve_order(expansions)


def filter_specific_actor_labels(actor_labels: Sequence[str]) -> tuple[str, ...]:
    return tuple(label for label in actor_labels if label not in RULE_CONFIG.GENERIC_ACTOR_FILTERS)


def prioritize_issue_filters(issue_labels: Sequence[str]) -> tuple[str, ...]:
    prioritized = tuple(issue_labels)
    query_context = RULE_CONFIG.QUERY_CONTEXT
    benefit_issues = query_context.get("benefit_priority_issues", ())
    drop_issue = query_context.get("issue_to_drop_for_benefit_priority")
    if drop_issue and any(label in prioritized for label in benefit_issues):
        return tuple(label for label in prioritized if label != drop_issue)
    return prioritized


def route_query_heuristic(query: str, rule_config: RuleConfig = RULE_CONFIG) -> QueryIntent:
    normalized_query = normalize_for_matching(f" {query} ")
    document_filters = collect_keyword_matches(normalized_query, rule_config.DOCUMENT_KEYWORDS)
    query_types = collect_keyword_matches(normalized_query, rule_config.QUERY_TYPE_KEYWORDS)
    rule_routing = collect_rule_based_routing(
        normalized_query,
        rule_config,
        query_types=query_types,
        document_filters=document_filters,
    )
    topic_filters = dedupe_preserve_order(
        (
            *collect_keyword_matches(normalized_query, rule_config.TOPIC_KEYWORDS),
            *rule_routing.topics,
        )
    )
    issue_filters = dedupe_preserve_order(
        (
            *collect_keyword_matches(normalized_query, rule_config.ISSUE_KEYWORDS),
            *rule_routing.issues,
        )
    )
    mapped_articles, mapped_expansions = collect_mapped_article_expansions(
        topic_filters=topic_filters,
        issue_filters=issue_filters,
        rule_config=rule_config,
    )
    article_numbers = parse_reference_values(ARTICLE_REF_RE, normalized_query)
    return QueryIntent(
        raw_query=query,
        normalized_query=normalized_query,
        actor_filters=collect_keyword_matches(normalized_query, rule_config.ACTOR_KEYWORDS),
        topic_filters=topic_filters,
        issue_filters=issue_filters,
        document_filters=document_filters,
        article_numbers=article_numbers,
        inferred_article_numbers=dedupe_preserve_order(
            (*rule_routing.inferred_articles, *mapped_articles)
        ),
        force_reference_article_numbers=dedupe_preserve_order(
            (*article_numbers, *rule_routing.force_reference_articles)
        ),
        forced_references=rule_routing.forced_references,
        matched_direct_reference_rules=rule_routing.direct_reference_rule_names,
        clause_refs=parse_reference_values(CLAUSE_REF_RE, normalized_query),
        point_refs=parse_reference_values(POINT_REF_RE, normalized_query),
        query_expansions=dedupe_preserve_order((*rule_routing.expansions, *mapped_expansions)),
        query_types=query_types,
    )


def format_intent_summary(intent: QueryIntent) -> str:
    parts: list[str] = []
    if intent.document_filters:
        parts.append(f"document={', '.join(intent.document_filters)}")
    if intent.actor_filters:
        parts.append(f"actor={', '.join(intent.actor_filters)}")
    if intent.topic_filters:
        parts.append(f"topic={', '.join(intent.topic_filters)}")
    if intent.issue_filters:
        parts.append(f"issue={', '.join(intent.issue_filters)}")
    if intent.article_numbers:
        parts.append(f"dieu={', '.join(intent.article_numbers)}")
    if intent.inferred_article_numbers:
        parts.append(f"dieu_suy_luan={', '.join(intent.inferred_article_numbers)}")
    if intent.force_reference_article_numbers:
        parts.append(f"dieu_force={', '.join(intent.force_reference_article_numbers)}")
    if intent.forced_references:
        formatted_refs = [
            ":".join(
                part
                for part in (
                    reference.document_id,
                    f"dieu={reference.article}" if reference.article else "",
                    f"khoan={reference.clause}" if reference.clause else "",
                    f"diem={reference.point}" if reference.point else "",
                )
                if part
            )
            for reference in intent.forced_references
        ]
        parts.append(f"forced_reference={', '.join(formatted_refs)}")
    if intent.matched_direct_reference_rules:
        parts.append(f"direct_reference_rule={', '.join(intent.matched_direct_reference_rules)}")
    if intent.clause_refs:
        parts.append(f"khoan={', '.join(intent.clause_refs)}")
    if intent.point_refs:
        parts.append(f"diem={', '.join(intent.point_refs)}")
    if intent.query_types:
        parts.append(f"loai_cau_hoi={', '.join(intent.query_types)}")
    if intent.query_expansions:
        parts.append(f"mo_rong={'; '.join(intent.query_expansions)}")
    return "; ".join(parts) if parts else "khong co filter heuristic"


def build_query_variants(intent: QueryIntent) -> tuple[str, ...]:
    variants: list[str] = [intent.raw_query.strip()]
    focuses_on_forced_definition = bool(intent.forced_references) and (
        "definition" in intent.query_types or "giai_thich_tu_ngu" in intent.issue_filters
    )

    if intent.query_expansions and not focuses_on_forced_definition:
        variants.append(" ".join((intent.raw_query, *intent.query_expansions)).strip())

    issue_parts: list[str] = []
    for issue in intent.issue_filters:
        issue_parts.extend(RULE_CONFIG.LEGAL_ISSUE_QUERY_HINTS.get(issue, ()))
    if issue_parts and not focuses_on_forced_definition:
        variants.append(" ".join(dedupe_preserve_order(issue_parts)))

    citation_document_name = RULE_CONFIG.QUERY_CONTEXT.get("citation_document_name", "")
    if intent.forced_references:
        forced_citation_parts = []
        for reference in intent.forced_references:
            if not reference.article:
                continue
            forced_citation_parts.append(
                " ".join(
                    part
                    for part in [
                        f"Dieu {reference.article}",
                        f"khoan {reference.clause}" if reference.clause else "",
                        f"diem {reference.point}" if reference.point else "",
                        citation_document_name,
                    ]
                    if part
                )
            )
        if forced_citation_parts:
            variants.append(" ".join(forced_citation_parts))

    if intent.all_article_numbers and not focuses_on_forced_definition:
        reference_suffix = " ".join(
            (
                *(f"khoan {clause}" for clause in intent.clause_refs),
                *(f"diem {point}" for point in intent.point_refs),
            )
        )
        citation_parts = [
            " ".join(part for part in [f"Dieu {article}", reference_suffix, citation_document_name] if part)
            for article in intent.all_article_numbers
        ]
        citation_parts.extend(intent.query_expansions[:3])
        variants.append(" ".join(citation_parts))

    return tuple(
        variant
        for variant in dedupe_preserve_order(tuple(variant for variant in variants if variant))
        if variant
    )


route_query = route_query_heuristic


__all__ = [
    "ARTICLE_REF_RE",
    "DirectReferenceRule",
    "LegalReference",
    "QueryIntent",
    "RuleBasedQueryExpansion",
    "RuleBasedRoutingResult",
    "YEAR_COUNT_RE",
    "build_query_variants",
    "collect_direct_reference_matches",
    "collect_keyword_matches",
    "collect_rule_based_routing",
    "collect_rule_based_query_expansions",
    "contains_normalized_phrase",
    "dedupe_preserve_order",
    "filter_specific_actor_labels",
    "format_intent_summary",
    "infer_employee_notice_period_reference",
    "parse_reference_values",
    "prioritize_issue_filters",
    "query_asks_for_enumeration",
    "query_asks_without_notice",
    "route_query",
    "route_query_heuristic",
    "rule_matches_normalized_query",
    "direct_reference_rule_matches",
    *_RULE_CONFIG_EXPORTS,
]
