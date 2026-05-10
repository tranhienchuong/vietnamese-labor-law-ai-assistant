from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import yaml

from .corpus_pipeline import normalize_for_matching

DEFAULT_RULES_PATH = Path(__file__).with_name("rules") / "routing_config.yaml"
RuleConfig = SimpleNamespace
_RULE_LISTS = ("expansion_rules", "legal_high_precision_rules", "legal_soft_hint_rules")
_RULE_FIELDS = ("phrases", "articles", "topics", "issues", "expansions", "excluded_phrases", "context_phrases")
_HINT_EXPORTS = {
    "calculation_query_hints": "CALCULATION_QUERY_HINTS",
    "calculation_context_hints": "CALCULATION_CONTEXT_HINTS",
    "implementation_detail_hints": "IMPLEMENTATION_DETAIL_HINTS",
    "delegation_context_hints": "DELEGATION_CONTEXT_HINTS",
    "termination_query_hints": "TERMINATION_QUERY_HINTS",
    "termination_section_hints": "TERMINATION_SECTION_HINTS",
    "termination_benefit_context_hints": "TERMINATION_BENEFIT_CONTEXT_HINTS",
    "benefit_computation_query_hints": "BENEFIT_COMPUTATION_QUERY_HINTS",
    "maternity_context_hints": "MATERNITY_CONTEXT_HINTS",
    "retirement_context_hints": "RETIREMENT_CONTEXT_HINTS",
    "enumeration_query_hints": "ENUMERATION_QUERY_HINTS",
    "enumeration_parent_context_hints": "ENUMERATION_PARENT_CONTEXT_HINTS",
    "no_notice_query_hints": "NO_NOTICE_QUERY_HINTS",
}


@dataclass(frozen=True)
class RuleBasedQueryExpansion:
    phrases: tuple[str, ...]
    articles: tuple[str, ...]
    topics: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    expansions: tuple[str, ...] = ()
    excluded_phrases: tuple[str, ...] = ()
    context_phrases: tuple[str, ...] = ()
    confidence: Literal["high", "medium", "low"] = "medium"


@dataclass(frozen=True)
class LegalReference:
    document_id: str = ""
    article: str = ""
    clause: str = ""
    point: str = ""


@dataclass(frozen=True)
class DirectReferenceRule:
    name: str
    references: tuple[LegalReference, ...]
    confidence: Literal["high", "medium", "low"] = "medium"
    canonical_term: str = ""
    phrases: tuple[str, ...] = ()
    markers: tuple[str, ...] = ()
    definition_markers: tuple[str, ...] = ()
    query_types: tuple[str, ...] = ()
    excluded_phrases: tuple[str, ...] = ()
    document_scope: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    expansions: tuple[str, ...] = ()


def _seq(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in (value or ()))


def _normalized_seq(value: Any) -> tuple[str, ...]:
    normalized_values = [
        normalized
        for item in _seq(value)
        if (normalized := normalize_for_matching(str(item)))
    ]
    return tuple(normalized_values)


def _keyword_map(section: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    return {str(label): _seq(data.get("keywords") if isinstance(data, dict) else data) for label, data in section.items()}


def _field_map(section: dict[str, Any], field: str) -> dict[str, tuple[str, ...]]:
    return {str(label): _seq(data.get(field)) for label, data in section.items() if isinstance(data, dict) and data.get(field)}


def _description_map(section: dict[str, Any]) -> dict[str, str]:
    return {str(label): str(data.get("description")) for label, data in section.items() if isinstance(data, dict) and data.get("description")}


def _rule_from_yaml(data: dict[str, Any], default_confidence: str = "medium") -> RuleBasedQueryExpansion:
    return RuleBasedQueryExpansion(
        **{field: _seq(data.get(field)) for field in _RULE_FIELDS},
        confidence=data.get("confidence", default_confidence),
    )


def _reference_from_yaml(data: dict[str, Any]) -> LegalReference:
    return LegalReference(
        document_id=str(data.get("document_id") or ""),
        article=str(data.get("article") or ""),
        clause=str(data.get("clause") or ""),
        point=str(data.get("point") or ""),
    )


def _direct_reference_rule_from_yaml(data: dict[str, Any]) -> DirectReferenceRule:
    return DirectReferenceRule(
        name=str(data.get("name") or ""),
        confidence=data.get("confidence", "medium"),
        canonical_term=normalize_for_matching(str(data.get("canonical_term") or "")),
        phrases=_normalized_seq(data.get("phrases")),
        markers=_normalized_seq(data.get("markers")),
        definition_markers=_normalized_seq(data.get("definition_markers")),
        query_types=_seq(data.get("query_types")),
        excluded_phrases=_normalized_seq(data.get("excluded_phrases")),
        document_scope=_seq(data.get("document_scope")),
        topics=_seq(data.get("topics")),
        issues=_seq(data.get("issues")),
        expansions=_seq(data.get("expansions")),
        references=tuple(
            _reference_from_yaml(reference)
            for reference in data.get("references", ())
            if isinstance(reference, dict)
        ),
    )


class RuleLoader:
    """Load the routing YAML once and expose the familiar Python rule structures.

    Missing YAML sections fall back to empty mappings/tuples, so callers can keep
    using the public config attributes without defensive checks.
    """

    def __init__(self, path: str | Path = DEFAULT_RULES_PATH) -> None:
        self.path = Path(path)

    def load(self) -> RuleConfig:
        with self.path.open(encoding="utf-8") as rule_file:
            raw = yaml.safe_load(rule_file) or {}
        actors, topics, issues = raw.get("actors", {}), raw.get("topics", {}), raw.get("issues", {})
        documents, query_types, hints = raw.get("documents", {}), raw.get("query_types", {}), raw.get("various_hints", {})
        config: dict[str, Any] = {
            "ACTOR_KEYWORDS": _keyword_map(actors),
            "TOPIC_KEYWORDS": _keyword_map(topics),
            "ISSUE_KEYWORDS": _keyword_map(issues),
            "DOCUMENT_KEYWORDS": _keyword_map(documents),
            "QUERY_TYPE_KEYWORDS": _keyword_map(query_types),
            "LEGAL_ISSUE_ARTICLE_MAP": raw.get("issue_article_map") or _field_map(issues, "mapped_articles"),
            "LEGAL_TOPIC_ARTICLE_MAP": raw.get("topic_article_map") or _field_map(topics, "mapped_articles"),
            "LEGAL_ISSUE_QUERY_HINTS": _field_map(issues, "query_hints"),
            "LABEL_ALIASES": raw.get("label_aliases", {}),
            "VALID_ACTORS": tuple(actors),
            "VALID_TOPICS": tuple(topics),
            "VALID_ISSUES": tuple(issues),
            "VALID_DOCUMENTS": tuple(documents),
            "VALID_QUERY_TYPES": tuple(query_types),
            "ACTOR_DESCRIPTIONS": raw.get("actor_descriptions") or _description_map(actors),
            "TOPIC_DESCRIPTIONS": raw.get("topic_descriptions") or _description_map(topics),
            "ISSUE_DESCRIPTIONS": raw.get("issue_descriptions") or _description_map(issues),
            "DOCUMENT_DESCRIPTIONS": raw.get("document_descriptions") or _description_map(documents),
            "QUERY_TYPE_DESCRIPTIONS": raw.get("query_type_descriptions") or _description_map(query_types),
            "GENERIC_ACTOR_FILTERS": frozenset(hints.get("generic_actor_filters", ())),
            "MAX_ENUMERATION_CONTEXT_RECORDS": int(hints.get("max_enumeration_context_records", 16)),
            "QUERY_CONTEXT": raw.get("query_context", {}),
            "BOOST_CONTEXT": raw.get("boost_context", {}),
            "BOOST_RULES": tuple(raw.get("boost_rules", ())),
            "DIRECT_REFERENCE_RULES": tuple(
                _direct_reference_rule_from_yaml(rule)
                for rule in raw.get("direct_reference_rules", ())
                if isinstance(rule, dict)
            ),
            "ROUTER_PROMPT": raw.get("router_prompt", {}),
        }
        for source, target in _HINT_EXPORTS.items():
            config[target] = _seq(hints.get(source))
        for key in _RULE_LISTS:
            default = "high" if key == "legal_high_precision_rules" else "medium"
            config[key.upper()] = tuple(_rule_from_yaml(rule, default) for rule in raw.get(key, ()))
        config["LEGAL_HIGH_PRECISION_QUERY_RULES"] = config.pop("LEGAL_HIGH_PRECISION_RULES")
        config["LEGAL_SOFT_HINT_QUERY_RULES"] = config.pop("LEGAL_SOFT_HINT_RULES")
        config["TERMINATION_ARTICLE_QUERY_RULES"] = tuple(
            _rule_from_yaml(rule) for rule in raw.get("termination_article_rules", ())
        )
        config["TERMINATION_ARTICLE_MAP"] = {
            rule.articles[0]: rule.phrases for rule in config["TERMINATION_ARTICLE_QUERY_RULES"] if len(rule.articles) == 1
        }
        return RuleConfig(**config)


@lru_cache(maxsize=1)
def get_default_rule_config() -> RuleConfig:
    return RuleLoader().load()


DEFAULT_RULE_CONFIG = get_default_rule_config()
