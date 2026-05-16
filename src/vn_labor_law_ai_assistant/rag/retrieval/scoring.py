from __future__ import annotations

from typing import Any, Sequence

from ...corpus_pipeline import normalize_for_matching
from ...heuristic_router import (
    QueryIntent,
    YEAR_COUNT_RE,
    contains_normalized_phrase,
    prioritize_issue_filters,
    query_asks_without_notice,
)
from .constants import RULE_CONFIG
from .models import RetrievedRecord, SearchHit
from .query_encoder import actor_labels_for_sparse


class RelevanceScorer:
    def __init__(self, *, rule_config: Any = RULE_CONFIG) -> None:
        self.rule_config = rule_config

    @staticmethod
    def boost_values(value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            return tuple(str(item) for item in value)
        return (str(value),)

    def hint_values(self, name: object) -> tuple[str, ...]:
        return self.boost_values(getattr(self.rule_config, str(name).upper(), ()))

    def has_any(self, text: str, phrases: object) -> bool:
        values = self.boost_values(phrases)
        return bool(values) and contains_normalized_phrase(text, values)

    def has_all(self, text: str, phrases: object) -> bool:
        values = self.boost_values(phrases)
        return bool(values) and all(phrase in text for phrase in values)

    def boost_flags(self, intent: QueryIntent) -> dict[str, bool]:
        context = self.rule_config.BOOST_CONTEXT
        benefit_issues = set(context.get("benefit_computation_issues", ()))
        termination_topics = set(context.get("termination_topics", ()))
        termination_issues = set(context.get("termination_issues", ()))
        benefit_topics = set(context.get("termination_benefit_topics", ()))
        benefit_issue_labels = set(context.get("termination_benefit_issues", ()))
        wants_implementation_detail = contains_normalized_phrase(
            intent.normalized_query,
            self.rule_config.IMPLEMENTATION_DETAIL_HINTS,
        )
        is_calculation_query = (
            contains_normalized_phrase(intent.normalized_query, self.rule_config.CALCULATION_QUERY_HINTS)
            or (
                benefit_issues.intersection(intent.issue_filters)
                and (
                    contains_normalized_phrase(
                        intent.normalized_query,
                        self.rule_config.BENEFIT_COMPUTATION_QUERY_HINTS,
                    )
                    or YEAR_COUNT_RE.search(intent.normalized_query) is not None
                )
            )
        )
        is_termination_query = (
            termination_topics.intersection(intent.topic_filters)
            or termination_issues.intersection(intent.issue_filters)
            or contains_normalized_phrase(intent.normalized_query, self.rule_config.TERMINATION_QUERY_HINTS)
        )
        implementation_hints = tuple(context.get("implementation_document_query_hints", ()))
        return {
            "is_calculation_query": bool(is_calculation_query),
            "wants_implementation_detail": wants_implementation_detail,
            "is_termination_query": bool(is_termination_query),
            "is_termination_benefit_query": bool(
                is_termination_query
                and (
                    benefit_topics.intersection(intent.topic_filters)
                    or benefit_issue_labels.intersection(intent.issue_filters)
                )
            ),
            "query_has_maternity_hint": contains_normalized_phrase(
                intent.normalized_query,
                self.rule_config.MATERNITY_CONTEXT_HINTS,
            ),
            "query_has_retirement_hint": contains_normalized_phrase(
                intent.normalized_query,
                self.rule_config.RETIREMENT_CONTEXT_HINTS,
            ),
            "prefers_primary_law": bool(
                is_termination_query
                and not wants_implementation_detail
                and not contains_normalized_phrase(intent.normalized_query, implementation_hints)
            ),
            "query_without_notice": query_asks_without_notice(intent),
        }

    def boost_condition_matches(
        self,
        condition: dict[str, Any],
        *,
        intent: QueryIntent,
        article_number: str,
        clause_ref: str,
        point_ref: str,
        point_refs: set[str],
        document_id: str,
        level: str,
        actor_values: set[str],
        topic_values: set[str],
        issue_values: set[str],
        section_heading: str,
        heading_text: str,
        normalized_text: str,
        prioritized_issue_filters: tuple[str, ...],
        specific_actor_filters: tuple[str, ...],
        flags: dict[str, bool],
    ) -> bool:
        for key, current in (
            ("article", article_number),
            ("clause", clause_ref),
            ("document_id", document_id),
            ("level", level),
        ):
            if key in condition and current != str(condition[key]):
                return False
        if "point" in condition and str(condition["point"]) not in ({point_ref} | point_refs):
            return False
        if condition.get("article_in_intent_articles") and article_number not in intent.all_article_numbers:
            return False
        if condition.get("clause_in_intent_clauses") and clause_ref not in intent.clause_refs:
            return False
        if "articles" in condition and article_number not in self.boost_values(condition["articles"]):
            return False
        if "document_not" in condition and document_id == str(condition["document_not"]):
            return False
        if "issue_any" in condition and not set(self.boost_values(condition["issue_any"])).intersection(intent.issue_filters):
            return False
        if "topic_any" in condition and not set(self.boost_values(condition["topic_any"])).intersection(intent.topic_filters):
            return False
        if "query_type_any" in condition and not set(self.boost_values(condition["query_type_any"])).intersection(intent.query_types):
            return False
        if "prioritized_issue_any" in condition and not set(self.boost_values(condition["prioritized_issue_any"])).intersection(prioritized_issue_filters):
            return False
        if "record_topic_any" in condition and not topic_values.intersection(self.boost_values(condition["record_topic_any"])):
            return False
        if "record_issue_any" in condition and not issue_values.intersection(self.boost_values(condition["record_issue_any"])):
            return False
        if condition.get("record_issue_intersects_prioritized_issues") and not issue_values.intersection(prioritized_issue_filters):
            return False
        if condition.get("record_topic_intersects_intent_topics") and not topic_values.intersection(intent.topic_filters):
            return False
        if condition.get("record_actor_intersects_specific_actors") and not actor_values.intersection(specific_actor_filters):
            return False
        if "flag" in condition and not flags.get(str(condition["flag"]), False):
            return False
        if "not_flag" in condition and flags.get(str(condition["not_flag"]), False):
            return False
        if condition.get("no_point_refs") and intent.point_refs:
            return False
        if condition.get("no_clause_refs") and intent.clause_refs:
            return False
        if "query_contains" in condition and not self.has_any(intent.normalized_query, condition["query_contains"]):
            return False
        if "query_contains_all" in condition and not self.has_all(intent.normalized_query, condition["query_contains_all"]):
            return False
        if "query_not_contains" in condition and self.has_any(intent.normalized_query, condition["query_not_contains"]):
            return False
        if "normalized_text_contains" in condition and not self.has_any(normalized_text, condition["normalized_text_contains"]):
            return False
        if "normalized_text_contains_hint" in condition and not self.has_any(normalized_text, self.hint_values(condition["normalized_text_contains_hint"])):
            return False
        if "heading_contains" in condition and not self.has_any(heading_text, condition["heading_contains"]):
            return False
        if "heading_not_contains" in condition and self.has_any(heading_text, condition["heading_not_contains"]):
            return False
        if "heading_contains_hint" in condition and not self.has_any(heading_text, self.hint_values(condition["heading_contains_hint"])):
            return False
        if "section_heading_contains_hint" in condition and not self.has_any(section_heading, self.hint_values(condition["section_heading_contains_hint"])):
            return False
        return True

    def score_hit_relevance(
        self,
        hit: SearchHit,
        record: RetrievedRecord,
        intent: QueryIntent,
    ) -> float:
        prioritized_issue_filters = prioritize_issue_filters(intent.issue_filters)
        article_number = str(record.payload.get("article_number") or "").lower()
        clause_ref = str(record.payload.get("clause_ref") or "").lower()
        point_ref = str(record.payload.get("point_ref") or "").lower()
        point_refs = {str(value).lower() for value in record.payload.get("point_refs") or []}
        document_id = str(record.payload.get("document_id") or "")
        level = str(record.payload.get("level") or "")
        actor_values = {str(value) for value in (record.payload.get("actor") or [])}
        topic_values = {str(value) for value in (record.payload.get("topic") or [])}
        issue_values = {str(value) for value in (record.payload.get("issue_type") or [])}
        section_heading = normalize_for_matching(str(record.payload.get("section_heading") or ""))
        heading_text = normalize_for_matching(
            " ".join(
                part
                for part in [
                    str(record.payload.get("article_title") or ""),
                    str(record.payload.get("heading") or ""),
                    record.citation_text,
                ]
                if part
            )
        )
        normalized_text = normalize_for_matching(f" {record.citation_text} {record.text} ")
        specific_actor_filters = actor_labels_for_sparse(intent, self.rule_config)
        flags = self.boost_flags(intent)
        boost = sum(
            float(rule.get("boost", 0.0))
            for rule in self.rule_config.BOOST_RULES
            if self.boost_condition_matches(
                rule.get("when", {}),
                intent=intent,
                article_number=article_number,
                clause_ref=clause_ref,
                point_ref=point_ref,
                point_refs=point_refs,
                document_id=document_id,
                level=level,
                actor_values=actor_values,
                topic_values=topic_values,
                issue_values=issue_values,
                section_heading=section_heading,
                heading_text=heading_text,
                normalized_text=normalized_text,
                prioritized_issue_filters=prioritized_issue_filters,
                specific_actor_filters=specific_actor_filters,
                flags=flags,
            )
        )
        return hit.score + boost

    def rerank_hits(
        self,
        hits: Sequence[SearchHit],
        intent: QueryIntent,
        direct_records: dict[str, RetrievedRecord],
    ) -> tuple[SearchHit, ...]:
        scored_hits: list[tuple[float, SearchHit]] = []
        for hit in hits:
            record = direct_records.get(hit.chunk_id)
            adjusted_score = self.score_hit_relevance(hit, record, intent) if record is not None else hit.score
            scored_hits.append((adjusted_score, hit))

        ordered = sorted(
            scored_hits,
            key=lambda item: (-item[0], -item[1].score, item[1].citation_text),
        )
        return tuple(
            SearchHit(
                chunk_id=hit.chunk_id,
                qdrant_point_id=hit.qdrant_point_id,
                score=adjusted_score,
                citation_text=hit.citation_text,
                payload=hit.payload,
            )
            for adjusted_score, hit in ordered
        )


__all__ = ["RelevanceScorer"]

