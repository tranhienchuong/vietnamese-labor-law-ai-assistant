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


GRAPH_EDGE_SCORE_BOOSTS: dict[str, float] = {
    "DETAILS": 0.12,
    "GUIDED_BY": 0.11,
    "REFERENCES": 0.10,
    "GUIDES": 0.08,
    "IMPLEMENTS": 0.07,
}
BLTTDS_LABOR_LITIGATION_ARTICLES = {"32", "33", "35", "37", "39", "40", "91", "119"}
LABOR_DISPUTE_ARTICLES = {"179", "188", "190"}
PRIMARY_LABOR_CODE_ID = "45-2019-qh14"
ND145_ID = "nghi-dinh-145-2020-nd-cp"
ND135_ID = "nghi-dinh-135-2020-nd-cp"
TT09_ID = "thong-tu-09-2020-tt-bldtbxh"
TT10_ID = "thong-tu-10-2020-tt-bldtbxh"
OVERTIME_LIMIT_ND145_ARTICLES = {"59", "60", "61", "62"}
OVERTIME_PAY_ND145_ARTICLES = {"55", "56", "57"}
OVERTIME_PAY_QUERY_HINTS = (
    "tien luong",
    "luong lam them",
    "tra luong",
    "tinh luong",
    "muc luong",
    "ban dem",
    "ca dem",
    "lam ban dem",
)
OVERTIME_LIMIT_TEXT_HINTS = (
    "gioi han",
    "khong qua",
    "so gio lam them",
    "40 gio trong 01 thang",
    "40 gio trong mot thang",
    "200 gio trong 01 nam",
    "300 gio trong 01 nam",
    "su dong y",
    "duoc su dung nguoi lao dong lam them",
)
NORMATIVE_RANK_BONUS = {
    1: 0.045,
    2: 0.025,
    3: 0.010,
}


def _query_has_any(intent: QueryIntent, phrases: Sequence[str]) -> bool:
    normalized_phrases = tuple(normalize_for_matching(value) for value in phrases)
    return any(phrase and phrase in intent.normalized_query for phrase in normalized_phrases)


def _has_intent(applied_query_intent: set[str], *names: str) -> bool:
    return bool(applied_query_intent.intersection(names))


class RelevanceScorer:
    def __init__(self, *, rule_config: Any = RULE_CONFIG) -> None:
        self.rule_config = rule_config

    @staticmethod
    def _policy_intents_for_hit(hit: SearchHit, intent: QueryIntent) -> set[str]:
        applied = {
            str(value)
            for value in (
                hit.payload.get("applied_query_intent")
                or hit.payload.get("query_types")
                or ()
            )
        }
        applied.update(str(value) for value in intent.matched_direct_reference_rules)
        try:
            from ..graph.expander import classify_graph_query_intent
        except Exception:
            graph_intents: tuple[str, ...] = ()
        else:
            graph_intents = classify_graph_query_intent(intent)
        applied.update(str(value) for value in graph_intents)
        if "prohibited_contracting_original_papers" in applied:
            applied.add("prohibited_contracting_acts")
        return applied

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
        graph_edge_types = tuple(str(value) for value in hit.payload.get("graph_edge_types") or ())
        graph_boost = max((GRAPH_EDGE_SCORE_BOOSTS.get(edge, 0.0) for edge in graph_edge_types), default=0.0)
        if hit.payload.get("retrieval_method") == "graph_query_policy":
            graph_boost = max(graph_boost, 0.08)
        try:
            normative_rank = int(record.payload.get("normative_rank") or 0)
        except (TypeError, ValueError):
            normative_rank = 0
        normative_boost = NORMATIVE_RANK_BONUS.get(normative_rank, 0.0)
        applied_query_intent = self._policy_intents_for_hit(hit, intent)
        policy_boost = 0.0
        is_comparison_unlawful_vs_structural = (
            "compare_employee_unlawful_termination_vs_structural_change" in applied_query_intent
        )
        if "legal_definition_lookup" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "3":
                policy_boost += 0.75
                if _query_has_any(intent, ("nguoi lao dong",)) and clause_ref == "1":
                    policy_boost += 4.0
                elif _query_has_any(intent, ("nguoi su dung lao dong",)) and clause_ref == "2":
                    policy_boost += 4.0
                elif _query_has_any(intent, ("to chuc dai dien nguoi lao dong",)) and clause_ref == "3":
                    policy_boost += 4.0
                elif _query_has_any(intent, ("quan he lao dong",)) and clause_ref == "5":
                    policy_boost += 4.0
                elif clause_ref:
                    policy_boost += 0.35
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number == "8" and not _query_has_any(
                intent,
                ("hanh vi bi cam", "bi nghiem cam", "phan biet doi xu"),
            ):
                policy_boost -= 2.0
        if "labor_contract_content" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "21":
                if clause_ref == "1":
                    policy_boost += 2.4
                elif clause_ref in {"2", "3"}:
                    policy_boost += 1.6
                else:
                    policy_boost += 0.8
            elif document_id == TT10_ID and article_number == "3":
                policy_boost += 2.7
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number == "22":
                policy_boost += 0.65
            elif document_id == ND145_ID and article_number == "17" and not _query_has_any(
                intent,
                ("ky quy", "tien ky quy", "quan ly tien ky quy", "dat coc", "bao dam", "cho thue lai lao dong"),
            ):
                policy_boost -= 8.0
        if "labor_contract_content_guidance" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "21":
                policy_boost += 3.0
                if clause_ref == "1":
                    policy_boost += 1.0
            elif document_id == TT10_ID and article_number == "3":
                policy_boost += 3.4
            elif document_id == ND145_ID and article_number == "17" and not _query_has_any(
                intent,
                ("ky quy", "tien ky quy", "dat coc", "bao dam", "cho thue lai lao dong"),
            ):
                policy_boost -= 8.0
        if "labor_contract_form" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "14":
                policy_boost += 4.0
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number in {"17", "21", "22"}:
                policy_boost -= 1.0
        if "labor_contract_appendix" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "22":
                policy_boost += 4.0
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number == "21":
                policy_boost += 0.35
        if "prohibited_contracting_acts" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "17":
                policy_boost += 4.0
            elif document_id == ND145_ID and article_number == "17":
                policy_boost -= 2.0
        if "probation_agreement" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "24":
                policy_boost += 4.0
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number in {"25", "26", "27"}:
                policy_boost += 0.25
        if "probation_end_result" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "27":
                policy_boost += 4.0
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number in {"24", "25", "26"}:
                policy_boost -= 0.4
        if "illegal_unilateral_termination_by_employee" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "40":
                policy_boost += 4.0
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number == "39":
                policy_boost += 2.2
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number == "35":
                policy_boost += 0.8
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number in {"46", "47"} and not _query_has_any(
                intent,
                ("tro cap thoi viec", "tro cap mat viec", "mat viec lam"),
            ):
                policy_boost -= 2.0
            elif (
                document_id == PRIMARY_LABOR_CODE_ID
                and article_number in {"42", "44"}
                and not is_comparison_unlawful_vs_structural
            ):
                policy_boost -= 1.5
        if "structural_change_job_loss_allowance" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "42":
                policy_boost += 2.0
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number == "47":
                policy_boost += 2.5
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number == "44":
                policy_boost += 1.4
            elif document_id == ND145_ID and article_number == "8":
                policy_boost += 2.0
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number == "40" and not _query_has_any(
                intent,
                ("nguoi lao dong don phuong", "trai phap luat", "boi thuong"),
            ):
                policy_boost -= 3.0
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number in {"34", "35", "36", "39"}:
                policy_boost -= 1.0
        if is_comparison_unlawful_vs_structural:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "40":
                policy_boost += 4.0
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number == "39":
                policy_boost += 1.8
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number == "42":
                policy_boost += 4.4
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number == "47":
                policy_boost += 4.2
            elif document_id == ND145_ID and article_number == "8":
                policy_boost += 2.2
        if "no_notice_resignation" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "35" and clause_ref == "2":
                policy_boost += 4.0
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number == "35" and clause_ref == "1":
                policy_boost += 0.7
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number in {"40", "41", "48"} and not _query_has_any(
                intent,
                ("trai phap luat", "boi thuong", "hau qua"),
            ):
                policy_boost -= 2.5
        if _has_intent(applied_query_intent, "minor_worker"):
            if document_id == PRIMARY_LABOR_CODE_ID and article_number in {"143", "145", "146", "147"}:
                policy_boost += 0.45
            elif document_id == TT09_ID and article_number == "3":
                policy_boost += 1.65
            elif document_id == TT09_ID:
                policy_boost += 1.10
        if _has_intent(applied_query_intent, "retirement_age"):
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "169":
                policy_boost += 0.55
            elif document_id == ND135_ID and article_number == "4":
                policy_boost += 0.55
            elif document_id == ND135_ID and not article_number and "phu_luc" in hit.chunk_id.lower():
                policy_boost += 0.45
        if "retirement_age_table_lookup" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "169":
                policy_boost += 2.0
            elif document_id == ND135_ID and article_number == "4":
                policy_boost += 2.5
            elif document_id == ND135_ID and not article_number and "phu_luc_i_bang_nu" in hit.chunk_id.lower():
                policy_boost += 4.0
                if "2026" in normalized_text or "57 tuoi" in normalize_for_matching(normalized_text):
                    policy_boost += 0.8
        if "early_retirement_hazardous_work" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "169":
                policy_boost += 1.6
            elif document_id == ND135_ID and article_number == "5":
                policy_boost += 4.0
            elif document_id == ND135_ID and not article_number and (
                "phu_luc_ii" in hit.chunk_id.lower() or "phu_luc_iii" in hit.chunk_id.lower()
            ):
                policy_boost += 1.2
        if "overtime_conditions_and_limits" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "107":
                policy_boost += 4.0
                if clause_ref == "2":
                    policy_boost += 1.8
                elif clause_ref == "3":
                    policy_boost += 1.25
                elif clause_ref == "1":
                    policy_boost += 0.45
                if self.has_any(normalized_text, OVERTIME_LIMIT_TEXT_HINTS):
                    policy_boost += 0.85
            elif document_id == ND145_ID and article_number == "60":
                policy_boost += 2.8
            elif document_id == ND145_ID and article_number == "61":
                policy_boost += 2.0
            elif document_id == ND145_ID and article_number == "59":
                policy_boost += 1.1
            elif document_id == ND145_ID and article_number == "62":
                policy_boost += 0.75
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number == "105":
                policy_boost += 0.55
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number == "98" and not _query_has_any(
                intent,
                OVERTIME_PAY_QUERY_HINTS,
            ):
                policy_boost -= 4.2
            elif document_id == ND145_ID and article_number in OVERTIME_PAY_ND145_ARTICLES and not _query_has_any(
                intent,
                OVERTIME_PAY_QUERY_HINTS,
            ):
                policy_boost -= 1.8
        if "overtime_pay" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "98":
                policy_boost += 4.0
                if clause_ref in {"1", "2", "3"}:
                    policy_boost += 0.75
            elif document_id == ND145_ID and article_number in OVERTIME_PAY_ND145_ARTICLES:
                policy_boost += 2.0
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number == "107":
                policy_boost -= 0.9
        if "night_overtime_pay" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "98":
                policy_boost += 4.5
                if clause_ref in {"1", "2", "3"}:
                    policy_boost += 0.6
            elif document_id == ND145_ID and article_number == "57":
                policy_boost += 4.2
            elif document_id == ND145_ID and article_number in {"55", "56"}:
                policy_boost += 0.8
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number == "107":
                policy_boost -= 2.0
        if "compare_overtime_conditions_vs_pay" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "107":
                policy_boost += 4.0
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number == "98":
                policy_boost += 4.0
            elif document_id == ND145_ID and article_number == "55":
                policy_boost += 2.0
            elif document_id == ND145_ID and article_number in {"60", "61"}:
                policy_boost += 1.0
        if _has_intent(applied_query_intent, "labor_dispute", "labor_dispute_litigation"):
            if document_id == PRIMARY_LABOR_CODE_ID and article_number in LABOR_DISPUTE_ARTICLES:
                policy_boost += 2.25
                if article_number == "188" and contains_normalized_phrase(
                    intent.normalized_query,
                    ("hoa giai", "truoc khi kien", "khi kien"),
                ):
                    policy_boost += 0.75
            if document_id == "92-2015-qh13-labor-only" and article_number in BLTTDS_LABOR_LITIGATION_ARTICLES:
                policy_boost += 0.25
                if "labor_dispute_litigation" in applied_query_intent and article_number == "32":
                    policy_boost += 3.0
                elif "labor_dispute_litigation" in applied_query_intent and article_number == "119":
                    policy_boost += 1.5
        if "litigation" in applied_query_intent and document_id == "92-2015-qh13-labor-only":
            if article_number in BLTTDS_LABOR_LITIGATION_ARTICLES:
                policy_boost += 0.35
        if "termination" in applied_query_intent and document_id == "nghi-dinh-145-2020-nd-cp":
            policy_boost += 0.35
        if "employer_unlawful_unilateral_termination" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "41":
                policy_boost += 4.5
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number == "39":
                policy_boost += 0.75
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number == "40":
                policy_boost -= 1.0
        if "structural_change_labor_usage_plan" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "44":
                policy_boost += 4.5
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number == "42":
                policy_boost += 1.25
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number == "47":
                policy_boost += 0.45
        if "night_work_definition" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "106":
                policy_boost += 4.0
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number in {"98", "107"}:
                policy_boost -= 1.0
        if "weekly_rest" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "111":
                policy_boost += 4.0
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number in {"105", "107", "98"}:
                policy_boost -= 0.6
        if "annual_leave" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "113":
                policy_boost += 4.0
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number in {"111", "112", "114"}:
                policy_boost += 0.25
        if "minor_worker_prohibited_jobs" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "147":
                policy_boost += 4.2
            elif document_id == TT09_ID and article_number == "9":
                policy_boost += 2.5
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number in {"143", "145", "146"}:
                policy_boost += 0.35
        if "female_worker_maternity_protection" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "137":
                policy_boost += 4.2
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number in {"138", "139", "140"}:
                policy_boost += 0.4
        if "maternity_leave" in applied_query_intent:
            if document_id == PRIMARY_LABOR_CODE_ID and article_number == "139":
                policy_boost += 4.2
            elif document_id == PRIMARY_LABOR_CODE_ID and article_number == "137":
                policy_boost += 0.35
        if query_asks_without_notice(intent):
            if document_id == "45-2019-qh14" and article_number == "35":
                policy_boost += 1.0
            elif document_id == "45-2019-qh14" and article_number == "36":
                policy_boost -= 0.45
        if "no_notice" in applied_query_intent and document_id == "nghi-dinh-145-2020-nd-cp":
            policy_boost += 0.30
        exact_reference_boost = 0.0
        if document_id == PRIMARY_LABOR_CODE_ID and article_number in intent.all_article_numbers:
            exact_reference_boost += 0.30
        if intent.clause_refs and clause_ref in intent.clause_refs:
            exact_reference_boost += 0.35
        if intent.point_refs and ({point_ref} | point_refs).intersection(intent.point_refs):
            exact_reference_boost += 0.35

        if hit.payload.get("retrieval_source") == "graph" and applied_query_intent:
            expected_graph_article = False
            if document_id == PRIMARY_LABOR_CODE_ID and article_number in {
                "3", "14", "17", "21", "22", "24", "27", "35", "39", "40", "41", "42", "44", "47", "98", "105", "106", "107", "111", "113", "137", "139", "143", "145", "146", "147", "169", "188", "190",
            }:
                expected_graph_article = True
            if document_id in {TT09_ID, TT10_ID, ND135_ID}:
                expected_graph_article = True
            if document_id == "92-2015-qh13-labor-only" and article_number in BLTTDS_LABOR_LITIGATION_ARTICLES:
                expected_graph_article = True
            if document_id == ND145_ID and article_number in {"7", "8"}:
                expected_graph_article = True
            if document_id == ND145_ID and article_number in OVERTIME_LIMIT_ND145_ARTICLES | OVERTIME_PAY_ND145_ARTICLES:
                expected_graph_article = True
            if document_id == ND135_ID and not article_number and "phu_luc" in hit.chunk_id.lower():
                expected_graph_article = True
            if not expected_graph_article:
                policy_boost -= 0.35

        return hit.score + boost + graph_boost + normative_boost + policy_boost + exact_reference_boost

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
        ordered = self._promote_comparison_coverage(ordered, intent, direct_records)
        ordered = self._promote_overtime_comparison_coverage(ordered, intent, direct_records)
        return tuple(
            SearchHit(
                chunk_id=hit.chunk_id,
                qdrant_point_id=hit.qdrant_point_id,
                score=adjusted_score,
                citation_text=hit.citation_text,
                payload={
                    **hit.payload,
                    "final_score": adjusted_score,
                },
            )
            for adjusted_score, hit in ordered
        )

    @staticmethod
    def _hit_applied_intents(hit: SearchHit) -> set[str]:
        return {
            str(value)
            for value in (
                hit.payload.get("applied_query_intent")
                or hit.payload.get("query_types")
                or ()
            )
        }

    def _is_unlawful_vs_structural_comparison(
        self,
        ordered_hits: Sequence[tuple[float, SearchHit]],
        intent: QueryIntent,
    ) -> bool:
        applied_intents = {
            intent_name
            for _, hit in ordered_hits
            for intent_name in self._hit_applied_intents(hit)
        }
        if "compare_employee_unlawful_termination_vs_structural_change" in applied_intents:
            return True
        return (
            _query_has_any(intent, ("so sanh", "doi chieu", "khac nhau", "khac gi"))
            and _query_has_any(intent, ("don phuong cham dut hop dong trai luat", "don phuong cham dut hop dong trai phap luat"))
            and _query_has_any(intent, ("thay doi co cau", "ly do kinh te"))
        )

    def _promote_comparison_coverage(
        self,
        ordered_hits: Sequence[tuple[float, SearchHit]],
        intent: QueryIntent,
        direct_records: dict[str, RetrievedRecord],
    ) -> tuple[tuple[float, SearchHit], ...]:
        if not self._is_unlawful_vs_structural_comparison(ordered_hits, intent):
            return tuple(ordered_hits)

        required_articles = (
            (PRIMARY_LABOR_CODE_ID, "40"),
            (PRIMARY_LABOR_CODE_ID, "42"),
            (PRIMARY_LABOR_CODE_ID, "47"),
        )
        promoted: list[tuple[float, SearchHit]] = []
        used_chunk_ids: set[str] = set()
        for document_id, article_number in required_articles:
            for scored_hit in ordered_hits:
                _, hit = scored_hit
                if hit.chunk_id in used_chunk_ids:
                    continue
                record = direct_records.get(hit.chunk_id)
                if record is None:
                    continue
                if (
                    str(record.payload.get("document_id") or "") == document_id
                    and str(record.payload.get("article_number") or "") == article_number
                ):
                    promoted.append(scored_hit)
                    used_chunk_ids.add(hit.chunk_id)
                    break

        if len(promoted) < 2:
            return tuple(ordered_hits)

        top_score = max((score for score, _ in ordered_hits), default=0.0)
        promoted = [
            (top_score + 0.30 - (rank * 1e-3), hit)
            for rank, (_, hit) in enumerate(promoted)
        ]
        remainder = tuple(
            scored_hit
            for scored_hit in ordered_hits
            if scored_hit[1].chunk_id not in used_chunk_ids
        )
        return tuple((*promoted, *remainder))

    def _is_overtime_conditions_vs_pay_comparison(
        self,
        ordered_hits: Sequence[tuple[float, SearchHit]],
        intent: QueryIntent,
    ) -> bool:
        applied_intents = {
            intent_name
            for _, hit in ordered_hits
            for intent_name in self._hit_applied_intents(hit)
        }
        if "compare_overtime_conditions_vs_pay" in applied_intents:
            return True
        return (
            _query_has_any(intent, ("phan biet", "so sanh", "khac nhau"))
            and _query_has_any(intent, ("lam them gio", "tang ca"))
            and _query_has_any(intent, ("dieu kien", "truong hop"))
            and _query_has_any(intent, ("tien luong", "luong lam them", "tra luong"))
        )

    def _promote_overtime_comparison_coverage(
        self,
        ordered_hits: Sequence[tuple[float, SearchHit]],
        intent: QueryIntent,
        direct_records: dict[str, RetrievedRecord],
    ) -> tuple[tuple[float, SearchHit], ...]:
        if not self._is_overtime_conditions_vs_pay_comparison(ordered_hits, intent):
            return tuple(ordered_hits)

        required_articles = (
            (PRIMARY_LABOR_CODE_ID, "107"),
            (PRIMARY_LABOR_CODE_ID, "98"),
        )
        promoted: list[tuple[float, SearchHit]] = []
        used_chunk_ids: set[str] = set()
        for document_id, article_number in required_articles:
            for scored_hit in ordered_hits:
                _, hit = scored_hit
                if hit.chunk_id in used_chunk_ids:
                    continue
                record = direct_records.get(hit.chunk_id)
                if record is None:
                    continue
                if (
                    str(record.payload.get("document_id") or "") == document_id
                    and str(record.payload.get("article_number") or "") == article_number
                ):
                    promoted.append(scored_hit)
                    used_chunk_ids.add(hit.chunk_id)
                    break

        if len(promoted) < 2:
            return tuple(ordered_hits)

        top_score = max((score for score, _ in ordered_hits), default=0.0)
        promoted = [
            (top_score + 0.25 - (rank * 1e-3), hit)
            for rank, (_, hit) in enumerate(promoted)
        ]
        remainder = tuple(
            scored_hit
            for scored_hit in ordered_hits
            if scored_hit[1].chunk_id not in used_chunk_ids
        )
        return tuple((*promoted, *remainder))


__all__ = ["RelevanceScorer"]

