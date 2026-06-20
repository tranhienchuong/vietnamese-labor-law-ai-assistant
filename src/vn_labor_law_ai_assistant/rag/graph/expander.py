from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from ...corpus_pipeline import normalize_for_matching
from ...heuristic_router import QueryIntent, dedupe_preserve_order
from ...indexing import make_qdrant_point_id
from ..retrieval.models import RetrievedRecord, SearchHit
from .config import LegalGraphConfig
from .models import GraphExpansionResult
from .ontology import GRAPH_EXPANSION_EDGE_TYPES
from .store import LegalGraphStore


logger = logging.getLogger(__name__)


MAX_GRAPH_SEED_CHUNKS = 8


@dataclass(frozen=True)
class GraphPriorityReference:
    document_id: str
    article_numbers: tuple[str, ...] = ()
    limit: int = 1
    reason: str = ""
    chunk_id_contains: str = ""


@dataclass(frozen=True)
class GraphExpansionPlan:
    profile: str
    should_expand: bool
    expansion_depth: int
    max_expanded_chunks: int
    min_confidence: float
    seed_confidence: float


RELATION_WEIGHTS: dict[str, float] = {
    "DETAILS": 0.24,
    "GUIDED_BY": 0.22,
    "REFERENCES": 0.20,
    "GUIDES": 0.18,
    "IMPLEMENTS": 0.16,
    "MUST_COMPLY_WITH": 0.14,
    "SUPERIOR_TO": 0.12,
    "SUBORDINATE_TO": 0.10,
    "MENTIONS_TOPIC": 0.11,
    "APPLIES_TO_ACTOR": 0.09,
    "HAS_ISSUE_TYPE": 0.09,
    "MENTIONS_CONCEPT": 0.10,
    "APPLIES_TO": 0.08,
    "HAS_SOURCE_CHUNK": 0.06,
    "SOURCE_OF": 0.06,
    "HAS_ARTICLE": 0.04,
    "HAS_CLAUSE": 0.04,
    "HAS_POINT": 0.04,
}


NATURAL_GRAPH_QUERY_HINTS: tuple[str, ...] = tuple(
    normalize_for_matching(value)
    for value in (
        "khi nào",
        "điều kiện",
        "trường hợp",
        "ngoại lệ",
        "có được không",
        "bồi thường",
        "trợ cấp",
        "đơn phương chấm dứt",
        "không cần báo trước",
        "cho nghỉ việc",
        "sa thải",
        "người lao động được gì",
        "công ty phải làm gì",
    )
)

GRAPH_QUERY_POLICY_REFERENCES: dict[str, tuple[GraphPriorityReference, ...]] = {
    "legal_definition_lookup": (
        GraphPriorityReference("45-2019-qh14", ("3",), 4, "legal_definition_labor_code"),
    ),
    "minor_worker": (
        GraphPriorityReference("45-2019-qh14", ("143", "145", "146", "147"), 1, "minor_worker_labor_code"),
        GraphPriorityReference("thong-tu-09-2020-tt-bldtbxh", ("3",), 2, "minor_worker_tt09_conditions"),
        GraphPriorityReference("thong-tu-09-2020-tt-bldtbxh", ("8", "9", "10"), 1, "minor_worker_tt09_lists"),
    ),
    "retirement_age": (
        GraphPriorityReference("45-2019-qh14", ("169",), 2, "retirement_labor_code"),
        GraphPriorityReference("nghi-dinh-135-2020-nd-cp", ("4", "5", "6", "7"), 1, "retirement_nd135"),
        GraphPriorityReference(
            "nghi-dinh-135-2020-nd-cp",
            (),
            2,
            "retirement_age_female_table",
            chunk_id_contains="Phu_Luc_I_Bang_Nu",
        ),
    ),
    "retirement_age_table_lookup": (
        GraphPriorityReference("45-2019-qh14", ("169",), 2, "retirement_table_labor_code"),
        GraphPriorityReference("nghi-dinh-135-2020-nd-cp", ("4",), 2, "retirement_table_nd135_article_4"),
        GraphPriorityReference(
            "nghi-dinh-135-2020-nd-cp",
            (),
            4,
            "retirement_table_nd135_appendix_i_female",
            chunk_id_contains="Phu_Luc_I_Bang_Nu",
        ),
    ),
    "early_retirement_hazardous_work": (
        GraphPriorityReference("45-2019-qh14", ("169",), 2, "early_retirement_labor_code"),
        GraphPriorityReference("nghi-dinh-135-2020-nd-cp", ("5",), 3, "early_retirement_nd135_article_5"),
        GraphPriorityReference(
            "nghi-dinh-135-2020-nd-cp",
            (),
            2,
            "early_retirement_lowest_age_table",
            chunk_id_contains="Phu_Luc_II",
        ),
        GraphPriorityReference(
            "nghi-dinh-135-2020-nd-cp",
            (),
            1,
            "early_retirement_hazardous_work_list",
            chunk_id_contains="Phu_Luc_III",
        ),
    ),
    "labor_contract": (
        GraphPriorityReference("45-2019-qh14", ("21",), 2, "labor_contract_labor_code"),
        GraphPriorityReference("thong-tu-10-2020-tt-bldtbxh", ("3",), 2, "labor_contract_tt10"),
    ),
    "labor_contract_content": (
        GraphPriorityReference("45-2019-qh14", ("21",), 4, "labor_contract_content_labor_code"),
        GraphPriorityReference("thong-tu-10-2020-tt-bldtbxh", ("3",), 3, "labor_contract_content_tt10"),
        GraphPriorityReference("45-2019-qh14", ("22",), 2, "labor_contract_appendix_context"),
    ),
    "labor_contract_content_guidance": (
        GraphPriorityReference("45-2019-qh14", ("21",), 4, "labor_contract_content_guidance_labor_code"),
        GraphPriorityReference("thong-tu-10-2020-tt-bldtbxh", ("3",), 4, "labor_contract_content_guidance_tt10"),
    ),
    "labor_contract_form": (
        GraphPriorityReference("45-2019-qh14", ("14",), 3, "labor_contract_form_labor_code"),
    ),
    "labor_contract_appendix": (
        GraphPriorityReference("45-2019-qh14", ("22",), 3, "labor_contract_appendix_labor_code"),
    ),
    "prohibited_contracting_acts": (
        GraphPriorityReference("45-2019-qh14", ("17",), 3, "prohibited_contracting_acts_labor_code"),
    ),
    "probation_agreement": (
        GraphPriorityReference("45-2019-qh14", ("24",), 3, "probation_agreement_labor_code"),
    ),
    "probation_end_result": (
        GraphPriorityReference("45-2019-qh14", ("27",), 3, "probation_end_labor_code"),
    ),
    "labor_dispute": (
        GraphPriorityReference("45-2019-qh14", ("179", "188", "190"), 1, "labor_dispute_labor_code"),
        GraphPriorityReference("92-2015-qh13-labor-only", ("32", "33", "35", "91", "119"), 1, "labor_dispute_blttds"),
    ),
    "labor_dispute_litigation": (
        GraphPriorityReference("45-2019-qh14", ("188", "190"), 2, "labor_dispute_litigation_labor_code"),
        GraphPriorityReference("92-2015-qh13-labor-only", ("32", "119"), 2, "labor_dispute_litigation_blttds"),
    ),
    "litigation": (
        GraphPriorityReference("45-2019-qh14", ("188", "190"), 1, "litigation_labor_code"),
        GraphPriorityReference("92-2015-qh13-labor-only", ("32", "35", "37", "39", "40"), 1, "litigation_blttds"),
    ),
    "wage_working_time": (
        GraphPriorityReference("45-2019-qh14", ("94", "95", "96", "97", "98", "105", "106", "107", "111", "113"), 1, "wage_time_labor_code"),
        GraphPriorityReference("nghi-dinh-145-2020-nd-cp", ("54", "55", "56", "57", "58", "59", "60", "61"), 1, "wage_time_nd145"),
    ),
    "night_work_definition": (
        GraphPriorityReference("45-2019-qh14", ("106",), 3, "night_work_definition_labor_code"),
    ),
    "weekly_rest": (
        GraphPriorityReference("45-2019-qh14", ("111",), 3, "weekly_rest_labor_code"),
    ),
    "annual_leave": (
        GraphPriorityReference("45-2019-qh14", ("113",), 3, "annual_leave_labor_code"),
    ),
    "overtime_conditions_and_limits": (
        GraphPriorityReference("45-2019-qh14", ("107",), 5, "overtime_conditions_labor_code"),
        GraphPriorityReference("nghi-dinh-145-2020-nd-cp", ("60",), 3, "overtime_limits_nd145"),
        GraphPriorityReference("nghi-dinh-145-2020-nd-cp", ("61",), 2, "overtime_200_300_hours_nd145"),
        GraphPriorityReference("nghi-dinh-145-2020-nd-cp", ("59",), 1, "overtime_consent_nd145"),
        GraphPriorityReference("45-2019-qh14", ("105",), 1, "normal_working_time_context"),
    ),
    "overtime_pay": (
        GraphPriorityReference("45-2019-qh14", ("98",), 4, "overtime_pay_labor_code"),
        GraphPriorityReference("nghi-dinh-145-2020-nd-cp", ("55", "56", "57"), 2, "overtime_pay_nd145"),
    ),
    "night_overtime_pay": (
        GraphPriorityReference("45-2019-qh14", ("98",), 4, "night_overtime_pay_labor_code"),
        GraphPriorityReference("nghi-dinh-145-2020-nd-cp", ("57",), 3, "night_overtime_pay_nd145_article_57"),
        GraphPriorityReference("nghi-dinh-145-2020-nd-cp", ("55", "56"), 1, "night_overtime_pay_context_nd145"),
    ),
    "compare_overtime_conditions_vs_pay": (
        GraphPriorityReference("45-2019-qh14", ("107",), 4, "comparison_overtime_conditions_labor_code"),
        GraphPriorityReference("45-2019-qh14", ("98",), 4, "comparison_overtime_pay_labor_code"),
        GraphPriorityReference("nghi-dinh-145-2020-nd-cp", ("55",), 2, "comparison_overtime_pay_nd145"),
    ),
    "discipline": (
        GraphPriorityReference("45-2019-qh14", ("118", "122", "123", "124", "125", "127", "128", "129"), 1, "discipline_labor_code"),
        GraphPriorityReference("nghi-dinh-145-2020-nd-cp", ("69", "70", "71", "72", "73"), 1, "discipline_nd145"),
    ),
    "termination": (
        GraphPriorityReference("45-2019-qh14", ("34", "35", "36", "39", "40", "41", "42", "44", "46", "47", "48"), 1, "termination_labor_code"),
        GraphPriorityReference("nghi-dinh-145-2020-nd-cp", ("7", "8", "9", "10", "11", "12", "13", "14"), 1, "termination_nd145"),
    ),
    "employer_unlawful_unilateral_termination": (
        GraphPriorityReference("45-2019-qh14", ("41",), 4, "employer_unlawful_termination_obligation"),
        GraphPriorityReference("45-2019-qh14", ("39",), 1, "unlawful_unilateral_termination_definition"),
    ),
    "illegal_unilateral_termination_by_employee": (
        GraphPriorityReference("45-2019-qh14", ("40",), 4, "employee_unlawful_termination_obligation"),
        GraphPriorityReference("45-2019-qh14", ("39",), 2, "unlawful_unilateral_termination_definition"),
        GraphPriorityReference("45-2019-qh14", ("35",), 2, "employee_unilateral_termination_context"),
    ),
    "structural_change_job_loss_allowance": (
        GraphPriorityReference("45-2019-qh14", ("42",), 3, "structural_change_labor_code"),
        GraphPriorityReference("45-2019-qh14", ("47",), 4, "job_loss_allowance_labor_code"),
        GraphPriorityReference("45-2019-qh14", ("44",), 2, "labor_use_plan_context"),
        GraphPriorityReference("nghi-dinh-145-2020-nd-cp", ("8",), 3, "severance_job_loss_allowance_nd145"),
    ),
    "structural_change_labor_usage_plan": (
        GraphPriorityReference("45-2019-qh14", ("44",), 4, "structural_change_labor_usage_plan"),
        GraphPriorityReference("45-2019-qh14", ("42",), 2, "structural_change_context"),
    ),
    "minor_worker_prohibited_jobs": (
        GraphPriorityReference("45-2019-qh14", ("147",), 4, "minor_worker_prohibited_jobs_labor_code"),
        GraphPriorityReference("thong-tu-09-2020-tt-bldtbxh", ("9",), 3, "minor_worker_prohibited_jobs_tt09"),
    ),
    "female_worker_maternity_protection": (
        GraphPriorityReference("45-2019-qh14", ("137",), 4, "female_maternity_protection_labor_code"),
    ),
    "maternity_leave": (
        GraphPriorityReference("45-2019-qh14", ("139",), 4, "maternity_leave_labor_code"),
    ),
    "compare_employee_unlawful_termination_vs_structural_change": (
        GraphPriorityReference("45-2019-qh14", ("40",), 3, "comparison_employee_unlawful_termination"),
        GraphPriorityReference("45-2019-qh14", ("39",), 1, "comparison_unlawful_termination_definition"),
        GraphPriorityReference("45-2019-qh14", ("42",), 3, "comparison_structural_change"),
        GraphPriorityReference("45-2019-qh14", ("47",), 3, "comparison_job_loss_allowance"),
        GraphPriorityReference("nghi-dinh-145-2020-nd-cp", ("8",), 2, "comparison_nd145_job_loss_details"),
    ),
    "no_notice": (
        GraphPriorityReference("45-2019-qh14", ("35",), 3, "no_notice_labor_code"),
        GraphPriorityReference("nghi-dinh-145-2020-nd-cp", ("7",), 2, "no_notice_nd145"),
    ),
    "no_notice_resignation": (
        GraphPriorityReference("45-2019-qh14", ("35",), 4, "no_notice_resignation_labor_code"),
    ),
}

DIRECT_REFERENCE_QUERY_TYPE_ALIASES: dict[str, str] = {
    "prohibited_contracting_original_papers": "prohibited_contracting_acts",
}
HIGH_CONFIDENCE_POLICY_TYPES: frozenset[str] = frozenset(
    {
        "legal_definition_lookup",
        "minor_worker",
        "retirement_age_table_lookup",
        "early_retirement_hazardous_work",
        "labor_contract_content",
        "labor_contract_content_guidance",
        "labor_contract_form",
        "labor_contract_appendix",
        "prohibited_contracting_acts",
        "probation_agreement",
        "probation_end_result",
        "labor_dispute_litigation",
        "overtime_conditions_and_limits",
        "overtime_pay",
        "night_overtime_pay",
        "compare_overtime_conditions_vs_pay",
        "employer_unlawful_unilateral_termination",
        "illegal_unilateral_termination_by_employee",
        "structural_change_job_loss_allowance",
        "structural_change_labor_usage_plan",
        "minor_worker_prohibited_jobs",
        "female_worker_maternity_protection",
        "maternity_leave",
        "compare_employee_unlawful_termination_vs_structural_change",
        "no_notice_resignation",
    }
)
MULTI_HOP_QUERY_HINTS: tuple[str, ...] = tuple(
    normalize_for_matching(value)
    for value in (
        "liên quan",
        "theo nghị định",
        "hướng dẫn",
        "quy định chi tiết",
        "trừ trường hợp",
        "ngoại lệ",
        "điều kiện",
        "hậu quả",
        "bồi thường",
        "trợ cấp",
        "so sánh",
        "áp dụng như thế nào",
    )
)

GRAPH_FAVORED_QUERY_TYPES: frozenset[str] = frozenset(
    {
        "labor_dispute",
        "labor_dispute_litigation",
        "litigation",
        "retirement_age_table_lookup",
        "early_retirement_hazardous_work",
        "labor_contract_content_guidance",
        "structural_change_job_loss_allowance",
        "structural_change_labor_usage_plan",
        "compare_employee_unlawful_termination_vs_structural_change",
        "compare_overtime_conditions_vs_pay",
        "minor_worker_prohibited_jobs",
        "overtime_conditions_and_limits",
    }
)
GRAPH_FAVORED_QUERY_HINTS: tuple[str, ...] = tuple(
    normalize_for_matching(value)
    for value in (
        "huong dan",
        "quy dinh chi tiet",
        "theo nghi dinh",
        "theo thong tu",
        "phu luc",
        "mau so",
        "toa an",
        "tham quyen",
        "hoa giai",
        "truoc khi kien",
        "khoi kien",
    )
)
PARAPHRASED_REAL_USER_HINTS: tuple[str, ...] = tuple(
    normalize_for_matching(value)
    for value in (
        "toi",
        "em",
        "minh",
        "ben em",
        "cong ty toi",
        "cong ty em",
        "sep",
        "bi cong ty",
        "muon kien",
        "muon nghi",
        "nghi ngay",
        "duoc nghi ngay",
        "co duoc nghi ngay",
        "co duoc khong",
    )
)
HARD_NEGATIVE_QUERY_HINTS: tuple[str, ...] = tuple(
    normalize_for_matching(value)
    for value in (
        "dung nham",
        "dung nham voi",
        "khong nham",
        "phan biet",
        "khac voi",
        "chu khong phai",
        "hay la",
        "hay dung",
    )
)
PRIMARY_LAW_DOCUMENT_ID = "45-2019-qh14"


def _query_has_any(intent: QueryIntent, phrases: Sequence[str]) -> bool:
    normalized_phrases = tuple(normalize_for_matching(value) for value in phrases)
    return any(phrase and phrase in intent.normalized_query for phrase in normalized_phrases)


def classify_graph_query_intent(intent: QueryIntent) -> tuple[str, ...]:
    query_types: list[str] = []
    topics = set(intent.topic_filters)
    issues = set(intent.issue_filters)
    actors = set(intent.actor_filters)
    for rule_name in intent.matched_direct_reference_rules:
        query_types.append(DIRECT_REFERENCE_QUERY_TYPE_ALIASES.get(rule_name, rule_name))

    if (
        intent.article_numbers
        and not intent.clause_refs
        and not intent.point_refs
        and not topics
        and not issues
    ):
        query_types.append("direct_article_lookup")

    if (
        "definition" in intent.query_types
        or "giai_thich_tu_ngu" in issues
        or any(rule_name.startswith("definition_") for rule_name in intent.matched_direct_reference_rules)
        or _query_has_any(intent, ("dinh nghia", "khai niem", "duoc dinh nghia nhu the nao", "la gi theo bo luat lao dong"))
    ):
        query_types.append("legal_definition_lookup")

    if (
        "lao_dong_chua_thanh_nien" in topics
        or "lao_dong_chua_thanh_nien" in issues
        or "lao_dong_chua_thanh_nien" in actors
        or _query_has_any(
            intent,
            (
                "14 tuoi",
                "chua du 15 tuoi",
                "nguoi chua thanh nien",
                "under 15",
                "under-15",
                "minor worker",
                "underage worker",
                "child labor",
            ),
        )
    ):
        query_types.append("minor_worker")

    if (
        "tuoi_nghi_huu" in topics
        or "tuoi_nghi_huu" in issues
        or _query_has_any(intent, ("nghi huu", "huu tri", "tuoi nghi huu", "retirement age", "decree 135"))
    ):
        query_types.append("retirement_age")
    if _query_has_any(intent, ("nu nghi huu nam 2026", "bang tuoi nghi huu", "phu luc i", "nam 2026")) and _query_has_any(
        intent,
        ("nghi huu", "tuoi nghi huu", "huu tri"),
    ):
        query_types.append("retirement_age_table_lookup")
    if _query_has_any(intent, ("nghe nang nhoc", "doc hai", "nguy hiem", "nghi huu som", "tuoi thap hon")) and _query_has_any(
        intent,
        ("nghi huu", "tuoi nghi huu", "huu tri"),
    ):
        query_types.append("early_retirement_hazardous_work")

    if (
        "hop_dong_lao_dong" in topics
        or "giao_ket_hop_dong" in issues
        or "noi_dung_hop_dong" in issues
        or _query_has_any(intent, ("noi dung hop dong", "hop dong lao dong can co"))
    ):
        query_types.append("labor_contract")
    if (
        "noi_dung_hop_dong" in issues
        or _query_has_any(
            intent,
            (
                "noi dung hop dong",
                "hop dong lao dong can co",
                "can co nhung noi dung",
                "noi dung chu yeu cua hop dong",
            ),
        )
    ):
        query_types.append("labor_contract_content")
    if _query_has_any(intent, ("hinh thuc hop dong", "bang van ban", "lap bang van ban", "loi noi", "dien tu")) and _query_has_any(
        intent,
        ("hop dong lao dong", "hop dong"),
    ):
        query_types.append("labor_contract_form")
    if _query_has_any(intent, ("phu luc hop dong", "phu luc hop dong lao dong")):
        query_types.append("labor_contract_appendix")
    if _query_has_any(
        intent,
        (
            "giu ban chinh",
            "giu can cuoc",
            "can cuoc",
            "bang cap",
            "giay to goc",
            "giay to",
            "dat coc",
            "tai san bao dam",
            "bao dam bang tien",
        ),
    ):
        query_types.append("prohibited_contracting_acts")
    if _query_has_any(intent, ("thu viec", "hop dong thu viec")) and _query_has_any(
        intent,
        ("thoa thuan", "duoc thoa thuan", "noi dung thu viec"),
    ):
        query_types.append("probation_agreement")
    if _query_has_any(intent, ("thu viec", "thoi gian thu viec")) and _query_has_any(
        intent,
        ("ket thuc", "het thoi gian", "dat yeu cau", "khong dat yeu cau"),
    ):
        query_types.append("probation_end_result")
    if "labor_contract_content" in query_types and _query_has_any(intent, ("thong tu", "huong dan", "chi tiet")):
        query_types.append("labor_contract_content_guidance")

    if (
        "tranh_chap_lao_dong" in topics
        or "tranh_chap_lao_dong" in issues
        or _query_has_any(intent, ("tranh chap", "hoa giai", "hoa giai vien"))
    ):
        query_types.append("labor_dispute")

    if _query_has_any(intent, ("toa an", "khoi kien", "khi kien", "truoc khi kien", "kien ra toa", "tham quyen")):
        query_types.append("litigation")
    if "labor_dispute" in query_types and "litigation" in query_types:
        query_types.append("labor_dispute_litigation")

    if (
        {"tien_luong", "thoi_gio_lam_viec", "lam_them_gio", "lam_ban_dem"} & topics
        or {"tien_luong", "thoi_gio_lam_viec", "lam_them_gio", "lam_ban_dem"} & issues
        or _query_has_any(intent, ("tien luong", "lam them gio", "thoi gio lam viec", "ca dem", "nghi hang tuan", "nghi hang nam"))
    ):
        query_types.append("wage_working_time")
    if _query_has_any(intent, ("gio lam viec ban dem", "lam viec ban dem", "ca dem", "ban dem")) and _query_has_any(
        intent,
        ("tu may gio", "den may gio", "duoc tinh", "gio"),
    ):
        query_types.append("night_work_definition")
    if _query_has_any(intent, ("nghi hang tuan", "nghi hang tuan nhu the nao")):
        query_types.append("weekly_rest")
    if _query_has_any(intent, ("nghi hang nam", "nghi phep nam", "du 12 thang", "bao nhieu ngay")) and _query_has_any(
        intent,
        ("nghi", "ngay"),
    ):
        query_types.append("annual_leave")

    has_overtime_signal = (
        "lam_them_gio" in topics
        or "lam_them_gio" in issues
        or _query_has_any(intent, ("lam them gio", "lam them", "tang ca"))
    )
    overtime_pay_query = has_overtime_signal and _query_has_any(
        intent,
        (
            "tien luong",
            "luong lam them",
            "tra luong",
            "tinh luong",
            "muc luong",
            "ban dem",
            "ca dem",
            "lam ban dem",
        ),
    )
    overtime_conditions_query = has_overtime_signal and (
        _query_has_any(
            intent,
            (
                "gioi han lam them",
                "so gio lam them",
                "lam them theo thang",
                "lam them theo nam",
                "lam them toi da",
                "toi da bao nhieu",
                "bao nhieu gio",
                "truong hop nao duoc lam them",
                "duoc lam them",
                "dieu kien lam them",
                "khong qua",
                "theo thang",
                "theo nam",
            ),
        )
        or "107" in intent.force_reference_article_numbers
        or "107" in intent.article_numbers
    )
    if overtime_conditions_query and not overtime_pay_query:
        query_types.append("overtime_conditions_and_limits")
    if overtime_pay_query:
        query_types.append("overtime_pay")
        if _query_has_any(intent, ("lam them gio vao ban dem", "lam them vao ban dem", "vao ban dem")):
            query_types.append("night_overtime_pay")
    if has_overtime_signal and _query_has_any(intent, ("phan biet", "so sanh", "khac nhau")) and _query_has_any(
        intent,
        ("dieu kien", "truong hop"),
    ) and _query_has_any(intent, ("tien luong", "luong lam them", "tra luong")):
        query_types.append("compare_overtime_conditions_vs_pay")

    if (
        {"ky_luat_sa_thai"} & topics
        or {"sa_thai", "xu_ly_ky_luat_lao_dong", "ky_luat"} & issues
        or _query_has_any(intent, ("sa thai", "ky luat", "noi quy lao dong"))
    ):
        query_types.append("discipline")

    if (
        {"cham_dut_hop_dong_lao_dong", "don_phuong_cham_dut", "tro_cap", "thay_doi_co_cau_kinh_te"} & topics
        or {
            "can_cu_cham_dut",
            "quyen_don_phuong_cham_dut",
            "thoi_han_bao_truoc",
            "tro_cap_thoi_viec",
            "tro_cap_mat_viec",
            "trai_phap_luat",
            "boi_thuong",
            "thay_doi_co_cau_kinh_te",
        }
        & issues
        or _query_has_any(intent, ("cham dut hop dong", "nghi viec", "bao truoc", "tro cap", "boi thuong"))
    ):
        query_types.append("termination")

    if (
        "trai_phap_luat" in issues
        and (
            _query_has_any(
                intent,
                (
                    "nguoi lao dong don phuong",
                    "nguoi lao dong tu y cham dut",
                    "nguoi lao dong nghi ngang",
                    "nhan vien don phuong",
                    "don phuong cham dut hop dong trai luat thi phai boi thuong",
                ),
            )
            or (
                "nguoi_lao_dong" in actors
                and not _query_has_any(
                    intent,
                    (
                        "cong ty don phuong",
                        "nguoi su dung lao dong don phuong",
                        "lao dong nu mang thai",
                        "sa thai hoac don phuong",
                    ),
                )
            )
        )
    ):
        query_types.append("illegal_unilateral_termination_by_employee")
    if (
        "trai_phap_luat" in issues
        and (
            "nguoi_su_dung_lao_dong" in actors
            or _query_has_any(intent, ("cong ty don phuong", "nguoi su dung lao dong don phuong", "phai nhan nguoi lao dong lai"))
        )
    ):
        query_types.append("employer_unlawful_unilateral_termination")

    if (
        "thay_doi_co_cau_kinh_te" in topics
        or "thay_doi_co_cau_kinh_te" in issues
        or _query_has_any(intent, ("thay doi co cau", "ly do kinh te", "co cau cong nghe"))
    ):
        query_types.append("structural_change_job_loss_allowance")
    if _query_has_any(intent, ("phuong an su dung lao dong", "phai lam gi voi nguoi lao dong", "co cau cong nghe")) and _query_has_any(
        intent,
        ("thay doi co cau", "co cau cong nghe", "ly do kinh te"),
    ):
        query_types.append("structural_change_labor_usage_plan")

    if "minor_worker" in query_types and _query_has_any(intent, ("khong duoc lam", "cam", "cong viec nao", "noi lam viec nao")):
        query_types.append("minor_worker_prohibited_jobs")
    if _query_has_any(intent, ("lao dong nu", "mang thai", "nuoi con duoi 12 thang")) and _query_has_any(
        intent,
        ("sa thai", "don phuong cham dut", "cham dut hop dong"),
    ):
        query_types.append("female_worker_maternity_protection")
    if _query_has_any(intent, ("nghi thai san", "thai san bao lau", "sinh con")):
        query_types.append("maternity_leave")

    if (
        _query_has_any(intent, ("so sanh", "doi chieu", "khac nhau", "khac gi"))
        and (
            "illegal_unilateral_termination_by_employee" in query_types
            or _query_has_any(intent, ("nguoi lao dong don phuong cham dut hop dong trai luat", "don phuong cham dut hop dong trai phap luat"))
        )
        and (
            "structural_change_job_loss_allowance" in query_types
            or _query_has_any(intent, ("cong ty thay doi co cau", "thay doi co cau", "ly do kinh te"))
        )
    ):
        query_types.append("compare_employee_unlawful_termination_vs_structural_change")

    if _query_has_any(
        intent,
        (
            "khong can bao truoc",
            "khong phai bao truoc",
            "khong bao truoc",
            "nghi viec ngay",
            "duoc nghi ngay",
            "without prior notice",
            "without notice",
            "no prior notice",
            "no notice",
        ),
    ):
        query_types.append("no_notice")
        query_types.append("no_notice_resignation")

    high_confidence_types = [
        query_type for query_type in dedupe_preserve_order(tuple(query_types))
        if query_type in HIGH_CONFIDENCE_POLICY_TYPES
    ]
    if "structural_change_labor_usage_plan" in high_confidence_types and not _query_has_any(
        intent,
        ("tro cap", "mat viec", "thoi viec"),
    ):
        high_confidence_types = [
            query_type
            for query_type in high_confidence_types
            if query_type != "structural_change_job_loss_allowance"
        ]
    has_high_confidence_policy = bool(high_confidence_types)
    if has_high_confidence_policy:
        query_types = high_confidence_types

    if not has_high_confidence_policy and (
        len(intent.all_article_numbers) > 1
        or len(intent.document_filters) > 1
        or (
            len(query_types) > 1
            and "direct_article_lookup" not in query_types
        )
        or _query_has_any(intent, ("lien quan", "huong dan", "quy dinh chi tiet", "tru truong hop", "ngoai le"))
    ):
        query_types.append("multi_hop_legal_question")

    return dedupe_preserve_order(tuple(query_types))


def classify_graph_expansion_profile(intent: QueryIntent) -> str:
    graph_query_types = set(classify_graph_query_intent(intent))
    if (
        graph_query_types.intersection(GRAPH_FAVORED_QUERY_TYPES)
        or _query_has_any(intent, GRAPH_FAVORED_QUERY_HINTS)
    ):
        return "graph_favored"
    if _query_has_any(intent, HARD_NEGATIVE_QUERY_HINTS):
        return "hard_negative"
    if (
        _query_has_any(intent, PARAPHRASED_REAL_USER_HINTS)
        and not intent.article_numbers
        and not _query_has_any(intent, GRAPH_FAVORED_QUERY_HINTS)
    ):
        return "paraphrased_real_user"
    return "default"


def graph_priority_references_for_intent(intent: QueryIntent) -> tuple[GraphPriorityReference, ...]:
    references: list[GraphPriorityReference] = []
    for query_type in classify_graph_query_intent(intent):
        references.extend(GRAPH_QUERY_POLICY_REFERENCES.get(query_type, ()))
    return tuple(
        dict.fromkeys(references)
    )


def dedupe_search_hits(hits: Sequence[SearchHit]) -> tuple[SearchHit, ...]:
    seen: set[str] = set()
    ordered: list[SearchHit] = []
    for hit in hits:
        if hit.chunk_id in seen:
            continue
        seen.add(hit.chunk_id)
        ordered.append(hit)
    return tuple(ordered)


class Neo4jLegalGraphExpander:
    def __init__(
        self,
        *,
        store: LegalGraphStore,
        config: LegalGraphConfig,
    ) -> None:
        self.store = store
        self.config = config

    @staticmethod
    def _contains_hint(normalized_query: str, hints: Sequence[str]) -> bool:
        return any(hint and hint in normalized_query for hint in hints)

    def _has_natural_graph_trigger(self, intent: QueryIntent) -> bool:
        return self._contains_hint(intent.normalized_query, NATURAL_GRAPH_QUERY_HINTS)

    @staticmethod
    def _priority_reference_targets(
        intent: QueryIntent,
    ) -> tuple[set[str], set[tuple[str, str]]]:
        priority_documents: set[str] = set()
        priority_articles: set[tuple[str, str]] = set()
        for reference in graph_priority_references_for_intent(intent):
            priority_documents.add(reference.document_id)
            for article_number in reference.article_numbers:
                priority_articles.add((reference.document_id, article_number))
        return priority_documents, priority_articles

    def _seed_support_score(
        self,
        *,
        hit: SearchHit,
        record: RetrievedRecord,
        intent: QueryIntent,
        priority_documents: set[str],
        priority_articles: set[tuple[str, str]],
    ) -> float:
        payload = record.payload
        document_id = str(payload.get("document_id") or "")
        article_number = str(payload.get("article_number") or "")
        clause_ref = str(payload.get("clause_ref") or "")
        point_ref = str(payload.get("point_ref") or "")
        issue_values = {str(value) for value in (payload.get("issue_type") or ())}
        topic_values = {str(value) for value in (payload.get("topic") or ())}

        support = 0.0
        if document_id == PRIMARY_LAW_DOCUMENT_ID:
            support += 0.15
        if document_id in priority_documents:
            support += 0.35
        if (document_id, article_number) in priority_articles:
            support += 0.55
        if article_number and article_number in intent.all_article_numbers:
            support += 0.60
        if clause_ref and clause_ref in intent.clause_refs:
            support += 0.35
        if point_ref and point_ref in intent.point_refs:
            support += 0.25
        if issue_values.intersection(intent.issue_filters):
            support += 0.45
        if topic_values.intersection(intent.topic_filters):
            support += 0.30
        if hit.score > 0:
            support += min(0.30, float(hit.score) * 0.08)
        return support

    def _seed_confidence(
        self,
        hits: Sequence[SearchHit],
        direct_records: dict[str, RetrievedRecord],
        intent: QueryIntent,
    ) -> float:
        priority_documents, priority_articles = self._priority_reference_targets(intent)
        supports: list[float] = []
        for hit in hits[:5]:
            record = direct_records.get(hit.chunk_id)
            if record is None:
                continue
            supports.append(
                self._seed_support_score(
                    hit=hit,
                    record=record,
                    intent=intent,
                    priority_documents=priority_documents,
                    priority_articles=priority_articles,
                )
            )
        if not supports:
            return 0.0
        average_support = sum(supports) / len(supports)
        strongest_support = max(supports)
        return min(1.0, (average_support * 0.6) + (strongest_support * 0.4))

    def _build_expansion_plan(
        self,
        hits: Sequence[SearchHit],
        direct_records: dict[str, RetrievedRecord],
        intent: QueryIntent,
    ) -> GraphExpansionPlan:
        profile = classify_graph_expansion_profile(intent)
        expansion_depth = self._expansion_depth(intent)
        max_expanded_chunks = max(0, int(self.config.max_expanded_chunks))
        min_confidence = max(0.0, float(self.config.min_confidence))
        seed_confidence = self._seed_confidence(hits, direct_records, intent)
        should_expand = self._should_expand(intent)

        if profile == "paraphrased_real_user":
            expansion_depth = min(expansion_depth, 2)
            if seed_confidence >= 0.78:
                max_expanded_chunks = min(max_expanded_chunks, 8)
                min_confidence = min(0.98, min_confidence + 0.05)
            else:
                max_expanded_chunks = min(max_expanded_chunks, 4)
                min_confidence = min(0.98, min_confidence + 0.16)
                should_expand = should_expand and (
                    self._is_multi_hop_query(intent)
                    or self._has_natural_graph_trigger(intent)
                    or bool(set(classify_graph_query_intent(intent)).intersection(GRAPH_FAVORED_QUERY_TYPES))
                )
        elif profile == "hard_negative":
            expansion_depth = min(expansion_depth, 2)
            max_expanded_chunks = min(max_expanded_chunks, 4)
            min_confidence = min(0.98, min_confidence + 0.18)
        elif profile == "graph_favored":
            max_expanded_chunks = max_expanded_chunks
            min_confidence = max(0.0, min_confidence - 0.02)

        return GraphExpansionPlan(
            profile=profile,
            should_expand=should_expand and max_expanded_chunks > 0,
            expansion_depth=max(1, expansion_depth),
            max_expanded_chunks=max(0, max_expanded_chunks),
            min_confidence=min_confidence,
            seed_confidence=seed_confidence,
        )

    def _is_multi_hop_query(self, intent: QueryIntent) -> bool:
        graph_query_types = set(classify_graph_query_intent(intent))
        return bool(
            len(intent.all_article_numbers) > 1
            or len(intent.document_filters) > 1
            or "multi_hop_legal_question" in graph_query_types
            or self._contains_hint(intent.normalized_query, MULTI_HOP_QUERY_HINTS)
            or (
                self._has_natural_graph_trigger(intent)
                and (intent.issue_filters or intent.topic_filters)
            )
        )

    def _expansion_depth(self, intent: QueryIntent) -> int:
        configured_depth = max(1, min(4, int(self.config.expansion_depth)))
        graph_query_types = set(classify_graph_query_intent(intent))
        if graph_query_types == {"direct_article_lookup"}:
            return 1
        if graph_query_types.intersection(HIGH_CONFIDENCE_POLICY_TYPES):
            return 2
        if graph_query_types.intersection(
            {
                "minor_worker",
                "retirement_age",
                "retirement_age_table_lookup",
                "early_retirement_hazardous_work",
                "labor_contract",
                "labor_contract_content",
                "labor_contract_content_guidance",
                "legal_definition_lookup",
                "labor_dispute",
                "labor_dispute_litigation",
                "litigation",
                "overtime_conditions_and_limits",
                "overtime_pay",
                "compare_overtime_conditions_vs_pay",
                "illegal_unilateral_termination_by_employee",
                "employer_unlawful_unilateral_termination",
                "structural_change_job_loss_allowance",
                "compare_employee_unlawful_termination_vs_structural_change",
                "no_notice_resignation",
            }
        ):
            return max(configured_depth, 4)
        if graph_query_types.intersection(
            {
                "labor_contract_form",
                "labor_contract_appendix",
                "prohibited_contracting_acts",
                "probation_agreement",
                "probation_end_result",
                "structural_change_labor_usage_plan",
                "night_work_definition",
                "weekly_rest",
                "annual_leave",
                "minor_worker_prohibited_jobs",
                "female_worker_maternity_protection",
                "maternity_leave",
            }
        ):
            return max(configured_depth, 2)
        if self._is_multi_hop_query(intent):
            return max(configured_depth, 3)
        return max(configured_depth, 2)

    def _should_expand(self, intent: QueryIntent) -> bool:
        if self.config.max_expanded_chunks <= 0:
            return False
        graph_query_types = set(classify_graph_query_intent(intent))
        if graph_query_types == {"direct_article_lookup"}:
            return False
        if graph_query_types.intersection(HIGH_CONFIDENCE_POLICY_TYPES):
            return False
        if graph_query_types:
            return True
        if not self.config.complex_query_only:
            return True
        return bool(
            len(intent.all_article_numbers) > 1
            or intent.clause_refs
            or intent.point_refs
            or intent.issue_filters
            or intent.topic_filters
            or intent.forced_references
            or self._has_natural_graph_trigger(intent)
        )

    def expansion_depth_for_intent(self, intent: QueryIntent) -> int:
        return self._expansion_depth(intent)

    def query_types_for_intent(self, intent: QueryIntent) -> tuple[str, ...]:
        return classify_graph_query_intent(intent)

    def priority_references_for_intent(self, intent: QueryIntent) -> tuple[GraphPriorityReference, ...]:
        return graph_priority_references_for_intent(intent)

    def expansion_profile_for_intent(self, intent: QueryIntent) -> str:
        return classify_graph_expansion_profile(intent)

    @staticmethod
    def _score_graph_hit(
        *,
        seed_scores: Sequence[float],
        path: dict[str, object],
    ) -> float:
        min_seed_score = min(seed_scores) if seed_scores else 0.0
        max_seed_score = max(seed_scores) if seed_scores else 0.0
        depth = max(1, int(path.get("graph_depth") or 1))
        confidence = float(path.get("graph_confidence") or 0.75)
        edge_path = tuple(str(value) for value in path.get("graph_edge_path") or [])
        relation_weight = max((RELATION_WEIGHTS.get(edge, 0.03) for edge in edge_path), default=0.03)
        depth_decay = 1.0 / (1.0 + (0.35 * (depth - 1)))
        graph_score = (min_seed_score * 0.82) + (relation_weight * depth_decay) + (0.04 * confidence)
        if max_seed_score > 0:
            graph_score = min(graph_score, max_seed_score - 1e-6)
        return max(0.0, graph_score)

    def _result_to_hits(
        self,
        result: GraphExpansionResult,
        *,
        seed_hits: Sequence[SearchHit],
        intent: QueryIntent,
        expansion_depth: int,
    ) -> tuple[SearchHit, ...]:
        seed_scores = tuple(hit.score for hit in seed_hits)
        applied_query_intent = list(classify_graph_query_intent(intent))
        paths_by_chunk_id = {
            str(path.get("chunk_id")): path
            for path in result.paths
            if path.get("chunk_id")
        }
        hits: list[SearchHit] = []
        for chunk_id in result.expanded_chunk_ids:
            path = paths_by_chunk_id.get(chunk_id, {})
            edge_path = list(path.get("graph_edge_path") or [])
            node_path = list(path.get("graph_node_path") or [])
            graph_score = self._score_graph_hit(seed_scores=seed_scores, path=path)
            payload = {
                "chunk_id": chunk_id,
                "qdrant_point_id": make_qdrant_point_id(chunk_id),
                "retrieval_source": "graph",
                "retrieval_method": "neo4j_graph_expansion",
                "vector_score": 0.0,
                "graph_score": graph_score,
                "final_score": graph_score,
                "seed_chunk_ids": list(result.seed_chunk_ids),
                "graph_seed_chunk_ids": list(result.seed_chunk_ids),
                "expanded_node_ids": node_path,
                "graph_path": node_path,
                "graph_paths": [node_path] if node_path else [],
                "graph_edge_types": edge_path,
                "graph_edge_path": edge_path,
                "graph_node_path": node_path,
                "graph_depth": int(path.get("graph_depth") or 1),
                "graph_confidence": float(path.get("graph_confidence") or 0.0),
                "applied_query_intent": applied_query_intent,
                "expansion_depth": expansion_depth,
            }
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    qdrant_point_id=str(payload["qdrant_point_id"]),
                    score=graph_score,
                    citation_text="",
                    payload=payload,
                )
            )
        return tuple(hits)

    def expand_from_hits(
        self,
        hits: Sequence[SearchHit],
        direct_records: dict[str, RetrievedRecord],
        intent: QueryIntent,
    ) -> tuple[SearchHit, ...]:
        if not hits:
            return ()
        plan = self._build_expansion_plan(hits, direct_records, intent)
        if not plan.should_expand:
            return ()

        seed_chunk_ids = dedupe_preserve_order(
            tuple(hit.chunk_id for hit in hits if hit.chunk_id in direct_records)
        )[:MAX_GRAPH_SEED_CHUNKS]
        if not seed_chunk_ids:
            return ()

        expansion_depth = plan.expansion_depth
        result = self.store.expand_from_chunk_ids(
            seed_chunk_ids,
            depth=expansion_depth,
            limit=plan.max_expanded_chunks,
            min_confidence=plan.min_confidence,
            edge_types=GRAPH_EXPANSION_EDGE_TYPES,
        )
        if self.config.trace:
            logger.info(
                "Neo4j graph expansion: seeds=%s expanded=%s",
                result.seed_chunk_ids,
                result.expanded_chunk_ids,
            )
        seed_hit_lookup = {hit.chunk_id: hit for hit in hits}
        seed_hits = tuple(
            seed_hit_lookup[chunk_id] for chunk_id in seed_chunk_ids if chunk_id in seed_hit_lookup
        )
        return self._result_to_hits(
            result,
            seed_hits=seed_hits,
            intent=intent,
            expansion_depth=expansion_depth,
        )

    def filter_expanded_hits_for_intent(
        self,
        hits: Sequence[SearchHit],
        direct_records: dict[str, RetrievedRecord],
        intent: QueryIntent,
    ) -> tuple[SearchHit, ...]:
        profile = classify_graph_expansion_profile(intent)
        if profile not in {"paraphrased_real_user", "hard_negative"}:
            return tuple(hits)

        priority_documents, priority_articles = self._priority_reference_targets(intent)
        filtered: list[SearchHit] = []
        for hit in hits:
            retrieval_method = str(hit.payload.get("retrieval_method") or "")
            retrieval_source = str(hit.payload.get("retrieval_source") or "")
            vector_score = float(hit.payload.get("vector_score") or 0.0)
            is_graph_only = (
                vector_score <= 0.0
                and (retrieval_source == "graph" or retrieval_method.startswith("neo4j") or retrieval_method == "graph_query_policy")
            )
            if not is_graph_only:
                filtered.append(hit)
                continue

            record = direct_records.get(hit.chunk_id)
            if record is None:
                continue
            payload = record.payload
            document_id = str(payload.get("document_id") or "")
            article_number = str(payload.get("article_number") or "")
            issue_values = {str(value) for value in (payload.get("issue_type") or ())}
            topic_values = {str(value) for value in (payload.get("topic") or ())}
            clause_ref = str(payload.get("clause_ref") or "")
            point_ref = str(payload.get("point_ref") or "")

            exact_reference_match = bool(
                (article_number and article_number in intent.all_article_numbers)
                or (clause_ref and clause_ref in intent.clause_refs)
                or (point_ref and point_ref in intent.point_refs)
            )
            issue_or_topic_match = bool(
                issue_values.intersection(intent.issue_filters)
                or topic_values.intersection(intent.topic_filters)
            )
            priority_match = document_id in priority_documents or (document_id, article_number) in priority_articles

            keep = exact_reference_match or priority_match
            if profile == "paraphrased_real_user":
                keep = keep or issue_or_topic_match
            else:
                keep = keep or (issue_or_topic_match and document_id == PRIMARY_LAW_DOCUMENT_ID)

            if keep:
                filtered.append(hit)
        return tuple(filtered)


__all__ = [
    "Neo4jLegalGraphExpander",
    "GraphPriorityReference",
    "GraphExpansionPlan",
    "classify_graph_expansion_profile",
    "classify_graph_query_intent",
    "dedupe_search_hits",
    "graph_priority_references_for_intent",
]
