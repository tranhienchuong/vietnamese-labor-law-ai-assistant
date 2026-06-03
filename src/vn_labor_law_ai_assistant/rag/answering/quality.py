from __future__ import annotations

from typing import Sequence
import re

from ...corpus_pipeline import normalize_for_matching
from ...retriever import RetrievalContext
from .schema import AnswerQualityValidationResult, ParsedAnswer, QualityCheckStatus
from .synthesis import (
    ANSWER_INTENT_CONTRACT_CONTENT,
    ANSWER_INTENT_DISMISSAL_MEDIATION,
    ANSWER_INTENT_EARLY_RETIREMENT_HAZARDOUS_WORK,
    ANSWER_INTENT_ILLEGAL_EMPLOYEE_TERMINATION,
    ANSWER_INTENT_JOB_LOSS_ALLOWANCE,
    ANSWER_INTENT_MINOR_WORKER,
    ANSWER_INTENT_NO_NOTICE_RESIGNATION,
    ANSWER_INTENT_OVERTIME_LIMITS,
    ANSWER_INTENT_RETIREMENT_AGE,
    ANSWER_INTENT_SEVERANCE_VS_JOB_LOSS,
    ANSWER_INTENT_STRUCTURAL_CHANGE,
    classify_answer_intent,
)


LEGAL_RULE_KEYWORDS = (
    "phai",
    "duoc",
    "khong",
    "quyen",
    "nghia vu",
    "boi thuong",
    "tro cap",
    "tuoi",
    "gio",
    "luong",
    "hop dong",
    "toa an",
    "hoa giai",
    "giai quyet",
    "su dung",
    "bao dam",
    "cham dut",
    "nghi huu",
    "tra",
    "giao ket",
    "bo tri",
    "kiem tra",
)

ARTICLE_TITLE_ONLY_RE = re.compile(r"^\s*(?:dieu|điều)\s+\d+[a-z]?\s*\.?\s*$", re.IGNORECASE)
LEGAL_CITATION_MARKERS = (
    "bo luat",
    "luat",
    "nghi dinh",
    "thong tu",
    "blttds",
    "bllt",
    "dieu ",
)
NOT_APPLICABLE = "not_applicable"


def _meaningful_tokens(text: str) -> list[str]:
    return [token for token in normalize_for_matching(text).split() if len(token) >= 2]


def is_low_information_quote(quote: str) -> bool:
    normalized = normalize_for_matching(quote)
    if not normalized:
        return True
    if ARTICLE_TITLE_ONLY_RE.match(normalized):
        return True
    tokens = _meaningful_tokens(quote)
    if len(tokens) < 8:
        return True
    return not any(keyword in normalized for keyword in LEGAL_RULE_KEYWORDS)


def _contains_all(text: str, terms: Sequence[str]) -> bool:
    return all(term in text for term in terms)


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def _answer_has_citation_marker(line: str) -> bool:
    normalized = normalize_for_matching(line)
    return any(marker in normalized for marker in LEGAL_CITATION_MARKERS)


def _all_legal_claims_have_citations(answer: str) -> bool:
    lines = [
        line.strip()
        for line in answer.splitlines()
        if line.strip() and not line.strip().endswith(":")
    ]
    legal_lines = [
        line
        for line in lines
        if any(keyword in normalize_for_matching(line) for keyword in LEGAL_RULE_KEYWORDS)
        and "tom lai" not in normalize_for_matching(line)
    ]
    if not legal_lines:
        return False
    cited = sum(1 for line in legal_lines if _answer_has_citation_marker(line))
    return cited / len(legal_lines) >= 0.75


def _first_substantive_line(answer: str) -> str:
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        normalized = normalize_for_matching(stripped)
        if normalized in {"cau tra loi", "tom lai", "can cu phap ly", "noi dung cu the nhu sau"}:
            continue
        if stripped.endswith(":"):
            continue
        return stripped
    return ""


def _contexts_contain(contexts: Sequence[RetrievalContext], *terms: str) -> bool:
    normalized = normalize_for_matching(
        "\n".join(
            " ".join([context.citation_text, context.text, str(context.payload.get("retrieval_text") or "")])
            for context in contexts
        )
    )
    return all(normalize_for_matching(term) in normalized for term in terms)


def _count_terms(text: str, terms: Sequence[str]) -> int:
    return sum(1 for term in terms if term in text)


def _retirement_numeric_required(normalized_question: str) -> bool:
    return "bao nhieu" in normalized_question or re.search(r"\bnam\s+\d{4}\b", normalized_question) is not None


def _applicable_status(applicable: bool, passed: bool) -> QualityCheckStatus:
    return bool(passed) if applicable else NOT_APPLICABLE


def _status_passed(status: QualityCheckStatus) -> bool:
    return status is True or status == NOT_APPLICABLE


def _required_rule_present(
    intent: str,
    answer: str,
    contexts: Sequence[RetrievalContext],
    *,
    question: str,
) -> bool:
    normalized = normalize_for_matching(answer)
    normalized_question = normalize_for_matching(question)
    if intent == ANSWER_INTENT_MINOR_WORKER:
        condition_terms = (
            "giao ket hop dong lao dong bang van ban",
            "dai dien theo phap luat",
            "khong anh huong den thoi gian hoc tap",
            "giay kham suc khoe",
            "06 thang",
            "an toan ve sinh lao dong",
        )
        return _count_terms(normalized, condition_terms) >= 3 and "dieu 145" in normalized
    if intent == ANSWER_INTENT_RETIREMENT_AGE:
        if not _retirement_numeric_required(normalized_question):
            return "dieu 169" in normalized and "nghi dinh 135" in normalized and "lo trinh" in normalized
        if _contexts_contain(contexts, "2026", "57 tuoi"):
            return "57 tuoi" in normalized and "dieu 169" in normalized and "nghi dinh 135" in normalized
        return "khong du can cu" in normalized or "chua du can cu" in normalized
    if intent == ANSWER_INTENT_EARLY_RETIREMENT_HAZARDOUS_WORK:
        return (
            ("tuoi thap hon" in normalized or "nghi huu som" in normalized)
            and ("nghe" in normalized or "doc hai" in normalized or "nang nhoc" in normalized)
            and ("dieu 169" in normalized or "nghi dinh 135" in normalized)
            and ("dieu 5" in normalized or "phu luc" in normalized or "dieu 169" in normalized)
        )
    if intent == ANSWER_INTENT_CONTRACT_CONTENT:
        contract_terms = (
            "ten dia chi",
            "ngay thang nam sinh",
            "cong viec",
            "dia diem lam viec",
            "thoi han",
            "muc luong",
            "hinh thuc tra luong",
            "bao hiem xa hoi",
            "bao hiem y te",
            "bao hiem that nghiep",
            "dao tao",
        )
        return _count_terms(normalized, contract_terms) >= 7 and "dieu 21" in normalized
    if intent == ANSWER_INTENT_DISMISSAL_MEDIATION:
        return "khong bat buoc" in normalized and "sa thai" in normalized and "dieu 188" in normalized
    if intent == ANSWER_INTENT_ILLEGAL_EMPLOYEE_TERMINATION:
        terms = (
            "khong duoc tro cap thoi viec",
            "nua thang tien luong",
            "ngay khong bao truoc",
            "chi phi dao tao",
        )
        return _count_terms(normalized, terms) >= 4 and "dieu 40" in normalized
    if intent == ANSWER_INTENT_STRUCTURAL_CHANGE:
        return "tro cap mat viec" in normalized and "dieu 42" in normalized and "dieu 47" in normalized
    if intent == ANSWER_INTENT_JOB_LOSS_ALLOWANCE:
        return (
            "tro cap mat viec" in normalized
            and "moi nam lam viec" in normalized
            and ("01 thang tien luong" in normalized or "1 thang tien luong" in normalized)
            and ("it nhat bang 02 thang tien luong" in normalized or "toi thieu 02 thang tien luong" in normalized)
            and "dieu 47" in normalized
        )
    if intent == ANSWER_INTENT_SEVERANCE_VS_JOB_LOSS:
        severance_terms = (
            "tro cap thoi viec",
            "nua thang tien luong",
            "dieu 46",
        )
        job_loss_terms = (
            "tro cap mat viec",
            "01 thang tien luong",
            "dieu 47",
        )
        return (
            _count_terms(normalized, severance_terms) >= 3
            and _count_terms(normalized, job_loss_terms) >= 3
            and "moi nam lam viec" in normalized
        )
    if intent == ANSWER_INTENT_NO_NOTICE_RESIGNATION:
        terms = (
            "khong duoc bo tri",
            "khong duoc tra du luong",
            "nguoc dai",
            "quay roi tinh duc",
            "lao dong nu mang thai",
            "du tuoi nghi huu",
            "thong tin khong trung thuc",
        )
        return _count_terms(normalized, terms) >= 5 and "dieu 35" in normalized
    if intent == ANSWER_INTENT_OVERTIME_LIMITS:
        return "dong y" in normalized and "40 gio" in normalized and "dieu 107" in normalized
    return len(_meaningful_tokens(answer)) >= 20


def validate_answer_quality(
    question: str,
    parsed: ParsedAnswer,
    contexts: Sequence[RetrievalContext],
    *,
    final_answer: str | None = None,
) -> AnswerQualityValidationResult:
    answer = final_answer or parsed.answer
    normalized = normalize_for_matching(answer)
    normalized_question = normalize_for_matching(question)
    intent = classify_answer_intent(question)
    low_information_quotes = tuple(
        quote.quote for quote in parsed.evidence_quotes if is_low_information_quote(quote.quote)
    )

    direct_answer_present = len(_meaningful_tokens(_first_substantive_line(answer))) >= 4
    if parsed.insufficient_context:
        required_legal_rule_present = (
            "khong du can cu" in normalized
            or "chua du can cu" in normalized
            or "khong co du can cu" in normalized
        )
        warnings: list[str] = []
        if not direct_answer_present:
            warnings.append("Insufficient-context answer does not start with a direct conclusion.")
        if not required_legal_rule_present:
            warnings.append("Insufficient-context answer does not state that legal basis is insufficient.")
        if low_information_quotes:
            warnings.append("Insufficient-context answer should not include evidence quotes.")
        return AnswerQualityValidationResult(
            passed=direct_answer_present and required_legal_rule_present and not low_information_quotes,
            applied_answer_intent=intent,
            direct_answer_present=direct_answer_present,
            required_legal_rule_present=required_legal_rule_present,
            low_information_quotes_count=len(low_information_quotes),
            low_information_quotes=low_information_quotes,
            numeric_answer_present=NOT_APPLICABLE,
            yes_no_answer_present=NOT_APPLICABLE,
            conditions_listed=NOT_APPLICABLE,
            exception_answer_present=NOT_APPLICABLE,
            no_article_title_only_answer=True,
            all_legal_claims_have_citations=True,
            warnings=tuple(warnings),
        )

    required_legal_rule_present = _required_rule_present(
        intent,
        answer,
        contexts,
        question=question,
    )
    numeric_required = intent == ANSWER_INTENT_OVERTIME_LIMITS or (
        intent == ANSWER_INTENT_RETIREMENT_AGE
        and _retirement_numeric_required(normalized_question)
    )
    numeric_answer_present = _applicable_status(
        numeric_required,
        (
            "57 tuoi" in normalized
            or "40 gio" in normalized
            or "khong du can cu" in normalized
            or "chua du can cu" in normalized
        ),
    )
    yes_no_required = intent == ANSWER_INTENT_DISMISSAL_MEDIATION or (
        intent == ANSWER_INTENT_MINOR_WORKER
        and ("co duoc" in normalized_question or normalized_question.rstrip().endswith("khong"))
    )
    yes_no_answer_present = _applicable_status(
        yes_no_required,
        (
            "co the" in normalized
            or "duoc lam viec neu" in normalized
            or "khong bat buoc" in normalized
        ),
    )
    conditions_required = intent in {
        ANSWER_INTENT_MINOR_WORKER,
        ANSWER_INTENT_CONTRACT_CONTENT,
        ANSWER_INTENT_NO_NOTICE_RESIGNATION,
        ANSWER_INTENT_OVERTIME_LIMITS,
    }
    conditions_listed = _applicable_status(
        conditions_required,
        answer.count("\n- ") >= 3,
    )
    exception_answer_present = _applicable_status(
        intent == ANSWER_INTENT_DISMISSAL_MEDIATION,
        "khong bat buoc" in normalized and ("tru" in normalized or "ngoai le" in normalized),
    )
    no_article_title_only_answer = (
        len(_meaningful_tokens(answer)) >= 30
        and not _contains_all(normalized, ("noi dung cu the", "dieu "))
        or required_legal_rule_present
    )
    all_legal_claims_have_citations = _all_legal_claims_have_citations(answer)

    warnings: list[str] = []
    if not direct_answer_present:
        warnings.append("Answer does not start with a direct conclusion.")
    if not required_legal_rule_present:
        warnings.append("Answer is missing required legal rule content for the query type.")
    if low_information_quotes:
        warnings.append("Evidence quotes include low-information text.")
    if numeric_answer_present is False:
        warnings.append("Numeric/date query lacks a numeric answer or an explicit insufficiency statement.")
    if yes_no_answer_present is False:
        warnings.append("Yes/no query lacks a direct yes/no conclusion.")
    if conditions_listed is False:
        warnings.append("Condition/list query does not list concrete conditions.")
    if exception_answer_present is False:
        warnings.append("Exception query does not state the exception directly.")
    if not no_article_title_only_answer:
        warnings.append("Answer appears to rely mainly on article titles.")
    if not all_legal_claims_have_citations:
        warnings.append("Some legal-claim lines do not include citation markers.")

    passed = (
        direct_answer_present
        and required_legal_rule_present
        and not low_information_quotes
        and _status_passed(numeric_answer_present)
        and _status_passed(yes_no_answer_present)
        and _status_passed(conditions_listed)
        and _status_passed(exception_answer_present)
        and no_article_title_only_answer
        and all_legal_claims_have_citations
    )
    return AnswerQualityValidationResult(
        passed=passed,
        applied_answer_intent=intent,
        direct_answer_present=direct_answer_present,
        required_legal_rule_present=required_legal_rule_present,
        low_information_quotes_count=len(low_information_quotes),
        low_information_quotes=low_information_quotes,
        numeric_answer_present=numeric_answer_present,
        yes_no_answer_present=yes_no_answer_present,
        conditions_listed=conditions_listed,
        exception_answer_present=exception_answer_present,
        no_article_title_only_answer=no_article_title_only_answer,
        all_legal_claims_have_citations=all_legal_claims_have_citations,
        warnings=tuple(warnings),
    )


__all__ = [
    "LEGAL_RULE_KEYWORDS",
    "NOT_APPLICABLE",
    "is_low_information_quote",
    "validate_answer_quality",
]
