from __future__ import annotations

import re

from ...corpus_pipeline import normalize_for_matching
from .schema import ParsedAnswer

LEADING_QUESTION_SECTION_RE = re.compile(
    r"^\s*(?:Câu\s+hỏi|Cau\s+hoi|Question)\s*:[\s\S]*?"
    r"(?=(?:Câu\s+trả\s+lời|Cau\s+tra\s+loi|Answer)\s*:)",
    re.IGNORECASE,
)

ANSWER_LABEL_RE = re.compile(
    r"^\s*(?:Câu\s+trả\s+lời|Cau\s+tra\s+loi|Trả\s+lời|Tra\s+loi|Answer)\s*:\s*",
    re.IGNORECASE,
)
DETAIL_HEADING_RE = re.compile(
    r"(?im)^\s*(?:Nội\s+dung\s+cụ\s+thể\s+như\s+sau|Noi\s+dung\s+cu\s+the\s+nhu\s+sau|"
    r"Detailed\s+answer|Details)\s*:\s*"
)
SUMMARY_OR_RECOMMENDATION_RE = re.compile(
    r"(?im)^\s*(?:Tóm\s+lại|Tom\s+lai|Khuyến\s+nghị|Khuyen\s+nghi|"
    r"Recommendation|Recommendations|Summary)\s*:\s*"
)
LEGAL_SECTION_RE = re.compile(
    r"(?im)^\s*(?:Căn\s+cứ\s+pháp\s+lý|Can\s+cu\s+phap\s+ly|"
    r"Legal\s+basis|Evidence)\s*:\s*"
)
PARENTHETICAL_CITATION_RE = re.compile(
    r"\s*\((?=[^)]*(?:Điều|Dieu|Article|Bộ luật|Bo luat|Labor Code|"
    r"Nghị định|Nghi dinh|Decree|Thông tư|Thong tu|ND-CP|QH14|QH\s*14))[^)]{1,220}\)"
)
VIETNAMESE_DIACRITIC_RE = re.compile(
    r"[ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩị"
    r"óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    re.IGNORECASE,
)
ENGLISH_HINT_RE = re.compile(
    r"\b(what|when|where|who|whom|whose|why|how|can|could|does|do|did|"
    r"is|are|was|were|should|must|may|employee|employer|worker|contract|"
    r"wage|salary|working|hours|leave|termination|terminate|retirement|"
    r"minor|under|notice|law|legal|rules|definition|apply|determined)\b",
    re.IGNORECASE,
)
VIETNAMESE_HINT_RE = re.compile(
    r"\b(nguoi|lao dong|hop dong|nghi viec|tien luong|luong|bao truoc|"
    r"nghi huu|duoc|khong|the nao|bao nhieu|dieu kien|bo luat|nghi dinh|"
    r"can cu|phap ly|quy dinh|truong hop|cham dut|don phuong|thoi han|la gi)\b",
    re.IGNORECASE,
)


def answer_language(question: str) -> str:
    """Return the user-facing answer language: English for English questions, otherwise Vietnamese."""
    normalized_question = normalize_for_matching(question)
    if VIETNAMESE_DIACRITIC_RE.search(question) or VIETNAMESE_HINT_RE.search(normalized_question):
        return "vi"
    if ENGLISH_HINT_RE.search(question):
        return "en"
    return "vi"


def _labels(question: str) -> tuple[str, str, str, str]:
    if answer_language(question) == "en":
        return (
            "Answer:",
            "Legal basis and evidence:",
            "I could not find enough legal context in the indexed sources to answer this reliably.",
            "No supporting legal provision was retrieved for this answer.",
        )
    return (
        "Trả lời:",
        "Căn cứ và dẫn chứng:",
        "Tôi chưa tìm thấy đủ căn cứ pháp lý trong nguồn đã lập chỉ mục để trả lời đáng tin cậy.",
        "Chưa truy xuất được điều khoản pháp lý đủ để làm căn cứ cho câu trả lời này.",
    )


def _strip_section_from_heading(answer: str, heading_re: re.Pattern[str]) -> str:
    match = heading_re.search(answer)
    if not match:
        return answer
    return answer[: match.start()].strip()


def _clean_direct_answer(answer: str, *, question: str, fallback: str) -> str:
    cleaned = LEADING_QUESTION_SECTION_RE.sub("", answer or "").strip()
    cleaned = ANSWER_LABEL_RE.sub("", cleaned).strip()
    cleaned = _strip_section_from_heading(cleaned, LEGAL_SECTION_RE)
    cleaned = _strip_section_from_heading(cleaned, SUMMARY_OR_RECOMMENDATION_RE)
    cleaned = DETAIL_HEADING_RE.sub("", cleaned)
    cleaned = PARENTHETICAL_CITATION_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = cleaned.strip(" \n:-")
    if cleaned:
        return cleaned
    return fallback


def _quote_line(citation: str, quote: str) -> str:
    normalized_quote = re.sub(r"\s+", " ", quote).strip()
    if normalized_quote:
        return f"- {citation}: \"{normalized_quote}\""
    return f"- {citation}"


def format_answer_for_user(
    answer_payload: ParsedAnswer,
    *,
    question: str = "",
    include_citations: bool = True,
) -> str:
    answer_label, evidence_label, fallback_answer, fallback_evidence = _labels(question)
    direct_answer = _clean_direct_answer(
        answer_payload.answer,
        question=question,
        fallback=fallback_answer,
    )

    parts: list[str] = [answer_label, direct_answer, "", evidence_label]
    if not include_citations:
        parts.append(f"- {fallback_evidence}")
        return "\n".join(parts).strip()

    evidence_lines: list[str] = []
    citations_with_quotes: set[str] = set()
    for evidence_quote in answer_payload.evidence_quotes[:4]:
        citation = evidence_quote.citation.strip()
        if not citation:
            continue
        evidence_lines.append(_quote_line(citation, evidence_quote.quote))
        citations_with_quotes.add(citation)

    for citation in answer_payload.legal_basis:
        clean_citation = citation.strip()
        if clean_citation and clean_citation not in citations_with_quotes:
            evidence_lines.append(f"- {clean_citation}")

    if not evidence_lines:
        evidence_lines.append(f"- {fallback_evidence}")

    parts.extend(evidence_lines)
    return "\n".join(parts).strip()


__all__ = [
    "LEADING_QUESTION_SECTION_RE",
    "answer_language",
    "format_answer_for_user",
]
