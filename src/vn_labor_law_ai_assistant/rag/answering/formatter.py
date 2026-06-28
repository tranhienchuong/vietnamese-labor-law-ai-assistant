from __future__ import annotations

import re
from typing import Sequence

from ...corpus_pipeline import normalize_for_matching
from ...retriever import RetrievalContext, dedupe_preserve_order
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
    r"Căn\s+cứ\s+và\s+dẫn\s+chứng|Can\s+cu\s+va\s+dan\s+chung|"
    r"Legal\s+basis(?:\s+and\s+evidence)?|Evidence)\s*:\s*"
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
LEGAL_DETAIL_SPLIT_RE = re.compile(
    r",\s*(?:Điều|Dieu|Article|Chương|Chuong|Chapter|Mục|Muc|Section|Phụ\s+lục|Phu\s+luc)\b.*$",
    re.IGNORECASE,
)
LOWER_LEVEL_DETAIL_SPLIT_RE = re.compile(
    r",\s*(?:khoản|khoan|clause|điểm|diem|point)\b.*$",
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


def _labels(question: str) -> tuple[str, str, str]:
    if answer_language(question) == "en":
        return (
            "Legal basis:",
            "I could not find enough legal context in the indexed sources to answer this reliably.",
            "No supporting legal document was retrieved for this answer.",
        )
    return (
        "Căn cứ pháp lý:",
        "Tôi chưa tìm thấy đủ căn cứ pháp lý trong nguồn đã lập chỉ mục để trả lời đáng tin cậy.",
        "Chưa truy xuất được văn bản pháp lý đủ để làm căn cứ cho câu trả lời này.",
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
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = cleaned.strip(" \n:-")
    if cleaned:
        return cleaned
    return fallback


def _clean_document_title(title: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(title or "")).strip(" \t\n\r:;,.")
    return cleaned


def _document_title_from_citation(citation: str) -> str:
    cleaned = _clean_document_title(citation)
    cleaned = LEGAL_DETAIL_SPLIT_RE.sub("", cleaned)
    cleaned = LOWER_LEVEL_DETAIL_SPLIT_RE.sub("", cleaned)
    return _clean_document_title(cleaned)


def _document_title_from_context(context: RetrievalContext) -> str:
    payload = context.payload or {}
    hierarchy = payload.get("document_hierarchy")
    title = ""
    if isinstance(hierarchy, dict):
        title = str(hierarchy.get("document_title") or "")
    title = str(payload.get("document_title") or title or "")
    return _clean_document_title(title) or _document_title_from_citation(context.citation_text)


def _context_matches_citation(context: RetrievalContext, citation: str) -> bool:
    normalized_citation = normalize_for_matching(citation)
    if not normalized_citation:
        return False
    candidates = (context.citation_text, *context.matched_citations)
    return any(normalize_for_matching(candidate) == normalized_citation for candidate in candidates)


def document_titles_for_legal_basis(
    legal_basis: Sequence[str],
    *,
    contexts: Sequence[RetrievalContext] = (),
) -> tuple[str, ...]:
    titles: list[str] = []
    for citation in legal_basis:
        title = ""
        for context in contexts:
            if _context_matches_citation(context, citation):
                title = _document_title_from_context(context)
                break
        if not title:
            title = _document_title_from_citation(citation)
        if title:
            titles.append(title)
    return dedupe_preserve_order(titles)


def legal_basis_items_for_user(legal_basis: Sequence[str]) -> tuple[str, ...]:
    return dedupe_preserve_order(
        tuple(
            cleaned
            for citation in legal_basis
            if (cleaned := _clean_document_title(citation))
        )
    )


def format_answer_for_user(
    answer_payload: ParsedAnswer,
    *,
    question: str = "",
    include_citations: bool = True,
    contexts: Sequence[RetrievalContext] = (),
) -> str:
    legal_basis_label, fallback_answer, fallback_evidence = _labels(question)
    direct_answer = _clean_direct_answer(
        answer_payload.answer,
        question=question,
        fallback=fallback_answer,
    )

    if not include_citations:
        return direct_answer

    parts: list[str] = [direct_answer]
    legal_basis_items = ()
    if not answer_payload.insufficient_context:
        legal_basis_items = legal_basis_items_for_user(answer_payload.legal_basis)
    if legal_basis_items:
        parts.extend(["", legal_basis_label])
        parts.extend(f"- {basis}" for basis in legal_basis_items)
    elif not answer_payload.insufficient_context:
        parts.extend(["", legal_basis_label, f"- {fallback_evidence}"])
    return "\n".join(parts).strip()


__all__ = [
    "LEADING_QUESTION_SECTION_RE",
    "answer_language",
    "document_titles_for_legal_basis",
    "format_answer_for_user",
    "legal_basis_items_for_user",
]
