from __future__ import annotations

import re
from typing import Sequence

from ...corpus_pipeline import normalize_for_matching
from .citation_guard import answer_mentions_any_citation
from .schema import ParsedAnswer

LEADING_QUESTION_SECTION_RE = re.compile(
    r"^\s*Câu hỏi\s*:[\s\S]*?(?=Câu trả lời\s*:)",
    re.IGNORECASE,
)


def format_answer_for_user(
    answer_payload: ParsedAnswer,
    *,
    question: str = "",
    include_citations: bool = True,
) -> str:
    answer = LEADING_QUESTION_SECTION_RE.sub("", answer_payload.answer).strip()
    answer = answer or "Chưa đủ căn cứ để kết luận từ context hiện tại."
    normalized_answer = normalize_for_matching(answer)
    parts: list[str] = []

    if "cau tra loi" not in normalized_answer:
        parts.append("Câu trả lời:")
    parts.append(answer)

    if include_citations and answer_payload.legal_basis:
        has_citation_section = (
            "can cu" in normalized_answer
            or "co so phap ly" in normalized_answer
            or answer_mentions_any_citation(answer, answer_payload.legal_basis)
        )
        if not has_citation_section:
            parts.append("")
            parts.append("Căn cứ pháp lý:")
            parts.extend(f"- {citation}" for citation in answer_payload.legal_basis)

    if (
        include_citations
        and answer_payload.evidence_quotes
        and "noi dung cu the" not in normalized_answer
    ):
        parts.append("")
        parts.append("Nội dung cụ thể như sau:")
        for evidence_quote in answer_payload.evidence_quotes[:3]:
            parts.append(f"- {evidence_quote.quote}")

    if "tom lai" not in normalized_answer:
        parts.append("")
        parts.append("Tóm lại:")
        parts.append(f"- {answer}")

    if answer_payload.notes and "khuyen nghi" not in normalized_answer:
        parts.append("")
        parts.append(f"Khuyến nghị: {answer_payload.notes}")

    return "\n".join(parts).strip()


__all__ = [
    "LEADING_QUESTION_SECTION_RE",
    "format_answer_for_user",
]
