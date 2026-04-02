from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Sequence

from .retriever import RetrievalContext, dedupe_preserve_order, format_context_for_prompt


SYSTEM_PROMPT = """Bạn là trợ lý pháp lý về chấm dứt hợp đồng lao động theo pháp luật Việt Nam.

Quy tắc bắt buộc:
1. Chỉ được trả lời dựa trên CONTEXT đã cung cấp.
2. Không được tự bịa số Điều, khoản, điểm hoặc tên văn bản.
3. Nếu CONTEXT chưa đủ để kết luận chắc chắn, phải nói rõ là chưa đủ căn cứ.
4. Trường legal_basis chỉ được dùng các chuỗi citation nằm trong danh sách ALLOWED_CITATIONS.
5. Không được chép câu chữ nội dung luật vào legal_basis. legal_basis chỉ chứa citation_text.
6. Trả lời bằng tiếng Việt, ngắn gọn và thực tế.

Bạn phải trả đúng JSON với cấu trúc:
{
  "answer": "cau tra loi ngan gon",
  "legal_basis": ["citation_text 1", "citation_text 2"],
  "insufficient_context": false,
  "notes": "neu can thi ghi them 1 cau ngan, neu khong thi de chuoi rong"
}
"""


@dataclass(frozen=True)
class ParsedAnswer:
    answer: str
    legal_basis: tuple[str, ...]
    insufficient_context: bool
    notes: str
    raw_content: str


def build_allowed_citations(contexts: Sequence[RetrievalContext]) -> tuple[str, ...]:
    return dedupe_preserve_order([context.citation_text for context in contexts if context.citation_text])


def build_messages(question: str, contexts: Sequence[RetrievalContext]) -> list[dict[str, str]]:
    allowed_citations = build_allowed_citations(contexts)
    context_text = format_context_for_prompt(contexts)
    user_prompt = "\n\n".join(
        [
            f"Cau hoi:\n{question.strip()}",
            "ALLOWED_CITATIONS:",
            "\n".join(f"- {citation}" for citation in allowed_citations),
            f"CONTEXT:\n{context_text.strip()}",
        ]
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def sanitize_legal_basis(
    legal_basis: Sequence[str] | str | None,
    contexts: Sequence[RetrievalContext],
    *,
    fallback_limit: int = 3,
) -> tuple[str, ...]:
    allowed_citations = build_allowed_citations(contexts)
    allowed_set = set(allowed_citations)

    if isinstance(legal_basis, str):
        requested = [legal_basis]
    else:
        requested = [str(item) for item in (legal_basis or [])]

    filtered = [citation for citation in requested if citation in allowed_set]
    if filtered:
        return dedupe_preserve_order(filtered)

    return allowed_citations[:fallback_limit]


def parse_answer_payload(raw_content: str, contexts: Sequence[RetrievalContext]) -> ParsedAnswer:
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError:
        payload = {
            "answer": raw_content.strip(),
            "legal_basis": [],
            "insufficient_context": False,
            "notes": "Mo hinh khong tra ve JSON hop le; da fallback sang noi dung tho.",
        }

    return ParsedAnswer(
        answer=str(payload.get("answer") or "").strip(),
        legal_basis=sanitize_legal_basis(payload.get("legal_basis"), contexts),
        insufficient_context=bool(payload.get("insufficient_context")),
        notes=str(payload.get("notes") or "").strip(),
        raw_content=raw_content,
    )


__all__ = [
    "ParsedAnswer",
    "build_allowed_citations",
    "build_messages",
    "parse_answer_payload",
    "sanitize_legal_basis",
]
