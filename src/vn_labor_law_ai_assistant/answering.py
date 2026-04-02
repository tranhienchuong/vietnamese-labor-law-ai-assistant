from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Sequence

from .retriever import (
    DEFAULT_MAX_CONTEXT_CHARS,
    RetrievalContext,
    dedupe_preserve_order,
    format_context_for_prompt,
)


SYSTEM_PROMPT = """Ban la tro ly phap ly ve cham dut hop dong lao dong theo phap luat Viet Nam.

Quy tac bat buoc:
1. Chi duoc tra loi dua tren CONTEXT da cung cap.
2. Khong duoc tu bia so Dieu, khoan, diem hoac ten van ban.
3. Neu CONTEXT chua du de ket luan chac chan, phai noi ro la chua du can cu.
4. Truong legal_basis chi duoc dung cac chuoi citation nam trong danh sach ALLOWED_CITATIONS.
5. Khong duoc chep cau chu noi dung luat vao legal_basis. legal_basis chi chua citation_text.
6. Tra loi bang tieng Viet, ngan gon va thuc te.
7. Neu insufficient_context = true thi legal_basis phai la mang rong.

Ban phai tra dung JSON voi cau truc:
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


def build_messages(
    question: str,
    contexts: Sequence[RetrievalContext],
    *,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> list[dict[str, str]]:
    allowed_citations = build_allowed_citations(contexts)
    context_text = format_context_for_prompt(contexts, max_chars=max_context_chars)
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
) -> tuple[str, ...]:
    if not legal_basis:
        return ()

    allowed_set = set(build_allowed_citations(contexts))

    if isinstance(legal_basis, str):
        requested = [legal_basis]
    else:
        requested = [str(item) for item in legal_basis]

    filtered = [citation for citation in requested if citation in allowed_set]
    return dedupe_preserve_order(filtered)


def extract_json_candidate(raw_content: str) -> str:
    cleaned_content = raw_content.strip()

    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_content, re.IGNORECASE)
    if fenced_match:
        cleaned_content = fenced_match.group(1).strip()

    if (
        (cleaned_content.startswith("{") and cleaned_content.endswith("}"))
        or (cleaned_content.startswith("[") and cleaned_content.endswith("]"))
    ):
        return cleaned_content

    container_match = re.search(r"(\{.*\}|\[.*\])", cleaned_content, re.DOTALL)
    if container_match:
        return container_match.group(1).strip()

    return cleaned_content


def parse_answer_payload(raw_content: str, contexts: Sequence[RetrievalContext]) -> ParsedAnswer:
    cleaned_content = extract_json_candidate(raw_content)

    try:
        raw_payload = json.loads(cleaned_content)
        if isinstance(raw_payload, list) and raw_payload and isinstance(raw_payload[0], dict):
            payload = raw_payload[0]
        elif isinstance(raw_payload, dict):
            payload = raw_payload
        else:
            raise ValueError("JSON is valid but not a dictionary or list of dictionaries.")
    except (json.JSONDecodeError, ValueError):
        payload = {
            "answer": raw_content.strip(),
            "legal_basis": [],
            "insufficient_context": True,
            "notes": "Mo hinh khong tra ve JSON hop le; khong the xac nhan can cu phap ly.",
        }

    insufficient_context = bool(payload.get("insufficient_context"))
    legal_basis = ()
    if not insufficient_context:
        legal_basis = sanitize_legal_basis(payload.get("legal_basis"), contexts)
        if not legal_basis:
            insufficient_context = True
            payload["answer"] = (
                "He thong khong the xac nhan cau tra loi vi mo hinh khong trich dan dung nguon luat da cho."
            )
            existing_notes = str(payload.get("notes") or "").strip()
            payload["notes"] = (
                f"{existing_notes} Cau tra loi da bi vo hieu hoa do thieu co so phap ly hop le.".strip()
            )

    return ParsedAnswer(
        answer=str(payload.get("answer") or "").strip(),
        legal_basis=legal_basis,
        insufficient_context=insufficient_context,
        notes=str(payload.get("notes") or "").strip(),
        raw_content=raw_content,
    )


__all__ = [
    "ParsedAnswer",
    "build_allowed_citations",
    "build_messages",
    "extract_json_candidate",
    "parse_answer_payload",
    "sanitize_legal_basis",
]
