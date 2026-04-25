from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Sequence

from .corpus_pipeline import normalize_for_matching
from .indexing import extract_legal_hint_tokens
from .retriever import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_CONTEXT_TOKENS,
    RetrievalContext,
    dedupe_preserve_order,
    format_context_for_prompt,
    select_contexts_for_prompt,
)

PARENTHETICAL_CITATION_RE = re.compile(r"\([^)]*\)")


SYSTEM_PROMPT = """Ban la tro ly phap ly ve cham dut hop dong lao dong theo phap luat Viet Nam.

Quy tac bat buoc:
1. Chi duoc tra loi dua tren CONTEXT da cung cap.
2. Khong duoc tu bia so Dieu, khoan, diem hoac ten van ban.
3. Neu CONTEXT chua du de ket luan chac chan, phai noi ro la chua du can cu.
3a. Khi thieu context, phai tra loi bang ngon ngu tu nhien, lich su va neu ro thong tin nao con thieu hoac van de nao chua duoc context giai quyet.
3b. Tuyet doi khong duoc tra loi bang kieu thong bao loi he thong, khong duoc lap lai cau mau co dinh.
4. Truong legal_basis chi duoc dung cac chuoi citation nam trong danh sach ALLOWED_CITATIONS.
4a. Phai sao chep citation tu ALLOWED_CITATIONS, khong tu rut gon hoac che lai citation.
4b. Neu co citation cu the hon (vi du co diem a/b/c) thi uu tien citation cu the do.
5. Khong duoc chep cau chu noi dung luat vao legal_basis. legal_basis chi chua citation_text.
6. Tra loi bang tieng Viet, ngan gon va thuc te.
7. Neu insufficient_context = true thi legal_basis phai la mang rong.
8. Cau tra loi phai theo mau: ket luan ngan gon -> can cu phap ly -> dien giai rat ngan neu can.

Ban phai tra dung JSON voi cau truc:
{
  "answer": "cau tra loi ngan gon",
  "legal_basis": ["citation_text 1", "citation_text 2"],
  "insufficient_context": false,
  "notes": "neu can thi ghi them 1 cau ngan, neu khong thi de chuoi rong"
}
"""

ANSWER_FEW_SHOT_PROMPT = """VI DU DINH DANG:

Vi du 1:
- Cau hoi: Nguoi lao dong hop dong khong xac dinh thoi han co duoc nghi viec khong?
- Cau tra loi tot:
{
  "answer": "Co. Nguoi lao dong duoc don phuong cham dut hop dong, nhung phai bao truoc theo quy dinh neu context xac nhan dieu do.",
  "legal_basis": ["Bo luat so 45/2019/QH14, Dieu 35, khoan 1, diem a"],
  "insufficient_context": false,
  "notes": ""
}

Vi du 2:
- Cau hoi: Tro cap thoi viec tinh the nao?
- Cau tra loi tot:
{
  "answer": "Tro cap thoi viec duoc tinh theo thoi gian lam viec du hop le va tien luong binh quan theo context da cung cap.",
  "legal_basis": ["Bo luat so 45/2019/QH14, Dieu 46, khoan 1", "Bo luat so 45/2019/QH14, Dieu 46, khoan 2"],
  "insufficient_context": false,
  "notes": ""
}

Vi du 3:
- Cau hoi: Cong ty no luong 2 thang, toi tu nghi duoc khong?
- Neu context khong xac dinh du thong tin:
{
  "answer": "Chua du can cu de ket luan. Context hien tai chua co quy dinh truc tiep cho tinh huong no luong 2 thang trong bo dieu luat duoc cung cap.",
  "legal_basis": [],
  "insufficient_context": true,
  "notes": "Hay neu ro them can cu lien quan hoac cung cap dung dieu luat ap dung cho truong hop cu the."
}
"""

ANSWER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "legal_basis": {
            "type": "array",
            "items": {"type": "string"},
        },
        "insufficient_context": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": ["answer", "legal_basis", "insufficient_context", "notes"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ParsedAnswer:
    answer: str
    legal_basis: tuple[str, ...]
    insufficient_context: bool
    notes: str
    raw_content: str


def build_allowed_citations(contexts: Sequence[RetrievalContext]) -> tuple[str, ...]:
    allowed_citations: list[str] = []
    for context in contexts:
        allowed_citations.extend(
            citation
            for citation in context.matched_citations
            if str(citation or "").strip()
        )
        if context.citation_text:
            allowed_citations.append(context.citation_text)
    return dedupe_preserve_order(allowed_citations)


def normalize_citation_surface(text: str) -> str:
    without_parenthetical = PARENTHETICAL_CITATION_RE.sub(" ", text)
    normalized = normalize_for_matching(without_parenthetical)
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def citation_overlap_matches(allowed_citation: str, requested_citation: str) -> bool:
    allowed_normalized = normalize_for_matching(allowed_citation)
    requested_normalized = normalize_for_matching(requested_citation)
    allowed_surface = normalize_citation_surface(allowed_citation)
    requested_surface = normalize_citation_surface(requested_citation)
    if not allowed_normalized or not requested_normalized:
        return False
    if allowed_normalized == requested_normalized:
        return True
    if allowed_surface and allowed_surface == requested_surface:
        return True

    allowed_tokens = set(extract_legal_hint_tokens(allowed_citation))
    requested_tokens = set(extract_legal_hint_tokens(requested_citation))
    if allowed_tokens and requested_tokens:
        if not (
            allowed_tokens.issubset(requested_tokens)
            or requested_tokens.issubset(allowed_tokens)
        ):
            return False
        return (
            allowed_normalized in requested_normalized
            or requested_normalized in allowed_normalized
            or (allowed_surface and allowed_surface in requested_surface)
            or (requested_surface and requested_surface in allowed_surface)
        )

    return (
        allowed_normalized in requested_normalized
        or requested_normalized in allowed_normalized
        or (allowed_surface and allowed_surface in requested_surface)
        or (requested_surface and requested_surface in allowed_surface)
    )


def canonicalize_citation(
    requested_citation: str,
    allowed_citations: Sequence[str],
) -> str | None:
    requested_normalized = normalize_for_matching(requested_citation)
    requested_tokens = tuple(extract_legal_hint_tokens(requested_citation))

    for allowed_citation in allowed_citations:
        if normalize_for_matching(allowed_citation) == requested_normalized:
            return allowed_citation

    fuzzy_matches: list[tuple[int, int, int, str]] = []
    for index, allowed_citation in enumerate(allowed_citations):
        if not citation_overlap_matches(allowed_citation, requested_citation):
            continue

        allowed_tokens = tuple(extract_legal_hint_tokens(allowed_citation))
        fuzzy_matches.append(
            (
                abs(len(allowed_tokens) - len(requested_tokens)),
                abs(len(normalize_for_matching(allowed_citation)) - len(requested_normalized)),
                index,
                allowed_citation,
            )
        )

    if not fuzzy_matches:
        return None

    fuzzy_matches.sort()
    return fuzzy_matches[0][3]


def build_messages(
    question: str,
    contexts: Sequence[RetrievalContext],
    *,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_context_tokens: int | None = DEFAULT_MAX_CONTEXT_TOKENS,
) -> list[dict[str, str]]:
    selected_contexts = select_contexts_for_prompt(
        contexts,
        max_chars=max_context_chars,
        max_tokens=max_context_tokens,
    )
    allowed_citations = build_allowed_citations(selected_contexts)
    context_text = format_context_for_prompt(
        selected_contexts,
        max_chars=max_context_chars,
        max_tokens=max_context_tokens,
    )
    user_prompt = "\n\n".join(
        [
            ANSWER_FEW_SHOT_PROMPT,
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

    allowed_citations = build_allowed_citations(contexts)

    if isinstance(legal_basis, str):
        requested = [legal_basis]
    else:
        requested = [str(item) for item in legal_basis]

    filtered: list[str] = []
    for citation in requested:
        canonical_citation = canonicalize_citation(citation, allowed_citations)
        if canonical_citation is None:
            continue
        filtered.append(canonical_citation)
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
            existing_notes = str(payload.get("notes") or "").strip()
            payload["notes"] = " ".join(
                part
                for part in [
                    existing_notes,
                    "Khong xac nhan duoc co so phap ly vi legal_basis khong khop ALLOWED_CITATIONS.",
                ]
                if part
            ).strip()

            if not str(payload.get("answer") or "").strip():
                insufficient_context = True
                payload["answer"] = (
                    "Chua du can cu de ket luan mot cach chac chan tu context hien tai."
                )

    return ParsedAnswer(
        answer=str(payload.get("answer") or "").strip(),
        legal_basis=legal_basis,
        insufficient_context=insufficient_context,
        notes=str(payload.get("notes") or "").strip(),
        raw_content=raw_content,
    )


__all__ = [
    "ANSWER_JSON_SCHEMA",
    "ParsedAnswer",
    "build_allowed_citations",
    "build_messages",
    "canonicalize_citation",
    "citation_overlap_matches",
    "extract_json_candidate",
    "parse_answer_payload",
    "sanitize_legal_basis",
]
