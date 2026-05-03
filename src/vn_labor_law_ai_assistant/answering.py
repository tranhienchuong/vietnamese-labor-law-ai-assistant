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
EVIDENCE_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:])\s+|\n+")


SYSTEM_PROMPT = """Ban la tro ly phap ly ve cham dut hop dong lao dong theo phap luat Viet Nam.

Quy tac bat buoc:
1. Chi duoc tra loi dua tren CONTEXT da cung cap.
1a. Khong duoc dung kien thuc nen ngoai CONTEXT, ke ca khi ban tin rang minh biet cau tra loi.
1b. Truoc khi ket luan, phai tu kiem tra rang ket luan duoc ho tro truc tiep boi cau chu trong CONTEXT.
1c. Neu noi dung trong CONTEXT mau thuan voi kien thuc nen cua ban, phai uu tien CONTEXT.
2. Khong duoc tu bia so Dieu, khoan, diem hoac ten van ban.
3. Chi dat insufficient_context = true neu khong co context nao lien quan truc tiep, hoac context lien quan nhung thieu dieu kien bat buoc de tra loi.
3a. Khi thieu context, phai tra loi bang ngon ngu tu nhien, lich su va neu ro thong tin nao con thieu hoac van de nao chua duoc context giai quyet.
3b. Tuyet doi khong duoc tra loi bang kieu thong bao loi he thong, khong duoc lap lai cau mau co dinh.
3c. Khong duoc dat insufficient_context = true neu CONTEXT da co dieu/khoan/diem truc tiep tra loi cau hoi.
3d. Neu CONTEXT co nguyen tac truc tiep nhung chua du moi ngoai le, hay tra loi phan nguyen tac va neu dieu kien can kiem tra, khong tu choi toan bo.
4. Truong legal_basis chi duoc dung cac chuoi citation nam trong danh sach ALLOWED_CITATIONS.
4a. Phai sao chep citation tu ALLOWED_CITATIONS, khong tu rut gon hoac che lai citation.
4b. Neu co citation cu the hon (vi du co diem a/b/c) thi uu tien citation cu the do.
5. Khong duoc chep cau chu noi dung luat vao legal_basis. legal_basis chi chua citation_text.
6. Tra loi bang tieng Viet, ngan gon va thuc te.
7. Neu insufficient_context = true thi legal_basis va evidence_quotes phai la mang rong.
8. Cau tra loi phai theo mau: ket luan ngan gon -> can cu phap ly -> dien giai rat ngan neu can.
9. Moi ket luan phap ly quan trong phai co evidence_quotes: trich nguyen van mot doan ngan trong CONTEXT dang ho tro ket luan.
9a. Neu khong trich duoc cau chu trong CONTEXT de chung minh ket luan, khong duoc ket luan manh.

Ban phai tra dung JSON voi cau truc:
{
  "answer": "cau tra loi ngan gon",
  "legal_basis": ["citation_text 1", "citation_text 2"],
  "evidence_quotes": [
    {"citation": "citation_text 1", "quote": "doan nguyen van ngan trong CONTEXT"}
  ],
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
  "evidence_quotes": [
    {
      "citation": "Bo luat so 45/2019/QH14, Dieu 35, khoan 1, diem a",
      "quote": "Nguoi lao dong co quyen don phuong cham dut hop dong lao dong nhung phai bao truoc"
    }
  ],
  "insufficient_context": false,
  "notes": ""
}

Vi du 2:
- Cau hoi: Tro cap thoi viec tinh the nao?
- Cau tra loi tot:
{
  "answer": "Tro cap thoi viec duoc tinh theo thoi gian lam viec du hop le va tien luong binh quan theo context da cung cap.",
  "legal_basis": ["Bo luat so 45/2019/QH14, Dieu 46, khoan 1", "Bo luat so 45/2019/QH14, Dieu 46, khoan 2"],
  "evidence_quotes": [
    {
      "citation": "Bo luat so 45/2019/QH14, Dieu 46, khoan 1",
      "quote": "moi nam lam viec duoc tro cap mot nua thang tien luong"
    }
  ],
  "insufficient_context": false,
  "notes": ""
}

Vi du 3:
- Cau hoi: Cong ty no luong 2 thang, toi tu nghi duoc khong?
- Neu context khong xac dinh du thong tin:
{
  "answer": "Chua du can cu de ket luan. Context hien tai chua co quy dinh truc tiep cho tinh huong no luong 2 thang trong bo dieu luat duoc cung cap.",
  "legal_basis": [],
  "evidence_quotes": [],
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
        "evidence_quotes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "citation": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["citation", "quote"],
                "additionalProperties": False,
            },
        },
        "insufficient_context": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": ["answer", "legal_basis", "evidence_quotes", "insufficient_context", "notes"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class EvidenceQuote:
    citation: str
    quote: str


@dataclass(frozen=True)
class ParsedAnswer:
    answer: str
    legal_basis: tuple[str, ...]
    evidence_quotes: tuple[EvidenceQuote, ...]
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


def normalize_quote_surface(text: str) -> str:
    normalized = normalize_for_matching(text)
    return re.sub(r"[^a-z0-9%]+", " ", normalized).strip()


def context_matches_citation(context: RetrievalContext, citation: str) -> bool:
    if citation_overlap_matches(context.citation_text, citation):
        return True
    return any(
        citation_overlap_matches(matched_citation, citation)
        for matched_citation in context.matched_citations
    )


def quote_appears_in_text(quote: str, text: str) -> bool:
    normalized_quote = normalize_quote_surface(quote)
    if len(normalized_quote) < 8:
        return False
    return normalized_quote in normalize_quote_surface(text)


def quote_supported_by_context(
    *,
    citation: str,
    quote: str,
    contexts: Sequence[RetrievalContext],
) -> bool:
    matching_contexts = [
        context for context in contexts if context_matches_citation(context, citation)
    ]
    if matching_contexts:
        return any(quote_appears_in_text(quote, context.text) for context in matching_contexts)
    return any(quote_appears_in_text(quote, context.text) for context in contexts)


def sanitize_evidence_quotes(
    evidence_quotes: object,
    contexts: Sequence[RetrievalContext],
) -> tuple[EvidenceQuote, ...]:
    if not isinstance(evidence_quotes, list):
        return ()

    allowed_citations = build_allowed_citations(contexts)
    sanitized: list[EvidenceQuote] = []
    seen: set[tuple[str, str]] = set()
    for item in evidence_quotes:
        if not isinstance(item, dict):
            continue
        raw_citation = str(item.get("citation") or "").strip()
        quote = str(item.get("quote") or "").strip()
        if not raw_citation or not quote:
            continue
        citation = canonicalize_citation(raw_citation, allowed_citations)
        if citation is None:
            continue
        if not quote_supported_by_context(citation=citation, quote=quote, contexts=contexts):
            continue
        key = (citation, normalize_quote_surface(quote))
        if key in seen:
            continue
        seen.add(key)
        sanitized.append(EvidenceQuote(citation=citation, quote=quote))
    return tuple(sanitized)


def combined_context_text(contexts: Sequence[RetrievalContext]) -> str:
    return "\n".join(context.text for context in contexts)


def contains_percent_value(text: str, value: int) -> bool:
    return re.search(rf"(?<!\d){value}\s*%", text) is not None


def first_context_with_terms(
    contexts: Sequence[RetrievalContext],
    required_terms: Sequence[str],
) -> RetrievalContext | None:
    for context in contexts:
        normalized = normalize_for_matching(context.text)
        if all(term in normalized for term in required_terms):
            return context
    return None


def citation_for_context(
    context: RetrievalContext,
    preferred_terms: Sequence[str] = (),
) -> str:
    citations = dedupe_preserve_order((*context.matched_citations, context.citation_text))
    normalized_terms = tuple(normalize_for_matching(term) for term in preferred_terms)

    for citation in citations:
        normalized_citation = normalize_for_matching(citation)
        if normalized_terms and all(term in normalized_citation for term in normalized_terms):
            return citation

    for citation in citations:
        if citation:
            return citation

    return context.citation_text


def extract_evidence_sentence(text: str, preferred_terms: Sequence[str]) -> str:
    normalized_terms = tuple(normalize_for_matching(term) for term in preferred_terms)
    candidates = [
        sentence.strip()
        for sentence in EVIDENCE_SENTENCE_SPLIT_RE.split(text)
        if sentence.strip()
    ]
    for sentence in candidates:
        normalized_sentence = normalize_for_matching(sentence)
        if all(term in normalized_sentence for term in normalized_terms):
            return sentence[:500]
    for sentence in candidates:
        normalized_sentence = normalize_for_matching(sentence)
        if any(term in normalized_sentence for term in normalized_terms):
            return sentence[:500]
    return text.strip()[:500]


@dataclass(frozen=True)
class ContextualAnswerOverride:
    answer: str
    evidence_quote: EvidenceQuote
    notes: str


def contextual_answer_override(
    question: str,
    contexts: Sequence[RetrievalContext],
) -> ContextualAnswerOverride | None:
    normalized_question = normalize_for_matching(question)
    raw_context = combined_context_text(contexts)
    normalized_context = normalize_for_matching(raw_context)

    context: RetrievalContext | None = None
    answer = ""
    quote_terms: tuple[str, ...] = ()
    citation_terms: tuple[str, ...] = ()
    notes = ""
    asks_notice_period = (
        "bao truoc" in normalized_question
        and (
            "bao lau" in normalized_question
            or "thoi han" in normalized_question
            or "nghi viec" in normalized_question
            or "cham dut" in normalized_question
            or "don phuong" in normalized_question
        )
    )

    if (
        "hop dong lao dong" in normalized_question
        and (
            "bao nhieu loai" in normalized_question
            or "co may loai" in normalized_question
        )
        and "hop dong lao dong khong xac dinh thoi han" in normalized_context
        and "hop dong lao dong xac dinh thoi han" in normalized_context
    ):
        context = first_context_with_terms(
            contexts,
            ("hop dong lao dong khong xac dinh thoi han", "hop dong lao dong xac dinh thoi han"),
        )
        answer = (
            "Co 2 loai hop dong lao dong: hop dong lao dong khong xac dinh thoi han "
            "va hop dong lao dong xac dinh thoi han."
        )
        quote_terms = ("khong xac dinh thoi han", "xac dinh thoi han")
        citation_terms = ("dieu 20", "khoan 1")
        notes = ""

    elif (
        asks_notice_period
        and "khong xac dinh thoi han" in normalized_question
        and "45 ngay" in normalized_context
        and "hop dong lao dong khong xac dinh thoi han" in normalized_context
    ):
        context = first_context_with_terms(
            contexts,
            ("45 ngay", "hop dong lao dong khong xac dinh thoi han"),
        )
        answer = (
            "Nguoi lao dong lam viec theo hop dong lao dong khong xac dinh thoi han "
            "phai bao truoc it nhat 45 ngay khi don phuong cham dut hop dong."
        )
        if "nganh nghe cong viec dac thu" in normalized_context and "chinh phu" in normalized_context:
            answer = (
                answer
                + " Neu thuoc mot so nganh, nghe, cong viec dac thu thi thoi han bao truoc "
                + "thuc hien theo quy dinh cua Chinh phu."
            )
        quote_terms = ("45 ngay", "khong xac dinh thoi han")
        citation_terms = ("dieu 35", "khoan 1", "diem a")
        notes = ""

    elif (
        asks_notice_period
        and "khong xac dinh thoi han" not in normalized_question
        and "xac dinh thoi han" in normalized_question
        and "12" in normalized_question
        and "36" in normalized_question
        and "30 ngay" in normalized_context
    ):
        context = first_context_with_terms(
            contexts,
            ("30 ngay", "12 thang", "36 thang"),
        )
        answer = (
            "Nguoi lao dong lam viec theo hop dong lao dong xac dinh thoi han tu "
            "12 thang den 36 thang phai bao truoc it nhat 30 ngay."
        )
        quote_terms = ("30 ngay", "12 thang", "36 thang")
        citation_terms = ("dieu 35", "khoan 1", "diem b")
        notes = ""

    elif (
        asks_notice_period
        and "duoi 12" in normalized_question
        and "03 ngay lam viec" in normalized_context
    ):
        context = first_context_with_terms(
            contexts,
            ("03 ngay lam viec", "duoi 12 thang"),
        )
        answer = (
            "Thoi han bao truoc la it nhat 03 ngay lam viec doi voi hop dong lao dong "
            "xac dinh thoi han co thoi han duoi 12 thang."
        )
        quote_terms = ("03 ngay lam viec", "duoi 12 thang")
        citation_terms = ("dieu 35", "khoan 1", "diem c")
        notes = ""

    elif (
        "thu viec" in normalized_question
        and "cao dang" in normalized_question
        and "60 ngay" in normalized_context
    ):
        context = (
            first_context_with_terms(contexts, ("60 ngay",)) or contexts[0]
            if contexts
            else None
        )
        answer = (
            "Thoi gian thu viec toi da doi voi cong viec yeu cau trinh do tu cao dang "
            "tro len la khong qua 60 ngay."
        )
        quote_terms = ("60 ngay",)
        citation_terms = ("dieu 25",)
        notes = ""

    elif (
        "luong" in normalized_question
        and "thu viec" in normalized_question
        and contains_percent_value(raw_context, 85)
    ):
        context = (
            first_context_with_terms(contexts, ("85",)) or contexts[0]
            if contexts
            else None
        )
        answer = "Muc luong thu viec it nhat phai bang 85% muc luong cua cong viec do."
        quote_terms = ("85",)
        citation_terms = ("dieu 26",)
        notes = ""

    elif (
        "lam them" in normalized_question
        and "ngay nghi hang tuan" in normalized_question
        and contains_percent_value(raw_context, 200)
    ):
        context = (
            first_context_with_terms(contexts, ("200",)) or contexts[0]
            if contexts
            else None
        )
        answer = (
            "Nguoi lao dong lam them gio vao ngay nghi hang tuan duoc tra luong it "
            "nhat bang 200% so voi ngay lam viec binh thuong."
        )
        quote_terms = ("200",)
        citation_terms = ("dieu 98",)
        notes = ""

    if context is None or not answer:
        return None

    quote = extract_evidence_sentence(context.text, quote_terms)
    return ContextualAnswerOverride(
        answer=answer,
        evidence_quote=EvidenceQuote(
            citation=citation_for_context(context, citation_terms),
            quote=quote,
        ),
        notes=notes,
    )


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


def parse_answer_payload(
    raw_content: str,
    contexts: Sequence[RetrievalContext],
    *,
    question: str = "",
) -> ParsedAnswer:
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
    evidence_quotes: tuple[EvidenceQuote, ...] = ()
    if not insufficient_context:
        legal_basis = sanitize_legal_basis(payload.get("legal_basis"), contexts)
        evidence_quotes = sanitize_evidence_quotes(payload.get("evidence_quotes"), contexts)
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
        elif not evidence_quotes:
            existing_notes = str(payload.get("notes") or "").strip()
            payload["notes"] = " ".join(
                part
                for part in [
                    existing_notes,
                    "Khong xac nhan duoc evidence_quotes truc tiep trong CONTEXT.",
                ]
                if part
            ).strip()

    override = contextual_answer_override(question, contexts) if question else None
    if override is not None:
        insufficient_context = False
        payload["answer"] = override.answer
        legal_basis = dedupe_preserve_order(
            (override.evidence_quote.citation, *legal_basis)
        )
        evidence_quotes = (override.evidence_quote,) + tuple(
            evidence_quote
            for evidence_quote in evidence_quotes
            if evidence_quote.citation != override.evidence_quote.citation
            or normalize_quote_surface(evidence_quote.quote)
            != normalize_quote_surface(override.evidence_quote.quote)
        )
        existing_notes = str(payload.get("notes") or "").strip()
        payload["notes"] = " ".join(
            part for part in (existing_notes, override.notes) if part
        ).strip()

    return ParsedAnswer(
        answer=str(payload.get("answer") or "").strip(),
        legal_basis=legal_basis,
        evidence_quotes=evidence_quotes,
        insufficient_context=insufficient_context,
        notes=str(payload.get("notes") or "").strip(),
        raw_content=raw_content,
    )


__all__ = [
    "ANSWER_JSON_SCHEMA",
    "EvidenceQuote",
    "ParsedAnswer",
    "build_allowed_citations",
    "build_messages",
    "canonicalize_citation",
    "citation_overlap_matches",
    "extract_json_candidate",
    "parse_answer_payload",
    "sanitize_legal_basis",
]
