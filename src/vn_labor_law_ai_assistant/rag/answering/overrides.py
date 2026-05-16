from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ...corpus_pipeline import normalize_for_matching
from ...retriever import RetrievalContext
from .citation_guard import (
    citation_for_context,
    combined_context_text,
    contains_percent_value,
    extract_evidence_sentence,
    first_context_with_terms,
)
from .schema import EvidenceQuote

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


__all__ = [
    "ContextualAnswerOverride",
    "contextual_answer_override",
]
