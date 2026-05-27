from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Sequence

from ...corpus_pipeline import normalize_for_matching
from ...retriever import RetrievalContext, dedupe_preserve_order
from .citation_guard import extract_evidence_sentence
from .formatter import format_answer_for_user
from .parser import parse_answer_payload
from .prompt import build_messages, order_contexts_for_answer
from .schema import ANSWER_JSON_SCHEMA, AnswerValidationResult, EvidenceQuote, ParsedAnswer
from .synthesis import (
    build_rule_based_answer_payload,
    select_contexts_for_grounded_generation,
)
from .validation import validate_grounded_answer


DEFAULT_ANSWER_CONTEXTS = 8
GENERIC_RULE_KEYWORDS = (
    "phai",
    "duoc",
    "khong",
    "quyen",
    "nghia vu",
    "hop dong",
    "tien luong",
    "thoi han",
    "nguoi lao dong",
    "nguoi su dung",
    "bao dam",
    "tro cap",
    "boi thuong",
)


@dataclass(frozen=True)
class GroundedAnswerResult:
    question: str
    answer: str
    parsed: ParsedAnswer
    validation: AnswerValidationResult
    contexts: tuple[RetrievalContext, ...]
    provider: str
    model: str
    generation_method: str
    raw_content: str

    def to_json(self) -> dict[str, object]:
        return {
            "question": self.question,
            "answer": self.answer,
            "legal_basis": list(self.parsed.legal_basis),
            "evidence_quotes": [asdict(quote) for quote in self.parsed.evidence_quotes],
            "insufficient_context": self.parsed.insufficient_context,
            "notes": self.parsed.notes,
            "validation": asdict(self.validation),
            "provider": self.provider,
            "model": self.model,
            "generation_method": self.generation_method,
            "raw_content": self.raw_content,
        }


def _context_snippet(context: RetrievalContext, question: str) -> str:
    normalized_question = normalize_for_matching(question)
    preferred_terms = [
        token
        for token in normalized_question.split()
        if len(token) >= 4 and not token.isdigit()
    ][:6]
    source_text = str(context.payload.get("retrieval_text") or context.text)
    snippet = extract_evidence_sentence(source_text, preferred_terms) or source_text.strip()[:400]
    if _is_low_information_snippet(snippet):
        snippet = _best_rule_snippet(source_text, preferred_terms) or snippet
    if _is_low_information_snippet(snippet):
        snippet = _best_rule_snippet(context.text, preferred_terms) or snippet
    normalized_snippet = normalize_for_matching(snippet)
    if normalized_snippet.startswith("dieu ") and len(normalized_snippet.split()) <= 6:
        snippet = _best_rule_snippet(context.text, preferred_terms) or snippet
    return snippet


def _is_low_information_snippet(snippet: str) -> bool:
    normalized = normalize_for_matching(snippet)
    if not normalized:
        return True
    tokens = [token for token in normalized.split() if len(token) >= 2]
    if len(tokens) < 8:
        return True
    return not any(keyword in normalized for keyword in GENERIC_RULE_KEYWORDS)


def _best_rule_snippet(text: str, preferred_terms: Sequence[str]) -> str:
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in re.split(r"\n+", text)
        if line.strip()
    ]
    candidates = [line for line in lines if not _is_low_information_snippet(line)]
    if not candidates:
        return ""

    normalized_terms = tuple(normalize_for_matching(term) for term in preferred_terms if term)
    for line in candidates:
        normalized_line = normalize_for_matching(line)
        if normalized_terms and all(term in normalized_line for term in normalized_terms[:3]):
            return line[:500]
    for line in candidates:
        normalized_line = normalize_for_matching(line)
        if normalized_terms and any(term in normalized_line for term in normalized_terms):
            return line[:500]
    return candidates[0][:500]


def _context_rank(context: RetrievalContext) -> int:
    try:
        return int(context.payload.get("normative_rank") or 0)
    except (TypeError, ValueError):
        return 0


def _context_article_key(context: RetrievalContext) -> tuple[str, str]:
    document_id = str(context.payload.get("document_id") or "")
    article_number = str(context.payload.get("article_number") or context.chunk_id)
    return document_id, article_number


def select_answer_evidence_contexts(
    contexts: Sequence[RetrievalContext],
    *,
    max_citations: int,
) -> tuple[RetrievalContext, ...]:
    selected: list[RetrievalContext] = []
    seen_keys: set[tuple[str, str]] = set()
    rank_limits = {1: 3, 2: 2, 3: 2}

    for rank in (1, 2, 3):
        added_for_rank = 0
        for context in contexts:
            if _context_rank(context) != rank:
                continue
            key = _context_article_key(context)
            if key in seen_keys:
                continue
            selected.append(context)
            seen_keys.add(key)
            added_for_rank += 1
            if len(selected) >= max_citations or added_for_rank >= rank_limits[rank]:
                break
        if len(selected) >= max_citations:
            break

    for context in contexts:
        if len(selected) >= max_citations:
            break
        key = _context_article_key(context)
        if key in seen_keys:
            continue
        selected.append(context)
        seen_keys.add(key)

    return tuple(selected)


def _first_citation_for_rank(contexts: Sequence[RetrievalContext], *ranks: int) -> str:
    for context in contexts:
        if _context_rank(context) in ranks and context.citation_text:
            return context.citation_text
    return ""


def _query_lead(question: str, contexts: Sequence[RetrievalContext]) -> str:
    normalized = normalize_for_matching(question)
    citations = tuple(context.citation_text for context in contexts if context.citation_text)
    primary = citations[0] if citations else ""
    second_law = _first_citation_for_rank(contexts[1:], 1)
    guidance = _first_citation_for_rank(contexts, 2, 3)
    secondary = guidance or second_law or (citations[1] if len(citations) > 1 else "")

    if "14 tuoi" in normalized or "chua du 15 tuoi" in normalized:
        return (
            f"Người chưa đủ 15 tuổi chỉ được làm việc khi đáp ứng các điều kiện luật định; "
            f"căn cứ trước hết là {primary}"
            + (f", đồng thời đối chiếu hướng dẫn tại {secondary}." if secondary else ".")
        )
    if "nghi huu" in normalized or "huu tri" in normalized:
        return (
            f"Cần xác định tuổi nghỉ hưu theo lộ trình của Bộ luật Lao động và nghị định hướng dẫn; "
            f"căn cứ trước hết là {primary}"
            + (f", sau đó đối chiếu {secondary}." if secondary else ".")
        )
    if "noi dung" in normalized and "hop dong" in normalized:
        return (
            f"Hợp đồng lao động phải có các nội dung chủ yếu theo {primary}"
            + (f"; văn bản hướng dẫn chi tiết là {secondary}." if secondary else ".")
        )
    if "tranh chap" in normalized and ("sa thai" in normalized or "kien" in normalized):
        return (
            f"Với tranh chấp sa thải, cần kiểm tra quy định về hòa giải bắt buộc trước khi kiện theo {primary}"
            + (f" và thẩm quyền/tố tụng theo {secondary}." if secondary else ".")
        )
    if "trai luat" in normalized and "boi thuong" in normalized:
        return f"Nghĩa vụ bồi thường của người lao động khi đơn phương chấm dứt hợp đồng trái pháp luật được xác định trực tiếp theo {primary}."
    if "thay doi co cau" in normalized:
        return (
            f"Khi thay đổi cơ cấu dẫn đến mất việc, người sử dụng lao động phải xem xét nghĩa vụ theo {primary}"
            + (f" và trợ cấp mất việc theo {second_law}." if second_law else ".")
            + (f" Văn bản hướng dẫn chi tiết được dùng kèm là {guidance}." if guidance else "")
        )
    if "khong can bao truoc" in normalized or "khong phai bao truoc" in normalized:
        return f"Người lao động được nghỉ việc không cần báo trước trong các trường hợp được liệt kê tại {primary}."
    if "lam them" in normalized and ("gioi han" in normalized or "theo thang" in normalized):
        return (
            f"Điều kiện và giới hạn làm thêm giờ được xác định trực tiếp theo {primary}"
            + (f"; giới hạn chi tiết được hướng dẫn thêm tại {secondary}." if secondary else ".")
        )
    return f"Căn cứ vào {primary}, có cơ sở pháp lý để trả lời trong phạm vi dữ liệu được truy xuất."


def build_extractive_answer_payload(
    question: str,
    contexts: Sequence[RetrievalContext],
    *,
    max_citations: int = 4,
) -> dict[str, object]:
    ordered_contexts = order_contexts_for_answer(contexts)
    if not ordered_contexts:
        return {
            "answer": (
                "Không đủ căn cứ trong dữ liệu hiện có để kết luận chắc chắn. "
                "Cần bổ sung ngữ cảnh pháp lý trực tiếp liên quan đến câu hỏi."
            ),
            "legal_basis": [],
            "evidence_quotes": [],
            "insufficient_context": True,
            "notes": "No retrieved contexts were available.",
        }

    rule_based_payload = build_rule_based_answer_payload(question, ordered_contexts)
    if rule_based_payload is not None:
        return rule_based_payload

    selected = select_answer_evidence_contexts(ordered_contexts, max_citations=max_citations)
    legal_basis = dedupe_preserve_order(
        tuple(context.citation_text for context in selected if context.citation_text)
    )
    evidence_quotes_list: list[EvidenceQuote] = []
    for context in selected:
        if not context.citation_text:
            continue
        snippet = _context_snippet(context, question)
        if _is_low_information_snippet(snippet):
            continue
        evidence_quotes_list.append(
            EvidenceQuote(
                citation=context.citation_text,
                quote=snippet,
            )
        )
    evidence_quotes = tuple(evidence_quotes_list)
    lead = _query_lead(question, selected)
    detail_lines = [
        f"- {quote.quote} ({quote.citation})"
        for quote in evidence_quotes
    ]
    answer_parts = [
        lead,
        "",
        "Nội dung cụ thể như sau:",
        *detail_lines,
        "",
        "Tóm lại:",
        f"- Ưu tiên áp dụng căn cứ cấp cao hơn trước: {legal_basis[0]}.",
    ]
    if len(legal_basis) > 1:
        answer_parts.append(f"- Dùng căn cứ hướng dẫn/chi tiết khi phù hợp: {legal_basis[1]}.")
    return {
        "answer": "\n".join(answer_parts),
        "legal_basis": list(legal_basis),
        "evidence_quotes": [asdict(quote) for quote in evidence_quotes],
        "insufficient_context": False,
        "notes": "",
    }


def generate_grounded_answer(
    question: str,
    contexts: Sequence[RetrievalContext],
    *,
    provider: str = "auto",
    model: str = "",
    temperature: float = 0.0,
    max_context_chars: int = 9000,
    max_context_tokens: int | None = 1800,
    max_answer_contexts: int = DEFAULT_ANSWER_CONTEXTS,
    fallback_on_invalid: bool = True,
) -> GroundedAnswerResult:
    selected_contexts = select_contexts_for_grounded_generation(
        question,
        contexts,
        max_contexts=max_answer_contexts,
    )
    generation_method = "extractive"
    response_provider = provider
    response_model = model

    raw_content = ""
    if provider not in {"extractive", "local", "none"}:
        try:
            from ...llm import chat_completion

            response = chat_completion(
                provider="groq" if provider == "auto" else provider,
                model=model or None,
                messages=build_messages(
                    question,
                    selected_contexts,
                    max_context_chars=max_context_chars,
                    max_context_tokens=max_context_tokens,
                ),
                temperature=temperature,
                json_schema=ANSWER_JSON_SCHEMA,
                json_schema_name="grounded_legal_answer",
            )
            raw_content = response.content
            response_provider = response.provider
            response_model = response.model
            generation_method = "llm"
        except Exception as exc:
            if not fallback_on_invalid:
                raise
            raw_content = json.dumps(
                {
                    **build_extractive_answer_payload(question, selected_contexts),
                    "notes": f"LLM generation unavailable or invalid; used extractive fallback. {exc}",
                },
                ensure_ascii=False,
            )
            response_provider = "extractive"
            response_model = "deterministic"
            generation_method = "extractive_fallback"
    else:
        raw_content = json.dumps(
            build_extractive_answer_payload(question, selected_contexts),
            ensure_ascii=False,
        )
        response_provider = "extractive"
        response_model = "deterministic"

    parsed = parse_answer_payload(raw_content, selected_contexts, question=question)
    validation = validate_grounded_answer(parsed, selected_contexts)
    if fallback_on_invalid and generation_method == "llm" and not validation.passed:
        raw_content = json.dumps(
            build_extractive_answer_payload(question, selected_contexts),
            ensure_ascii=False,
        )
        parsed = parse_answer_payload(raw_content, selected_contexts, question=question)
        validation = validate_grounded_answer(parsed, selected_contexts)
        response_provider = "extractive"
        response_model = "deterministic"
        generation_method = "extractive_fallback"

    return GroundedAnswerResult(
        question=question,
        answer=format_answer_for_user(parsed, question=question),
        parsed=parsed,
        validation=validation,
        contexts=selected_contexts,
        provider=response_provider,
        model=response_model,
        generation_method=generation_method,
        raw_content=raw_content,
    )


__all__ = [
    "DEFAULT_ANSWER_CONTEXTS",
    "GroundedAnswerResult",
    "build_extractive_answer_payload",
    "generate_grounded_answer",
    "select_answer_evidence_contexts",
]
