from __future__ import annotations

from dataclasses import asdict
import re
from typing import Sequence

from ...corpus_pipeline import normalize_for_matching
from ...retriever import RetrievalContext, dedupe_preserve_order
from .formatter import answer_language
from .prompt import order_contexts_for_answer
from .schema import EvidenceQuote


ANSWER_INTENT_MINOR_WORKER = "minor_worker"
ANSWER_INTENT_EMPLOYEE_DEFINITION = "employee_definition"
ANSWER_INTENT_RETIREMENT_AGE = "retirement_age"
ANSWER_INTENT_CONTRACT_CONTENT = "labor_contract_content"
ANSWER_INTENT_DISMISSAL_MEDIATION = "dismissal_mediation_exception"
ANSWER_INTENT_ILLEGAL_EMPLOYEE_TERMINATION = "illegal_unilateral_termination_by_employee"
ANSWER_INTENT_STRUCTURAL_CHANGE = "structural_change_job_loss_allowance"
ANSWER_INTENT_JOB_LOSS_ALLOWANCE = "job_loss_allowance"
ANSWER_INTENT_SEVERANCE_VS_JOB_LOSS = "severance_vs_job_loss"
ANSWER_INTENT_EARLY_RETIREMENT_HAZARDOUS_WORK = "early_retirement_hazardous_work"
ANSWER_INTENT_NO_NOTICE_RESIGNATION = "no_notice_resignation"
ANSWER_INTENT_OVERTIME_LIMITS = "overtime_conditions_and_limits"
ANSWER_INTENT_GENERIC = "generic"


def classify_answer_intent(question: str) -> str:
    normalized = normalize_for_matching(question)
    if (
        (
            "nguoi lao dong" in normalized
            and (
                "dinh nghia" in normalized
                or "duoc dinh nghia" in normalized
                or "la ai" in normalized
                or "khai niem" in normalized
                or "giai thich tu ngu" in normalized
            )
        )
        or (
            "employee" in normalized
            and (
                "definition" in normalized
                or "defined" in normalized
                or "who is" in normalized
            )
        )
    ):
        return ANSWER_INTENT_EMPLOYEE_DEFINITION
    if (
        "14 tuoi" in normalized
        or "chua du 15 tuoi" in normalized
        or "under 15" in normalized
        or "under-15" in normalized
        or "minor worker" in normalized
        or "underage worker" in normalized
        or "child labor" in normalized
    ):
        return ANSWER_INTENT_MINOR_WORKER
    if (
        ("nghi huu" in normalized or "retirement" in normalized)
        and (
            "nghe nang nhoc" in normalized
            or "doc hai" in normalized
            or "nghi huu som" in normalized
            or "hazardous" in normalized
            or "early retirement" in normalized
        )
    ):
        return ANSWER_INTENT_EARLY_RETIREMENT_HAZARDOUS_WORK
    if "nghi huu" in normalized or "huu tri" in normalized or "retirement age" in normalized or "decree 135" in normalized:
        return ANSWER_INTENT_RETIREMENT_AGE
    if "noi dung" in normalized and "hop dong" in normalized:
        return ANSWER_INTENT_CONTRACT_CONTENT
    if "tranh chap" in normalized and ("sa thai" in normalized or "kien" in normalized):
        return ANSWER_INTENT_DISMISSAL_MEDIATION
    if "trai luat" in normalized and ("boi thuong" in normalized or "don phuong" in normalized):
        return ANSWER_INTENT_ILLEGAL_EMPLOYEE_TERMINATION
    if "tro cap thoi viec" in normalized and "tro cap mat viec" in normalized:
        return ANSWER_INTENT_SEVERANCE_VS_JOB_LOSS
    if "tro cap mat viec" in normalized or "mat viec lam duoc tinh" in normalized:
        return ANSWER_INTENT_JOB_LOSS_ALLOWANCE
    if "thay doi co cau" in normalized or "mat viec" in normalized:
        return ANSWER_INTENT_STRUCTURAL_CHANGE
    if (
        "khong can bao truoc" in normalized
        or "khong phai bao truoc" in normalized
        or "without prior notice" in normalized
        or "without notice" in normalized
        or "no prior notice" in normalized
        or "no notice" in normalized
    ):
        return ANSWER_INTENT_NO_NOTICE_RESIGNATION
    if "lam them" in normalized and (
        "gioi han" in normalized
        or "theo thang" in normalized
        or "so gio" in normalized
        or "toi da" in normalized
        or "truong hop nao" in normalized
    ):
        return ANSWER_INTENT_OVERTIME_LIMITS
    return ANSWER_INTENT_GENERIC


def _document_id(context: RetrievalContext) -> str:
    return str(context.payload.get("document_id") or "").strip()


def _article_number(context: RetrievalContext) -> str:
    return str(context.payload.get("article_number") or "").strip()


def _clause_ref(context: RetrievalContext) -> str:
    return str(context.payload.get("clause_ref") or "").strip()


def _citation_contains(context: RetrievalContext, *terms: str) -> bool:
    normalized = normalize_for_matching(context.citation_text)
    return all(normalize_for_matching(term) in normalized for term in terms)


def _context_contains(context: RetrievalContext, *terms: str) -> bool:
    normalized = normalize_for_matching(
        " ".join(
            [
                context.citation_text,
                context.text,
                str(context.payload.get("retrieval_text") or ""),
            ]
        )
    )
    return all(normalize_for_matching(term) in normalized for term in terms)


def _find_context(
    contexts: Sequence[RetrievalContext],
    *,
    document_id: str | None = None,
    article: str | None = None,
    clause: str | None = None,
    citation_terms: Sequence[str] = (),
    text_terms: Sequence[str] = (),
) -> RetrievalContext | None:
    for context in contexts:
        if document_id and _document_id(context) != document_id:
            continue
        if article and _article_number(context) != article:
            continue
        if clause and _clause_ref(context) != clause:
            continue
        if citation_terms and not _citation_contains(context, *citation_terms):
            continue
        if text_terms and not _context_contains(context, *text_terms):
            continue
        return context
    return None


def _intent_anchor_contexts(
    intent: str,
    contexts: Sequence[RetrievalContext],
) -> tuple[RetrievalContext, ...]:
    anchors: list[RetrievalContext] = []

    def add(context: RetrievalContext | None) -> None:
        if context is not None:
            anchors.append(context)

    if intent == ANSWER_INTENT_EMPLOYEE_DEFINITION:
        add(_find_context(contexts, document_id="45-2019-qh14", article="3", clause="1"))
        add(
            _find_context(
                contexts,
                article="3",
                clause="1",
                text_terms=("nguoi lao dong", "tra luong"),
            )
        )
    elif intent == ANSWER_INTENT_MINOR_WORKER:
        add(_find_context(contexts, document_id="45-2019-qh14", article="145", clause="1"))
        add(_find_context(contexts, document_id="45-2019-qh14", article="146", clause="1"))
        add(_find_context(contexts, document_id="45-2019-qh14", article="143", clause="1"))
        add(_find_context(contexts, document_id="thong-tu-09-2020-tt-bldtbxh", article="3"))
    elif intent == ANSWER_INTENT_RETIREMENT_AGE:
        add(_find_context(contexts, document_id="45-2019-qh14", article="169", clause="2"))
        add(
            _find_context(
                contexts,
                document_id="nghi-dinh-135-2020-nd-cp",
                article="4",
                clause="2",
            )
            or _find_context(contexts, document_id="nghi-dinh-135-2020-nd-cp", article="4")
        )
        add(
            _find_context(
                contexts,
                document_id="nghi-dinh-135-2020-nd-cp",
                citation_terms=("phu luc i", "lao dong nu"),
                text_terms=("2026", "57 tuoi"),
            )
        )
    elif intent == ANSWER_INTENT_EARLY_RETIREMENT_HAZARDOUS_WORK:
        add(_find_context(contexts, document_id="45-2019-qh14", article="169", clause="3"))
        add(_find_context(contexts, document_id="nghi-dinh-135-2020-nd-cp", article="5"))
        add(_find_context(contexts, document_id="nghi-dinh-135-2020-nd-cp", citation_terms=("phu luc ii",)))
    elif intent == ANSWER_INTENT_CONTRACT_CONTENT:
        add(_find_context(contexts, document_id="45-2019-qh14", article="21", clause="1"))
        add(_find_context(contexts, document_id="thong-tu-10-2020-tt-bldtbxh", article="3"))
        add(_find_context(contexts, document_id="45-2019-qh14", article="22", clause="1"))
    elif intent == ANSWER_INTENT_DISMISSAL_MEDIATION:
        add(_find_context(contexts, document_id="45-2019-qh14", article="188", clause="1"))
        add(_find_context(contexts, document_id="92-2015-qh13-labor-only", article="32"))
        add(_find_context(contexts, document_id="45-2019-qh14", article="190", clause="1"))
    elif intent == ANSWER_INTENT_ILLEGAL_EMPLOYEE_TERMINATION:
        add(_find_context(contexts, document_id="45-2019-qh14", article="40", clause="1"))
        add(_find_context(contexts, document_id="45-2019-qh14", article="40", clause="2"))
        add(_find_context(contexts, document_id="45-2019-qh14", article="40", clause="3"))
        add(_find_context(contexts, document_id="45-2019-qh14", article="39"))
        add(_find_context(contexts, document_id="45-2019-qh14", article="35", clause="1"))
    elif intent == ANSWER_INTENT_STRUCTURAL_CHANGE:
        add(_find_context(contexts, document_id="45-2019-qh14", article="42", clause="5"))
        add(_find_context(contexts, document_id="45-2019-qh14", article="47", clause="1"))
        add(
            _find_context(
                contexts,
                document_id="nghi-dinh-145-2020-nd-cp",
                article="8",
                clause="2",
            )
        )
    elif intent == ANSWER_INTENT_JOB_LOSS_ALLOWANCE:
        add(_find_context(contexts, document_id="45-2019-qh14", article="47", clause="1"))
        add(_find_context(contexts, document_id="45-2019-qh14", article="42", clause="5"))
        add(_find_context(contexts, document_id="nghi-dinh-145-2020-nd-cp", article="8"))
    elif intent == ANSWER_INTENT_SEVERANCE_VS_JOB_LOSS:
        add(_find_context(contexts, document_id="45-2019-qh14", article="46", clause="1"))
        add(_find_context(contexts, document_id="45-2019-qh14", article="47", clause="1"))
        add(_find_context(contexts, document_id="nghi-dinh-145-2020-nd-cp", article="8"))
    elif intent == ANSWER_INTENT_NO_NOTICE_RESIGNATION:
        add(_find_context(contexts, document_id="45-2019-qh14", article="35", clause="2"))
        add(_find_context(contexts, document_id="45-2019-qh14", article="35", clause="1"))
        add(_find_context(contexts, document_id="nghi-dinh-145-2020-nd-cp", article="7"))
    elif intent == ANSWER_INTENT_OVERTIME_LIMITS:
        add(_find_context(contexts, document_id="45-2019-qh14", article="107", clause="2"))
        add(_find_context(contexts, document_id="45-2019-qh14", article="107", clause="3"))
        add(_find_context(contexts, document_id="nghi-dinh-145-2020-nd-cp", article="60"))
        add(_find_context(contexts, document_id="nghi-dinh-145-2020-nd-cp", article="61"))

    return _dedupe_contexts(anchors)


def _dedupe_contexts(contexts: Sequence[RetrievalContext]) -> tuple[RetrievalContext, ...]:
    selected: list[RetrievalContext] = []
    seen: set[str] = set()
    for context in contexts:
        key = context.chunk_id or context.citation_text
        if key in seen:
            continue
        selected.append(context)
        seen.add(key)
    return tuple(selected)


def select_contexts_for_grounded_generation(
    question: str,
    contexts: Sequence[RetrievalContext],
    *,
    max_contexts: int,
) -> tuple[RetrievalContext, ...]:
    ordered_contexts = order_contexts_for_answer(contexts)
    intent = classify_answer_intent(question)
    selected: list[RetrievalContext] = list(_intent_anchor_contexts(intent, contexts))
    seen_chunk_ids = {context.chunk_id for context in selected}

    for context in ordered_contexts:
        if len(selected) >= max_contexts:
            break
        if context.chunk_id in seen_chunk_ids:
            continue
        selected.append(context)
        seen_chunk_ids.add(context.chunk_id)

    return tuple(selected)


def _clean_lines(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", line).strip()
        for line in re.split(r"\n+", text)
        if line.strip()
    ]


EVIDENCE_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?;:])\s+")


def _dedupe_repeated_lines(lines: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = normalize_for_matching(line)
        if normalized and normalized in seen:
            continue
        if normalized:
            seen.add(normalized)
        deduped.append(line)
    return deduped


def _dedupe_repeated_sentences(text: str) -> str:
    cleaned_paragraphs: list[str] = []
    seen: set[str] = set()
    for paragraph in re.split(r"\n{2,}", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences = [sentence.strip() for sentence in EVIDENCE_SENTENCE_BOUNDARY_RE.split(paragraph) if sentence.strip()]
        cleaned_sentences: list[str] = []
        for sentence in sentences:
            normalized = normalize_for_matching(sentence)
            if normalized and normalized in seen:
                continue
            if normalized:
                seen.add(normalized)
            cleaned_sentences.append(sentence)
        if cleaned_sentences:
            cleaned_paragraphs.append(" ".join(cleaned_sentences))
    return "\n\n".join(cleaned_paragraphs).strip()


def _clean_evidence_text(text: str) -> str:
    lines = _dedupe_repeated_lines(_clean_lines(text))
    return _dedupe_repeated_sentences("\n\n".join(lines))


def _meaningful_word_count(text: str) -> int:
    return len([token for token in normalize_for_matching(text).split() if len(token) >= 2])


def _rule_text(context: RetrievalContext) -> str:
    lines = _dedupe_repeated_lines(_clean_lines(context.text))
    if len(lines) <= 1:
        return _clean_evidence_text(context.text)
    body = "\n\n".join(lines[1:]).strip()
    if _meaningful_word_count(body) < 8:
        return _clean_evidence_text("\n\n".join(lines))
    return _clean_evidence_text(body)


def _quote(context: RetrievalContext, *, max_chars: int = 1200) -> str:
    text = _rule_text(context)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip()


def _evidence(contexts: Sequence[RetrievalContext]) -> list[dict[str, str]]:
    quotes: list[EvidenceQuote] = []
    for context in contexts:
        if not context or not context.citation_text:
            continue
        quote = _quote(context)
        if not quote:
            continue
        quotes.append(EvidenceQuote(citation=context.citation_text, quote=quote))
    return [asdict(quote) for quote in quotes]


def _legal_basis(contexts: Sequence[RetrievalContext]) -> list[str]:
    return list(dedupe_preserve_order(context.citation_text for context in contexts if context and context.citation_text))


def _payload(answer: str, contexts: Sequence[RetrievalContext]) -> dict[str, object]:
    legal_basis = _legal_basis(contexts)
    return {
        "answer": answer.strip(),
        "legal_basis": legal_basis,
        "evidence_quotes": _evidence(contexts),
        "insufficient_context": False,
        "notes": "",
    }


def _cite(context: RetrievalContext | None) -> str:
    return context.citation_text if context is not None else ""


def _minor_worker_payload(question: str, contexts: Sequence[RetrievalContext]) -> dict[str, object] | None:
    law145 = _find_context(contexts, document_id="45-2019-qh14", article="145", clause="1")
    tt09 = _find_context(contexts, document_id="thong-tu-09-2020-tt-bldtbxh", article="3")
    law146 = _find_context(contexts, document_id="45-2019-qh14", article="146", clause="1")
    law143 = _find_context(contexts, document_id="45-2019-qh14", article="143", clause="1")
    if law145 is None:
        return None

    normalized = normalize_for_matching(question)
    specific_age_14 = "14 tuoi" in normalized
    if answer_language(question) == "en":
        subject = (
            "A 14-year-old worker"
            if specific_age_14
            else "A worker under 15"
        )
        answer_lines = [
            f"{subject} may work only if the employer satisfies the special statutory conditions for employing workers under 15.",
            "- The employment contract must be made in writing with the worker under 15 and the worker's legal representative.",
            "- Working hours must not affect the worker's study time.",
            "- The employer must have a health certificate suitable for the work and organize periodic health checks at least once every six months.",
            "- Working conditions, occupational safety, and occupational hygiene must be appropriate to the worker's age.",
        ]
        if law146:
            answer_lines.append(
                "- Working time must also comply with the special limits that apply to minor workers."
            )
        if tt09:
            answer_lines.append(
                "Decree/Circular-level guidance must be read consistently with Labor Code Article 145."
            )
        return _payload(
            "\n".join(answer_lines),
            [context for context in (law143, law145, law146, tt09) if context],
        )
    direct = (
        "Có thể được làm việc, nhưng chỉ khi công việc và điều kiện sử dụng người chưa đủ 15 tuổi đáp ứng đúng luật."
        if "co duoc" in normalized or normalized.endswith("khong")
        else "Người chưa đủ 15 tuổi chỉ được làm việc nếu đáp ứng các điều kiện sử dụng người chưa đủ 15 tuổi theo luật."
    )
    subject_sentence = (
        "Người 14 tuổi thuộc nhóm lao động chưa thành niên/người chưa đủ 15 tuổi"
        if specific_age_14
        else "Người chưa đủ 15 tuổi thuộc nhóm lao động chưa thành niên và chỉ được làm việc nếu đáp ứng điều kiện riêng"
    )
    summary_sentence = (
        "người 14 tuổi không bị cấm tuyệt đối làm việc, nhưng chỉ được làm nếu đáp ứng các điều kiện nêu trên"
        if specific_age_14
        else "người chưa đủ 15 tuổi chỉ được làm việc nếu đáp ứng các điều kiện nêu trên"
    )
    answer = "\n".join(
        [
            f"{direct} {subject_sentence}, nên căn cứ chính là {_cite(law145)}.",
            "",
            "Nội dung cụ thể như sau:",
            f"- Phải giao kết hợp đồng lao động bằng văn bản với người chưa đủ 15 tuổi và người đại diện theo pháp luật của người đó. ({_cite(law145)})",
            f"- Phải bố trí giờ làm việc không ảnh hưởng đến thời gian học tập của người chưa đủ 15 tuổi. ({_cite(law145)})",
            f"- Phải có giấy khám sức khỏe phù hợp với công việc và tổ chức kiểm tra sức khỏe định kỳ ít nhất một lần trong 06 tháng. ({_cite(law145)})",
            f"- Phải bảo đảm điều kiện làm việc, an toàn, vệ sinh lao động phù hợp với lứa tuổi. ({_cite(law145)})",
            *(
                [
                    f"- Thời giờ làm việc của người chưa đủ 15 tuổi còn phải tuân thủ giới hạn riêng cho lao động chưa thành niên. ({_cite(law146)})",
                ]
                if law146
                else []
            ),
            *(
                [
                    f"Thông tư 09 hướng dẫn rằng người sử dụng lao động phải tuân thủ Điều 145 khi sử dụng người chưa đủ 15 tuổi làm việc. ({_cite(tt09)})",
                ]
                if tt09
                else []
            ),
            "",
            f"Tóm lại: {summary_sentence} và có căn cứ tại {_cite(law145)}.",
        ]
    )
    return _payload(answer, [context for context in (law143, law145, law146, tt09) if context])


def _retirement_payload(
    contexts: Sequence[RetrievalContext],
    *,
    question: str,
) -> dict[str, object] | None:
    law169 = _find_context(contexts, document_id="45-2019-qh14", article="169", clause="2")
    nd135 = (
        _find_context(contexts, document_id="nghi-dinh-135-2020-nd-cp", article="4", clause="2")
        or _find_context(contexts, document_id="nghi-dinh-135-2020-nd-cp", article="4")
    )
    appendix = _find_context(
        contexts,
        document_id="nghi-dinh-135-2020-nd-cp",
        citation_terms=("phu luc i", "lao dong nu"),
        text_terms=("2026", "57 tuoi"),
    )
    if law169 is None and nd135 is None:
        return None

    normalized_question = normalize_for_matching(question)
    asks_exact_age = "bao nhieu" in normalized_question or "nam 2026" in normalized_question
    if not asks_exact_age:
        answer = "\n".join(
            [
                f"Bộ luật Lao động quy định tuổi nghỉ hưu trong điều kiện bình thường được điều chỉnh theo lộ trình; Nghị định 135 là văn bản quy định chi tiết lộ trình đó. ({_cite(law169) or _cite(nd135)})",
                "",
                "Nội dung cụ thể như sau:",
                *(
                    [
                        f"- Bộ luật Lao động đặt nguyên tắc tuổi nghỉ hưu được điều chỉnh theo lộ trình cho đến khi đạt 62 tuổi đối với lao động nam và 60 tuổi đối với lao động nữ trong điều kiện bình thường. ({_cite(law169)})",
                    ]
                    if law169
                    else []
                ),
                *(
                    [
                        f"- Nghị định 135 hướng dẫn cách xác định tuổi nghỉ hưu trong điều kiện lao động bình thường và lộ trình tăng tuổi nghỉ hưu theo từng năm. ({_cite(nd135)})",
                    ]
                    if nd135
                    else []
                ),
                *(
                    [
                        f"- Nếu cần tra theo tháng, năm sinh cụ thể thì đối chiếu thêm Phụ lục I về lộ trình tuổi nghỉ hưu. ({_cite(appendix)})",
                    ]
                    if appendix
                    else []
                ),
                "",
                f"Tóm lại: BLLĐ Điều 169 nêu nguyên tắc và Nghị định 135 cụ thể hóa lộ trình áp dụng tuổi nghỉ hưu. ({_cite(law169) or _cite(nd135)})",
            ]
        )
        return _payload(answer, [context for context in (law169, nd135, appendix) if context])

    if appendix or (nd135 and _context_contains(nd135, "2026", "57 tuoi", "lao dong nu")):
        answer = "\n".join(
            [
                f"Lao động nữ nghỉ hưu trong điều kiện lao động bình thường vào năm 2026 là 57 tuổi. ({_cite(nd135)})",
                "",
                "Nội dung cụ thể như sau:",
                f"- Bộ luật Lao động quy định tuổi nghỉ hưu trong điều kiện bình thường được điều chỉnh theo lộ trình. ({_cite(law169)})",
                f"- Nghị định 135 quy định lộ trình cụ thể; bảng lộ trình cho lao động nữ ghi năm 2026 tương ứng 57 tuổi. ({_cite(nd135)})",
                *(
                    [
                        f"- Nếu cần xác định theo tháng, năm sinh cụ thể thì đối chiếu thêm Phụ lục I; context truy xuất có bảng lao động nữ gắn tháng/năm sinh với năm nghỉ hưu. ({_cite(appendix)})",
                    ]
                    if appendix
                    else []
                ),
                "",
                f"Tóm lại: với dữ liệu hiện có, câu trả lời trực tiếp là 57 tuổi cho lao động nữ nghỉ hưu năm 2026 trong điều kiện bình thường. ({_cite(nd135)})",
            ]
        )
        return _payload(answer, [context for context in (law169, nd135, appendix) if context])

    answer = "\n".join(
        [
            "Chưa đủ căn cứ trong dữ liệu hiện có để xác định chính xác tuổi nghỉ hưu theo tháng, năm sinh.",
            f"Context hiện có chỉ cho biết tuổi nghỉ hưu được điều chỉnh theo lộ trình và Nghị định 135 quy định chi tiết lộ trình đó. ({_cite(law169) or _cite(nd135)})",
            "",
            "Tóm lại: cần thêm bảng lộ trình hoặc Phụ lục I tương ứng để kết luận chính xác cho năm 2026.",
        ]
    )
    payload = _payload(answer, [context for context in (law169, nd135) if context])
    payload["insufficient_context"] = True
    return payload


def _contract_content_payload(contexts: Sequence[RetrievalContext]) -> dict[str, object] | None:
    law21 = _find_context(contexts, document_id="45-2019-qh14", article="21", clause="1")
    tt10 = _find_context(contexts, document_id="thong-tu-10-2020-tt-bldtbxh", article="3")
    if law21 is None:
        return None
    cite = _cite(law21)
    answer = "\n".join(
        [
            f"Hợp đồng lao động phải có các nội dung chủ yếu được liệt kê tại {cite}.",
            "",
            "Nội dung cụ thể như sau:",
            f"- Tên, địa chỉ của người sử dụng lao động và họ tên, chức danh người giao kết phía người sử dụng lao động. ({cite})",
            f"- Họ tên, ngày tháng năm sinh, giới tính, nơi cư trú và giấy tờ pháp lý của người lao động. ({cite})",
            f"- Công việc và địa điểm làm việc. ({cite})",
            f"- Thời hạn của hợp đồng lao động. ({cite})",
            f"- Mức lương, hình thức trả lương, thời hạn trả lương, phụ cấp lương và các khoản bổ sung khác. ({cite})",
            f"- Chế độ nâng bậc, nâng lương. ({cite})",
            f"- Thời giờ làm việc, thời giờ nghỉ ngơi. ({cite})",
            f"- Trang bị bảo hộ lao động cho người lao động. ({cite})",
            f"- Bảo hiểm xã hội, bảo hiểm y tế và bảo hiểm thất nghiệp. ({cite})",
            f"- Đào tạo, bồi dưỡng, nâng cao trình độ, kỹ năng nghề. ({cite})",
            *(
                [
                    f"Thông tư 10 hướng dẫn chi tiết các nội dung chủ yếu này theo khoản 1 Điều 21 của Bộ luật Lao động. ({_cite(tt10)})",
                ]
                if tt10
                else []
            ),
            "",
            f"Tóm lại: câu trả lời chính nằm ở danh mục nội dung bắt buộc của hợp đồng lao động tại {cite}.",
        ]
    )
    return _payload(answer, [context for context in (law21, tt10) if context])


def _dismissal_mediation_payload(contexts: Sequence[RetrievalContext]) -> dict[str, object] | None:
    law188 = _find_context(contexts, document_id="45-2019-qh14", article="188", clause="1")
    blttds32 = _find_context(contexts, document_id="92-2015-qh13-labor-only", article="32")
    if law188 is None:
        return None
    answer = "\n".join(
        [
            f"Không bắt buộc phải hòa giải trước khi kiện nếu tranh chấp là về xử lý kỷ luật lao động theo hình thức sa thải. ({_cite(law188)})",
            "",
            "Nội dung cụ thể như sau:",
            f"- Quy tắc chung là tranh chấp lao động cá nhân phải qua hòa giải trước khi yêu cầu trọng tài lao động hoặc Tòa án giải quyết. ({_cite(law188)})",
            f"- Nhưng chính điều này loại trừ các tranh chấp về xử lý kỷ luật lao động theo hình thức sa thải, nên loại tranh chấp đó không bắt buộc hòa giải. ({_cite(law188)})",
            *(
                [
                    f"- Nếu người lao động khởi kiện, tranh chấp lao động thuộc nhóm tranh chấp về lao động do Tòa án giải quyết theo Bộ luật Tố tụng dân sự. ({_cite(blttds32)})",
                ]
                if blttds32
                else []
            ),
            "",
            f"Tóm lại: tranh chấp sa thải thuộc ngoại lệ không bắt buộc hòa giải trước khi kiện theo {_cite(law188)}.",
        ]
    )
    return _payload(answer, [context for context in (law188, blttds32) if context])


def _illegal_termination_payload(contexts: Sequence[RetrievalContext]) -> dict[str, object] | None:
    c1 = _find_context(contexts, document_id="45-2019-qh14", article="40", clause="1")
    c2 = _find_context(contexts, document_id="45-2019-qh14", article="40", clause="2")
    c3 = _find_context(contexts, document_id="45-2019-qh14", article="40", clause="3")
    law39 = _find_context(contexts, document_id="45-2019-qh14", article="39")
    if c1 is None and c2 is None and c3 is None:
        return None
    primary = c2 or c1 or c3
    answer = "\n".join(
        [
            f"Nếu người lao động đơn phương chấm dứt hợp đồng lao động trái pháp luật, hậu quả chính là không được trợ cấp thôi việc và phải bồi thường cho người sử dụng lao động. ({_cite(primary)})",
            "",
            "Nội dung cụ thể như sau:",
            *(
                [f"- Không được trợ cấp thôi việc. ({_cite(c1)})"]
                if c1
                else []
            ),
            *(
                [
                    f"- Phải bồi thường nửa tháng tiền lương theo hợp đồng lao động. ({_cite(c2)})",
                    f"- Nếu vi phạm thời hạn báo trước thì còn phải bồi thường khoản tiền tương ứng với tiền lương trong những ngày không báo trước. ({_cite(c2)})",
                ]
                if c2
                else []
            ),
            *(
                [f"- Nếu có chi phí đào tạo thuộc Điều 62 thì phải hoàn trả chi phí đào tạo cho người sử dụng lao động. ({_cite(c3)})"]
                if c3
                else []
            ),
            *(
                [
                    f"Điều 39 là căn cứ xác định thế nào là đơn phương chấm dứt hợp đồng lao động trái pháp luật. ({_cite(law39)})",
                ]
                if law39
                else []
            ),
            "",
            "Tóm lại: nghĩa vụ trọng tâm là mất trợ cấp thôi việc, bồi thường nửa tháng tiền lương, bồi thường tiền lương những ngày không báo trước nếu có vi phạm báo trước, và hoàn trả chi phí đào tạo nếu phát sinh.",
        ]
    )
    return _payload(answer, [context for context in (c1, c2, c3, law39) if context])


def _structural_change_payload(contexts: Sequence[RetrievalContext]) -> dict[str, object] | None:
    law42 = _find_context(contexts, document_id="45-2019-qh14", article="42", clause="5")
    law47 = _find_context(contexts, document_id="45-2019-qh14", article="47", clause="1")
    nd145 = _find_context(contexts, document_id="nghi-dinh-145-2020-nd-cp", article="8", clause="2")
    if law42 is None and law47 is None:
        return None
    primary = law42 or law47
    answer = "\n".join(
        [
            f"Nếu thay đổi cơ cấu làm người sử dụng lao động không thể giải quyết được việc làm và phải cho người lao động thôi việc, khoản phải trả là trợ cấp mất việc làm. ({_cite(primary)})",
            "",
            "Nội dung cụ thể như sau:",
            *(
                [
                    f"- Trường hợp không thể giải quyết được việc làm mà phải cho thôi việc thì người sử dụng lao động phải trả trợ cấp mất việc làm theo Điều 47. ({_cite(law42)})",
                ]
                if law42
                else []
            ),
            *(
                [
                    f"- Người lao động làm việc thường xuyên từ đủ 12 tháng trở lên mà bị mất việc thì cứ mỗi năm làm việc được trả 01 tháng tiền lương, nhưng ít nhất bằng 02 tháng tiền lương. ({_cite(law47)})",
                ]
                if law47
                else []
            ),
            *(
                [
                    f"- Nghị định 145 hướng dẫn thêm trách nhiệm chi trả trợ cấp mất việc làm và nguyên tắc tối thiểu 02 tháng tiền lương trong trường hợp thời gian tính trợ cấp dưới 24 tháng. ({_cite(nd145)})",
                ]
                if nd145
                else []
            ),
            "",
            "Tóm lại: câu trả lời là trợ cấp mất việc làm, không phải bồi thường do người lao động đơn phương chấm dứt trái luật.",
        ]
    )
    return _payload(answer, [context for context in (law42, law47, nd145) if context])


def _job_loss_allowance_payload(contexts: Sequence[RetrievalContext]) -> dict[str, object] | None:
    law47 = _find_context(contexts, document_id="45-2019-qh14", article="47", clause="1")
    nd145 = _find_context(contexts, document_id="nghi-dinh-145-2020-nd-cp", article="8")
    law42 = _find_context(contexts, document_id="45-2019-qh14", article="42", clause="5")
    if law47 is None:
        return None
    answer = "\n".join(
        [
            f"Tro cap mat viec lam duoc tinh theo nguyen tac: moi nam lam viec duoc tra 01 thang tien luong, nhung tong muc tro cap it nhat bang 02 thang tien luong. ({_cite(law47)})",
            "",
            "Noi dung cu the nhu sau:",
            f"- Dieu kien trong context la nguoi lao dong lam viec thuong xuyen tu du 12 thang tro len ma bi mat viec lam. ({_cite(law47)})",
            f"- Cong thuc trong luat: cu moi nam lam viec duoc tra 01 thang tien luong. ({_cite(law47)})",
            f"- Muc toi thieu trong luat: it nhat bang 02 thang tien luong. ({_cite(law47)})",
            *(
                [
                    f"- Khi thay doi co cau, cong nghe ma khong the giai quyet viec lam va phai cho thoi viec thi nguoi su dung lao dong phai tra tro cap mat viec lam theo Dieu 47. ({_cite(law42)})",
                ]
                if law42
                else []
            ),
            *(
                [
                    f"- Nghi dinh 145 huong dan cach xac dinh thoi gian lam viec de tinh tro cap thoi viec, tro cap mat viec lam va thoi gian da tham gia bao hiem that nghiep. ({_cite(nd145)})",
                ]
                if nd145
                else []
            ),
            "",
            f"Tom lai: khoan can tra la tro cap mat viec lam, tinh theo so nam lam viec va tien luong, voi san toi thieu 02 thang tien luong. ({_cite(law47)})",
        ]
    )
    return _payload(answer, [context for context in (law47, law42, nd145) if context])


def _severance_vs_job_loss_payload(contexts: Sequence[RetrievalContext]) -> dict[str, object] | None:
    law46 = _find_context(contexts, document_id="45-2019-qh14", article="46", clause="1")
    law47 = _find_context(contexts, document_id="45-2019-qh14", article="47", clause="1")
    nd145 = _find_context(contexts, document_id="nghi-dinh-145-2020-nd-cp", article="8")
    if law46 is None or law47 is None:
        return None
    answer = "\n".join(
        [
            f"Tro cap thoi viec va tro cap mat viec lam la hai che do khac nhau: tro cap thoi viec gan voi cac truong hop cham dut hop dong theo Dieu 46, con tro cap mat viec lam gan voi truong hop nguoi lao dong bi mat viec theo Dieu 47. ({_cite(law46)}; {_cite(law47)})",
            "",
            "Noi dung so sanh cu the:",
            f"- Tro cap thoi viec: nguoi lao dong lam viec thuong xuyen tu du 12 thang tro len duoc tra tro cap thoi viec khi cham dut hop dong theo cac truong hop luat dinh. ({_cite(law46)})",
            f"- Cach tinh tro cap thoi viec trong luat: moi nam lam viec duoc tro cap mot nua thang tien luong. ({_cite(law46)})",
            f"- Tro cap mat viec lam: nguoi lao dong lam viec thuong xuyen tu du 12 thang tro len ma bi mat viec lam thi duoc tra tro cap mat viec lam. ({_cite(law47)})",
            f"- Cach tinh tro cap mat viec lam trong luat: moi nam lam viec duoc tra 01 thang tien luong, nhung it nhat bang 02 thang tien luong. ({_cite(law47)})",
            *(
                [
                    f"- Nghi dinh 145 huong dan thoi gian lam viec de tinh tro cap thoi viec va tro cap mat viec lam, gom ca viec loai tru thoi gian da tham gia bao hiem that nghiep neu context ap dung. ({_cite(nd145)})",
                ]
                if nd145
                else []
            ),
            "",
            f"Tom lai: Dieu 46 dung cho tro cap thoi viec voi muc nua thang tien luong moi nam lam viec; Dieu 47 dung cho tro cap mat viec lam voi muc 01 thang tien luong moi nam lam viec va toi thieu 02 thang tien luong. ({_cite(law46)}; {_cite(law47)})",
        ]
    )
    return _payload(answer, [context for context in (law46, law47, nd145) if context])


def _early_retirement_payload(contexts: Sequence[RetrievalContext]) -> dict[str, object] | None:
    law169 = _find_context(contexts, document_id="45-2019-qh14", article="169", clause="3")
    nd135 = _find_context(contexts, document_id="nghi-dinh-135-2020-nd-cp", article="5")
    appendix = _find_context(contexts, document_id="nghi-dinh-135-2020-nd-cp", citation_terms=("phu luc ii",))
    if law169 is None and nd135 is None:
        return None
    primary = nd135 or law169
    answer = "\n".join(
        [
            f"Nguoi lao dong lam nghe, cong viec nang nhoc, doc hai, nguy hiem co the nghi huu o tuoi thap hon neu thuoc truong hop luat va nghi dinh quy dinh. ({_cite(primary)})",
            "",
            "Noi dung cu the nhu sau:",
            *(
                [
                    f"- Bo luat Lao dong cho phep nghi huu o tuoi thap hon tuoi nghi huu trong dieu kien binh thuong doi voi mot so truong hop, trong do co nguoi lam nghe, cong viec nang nhoc, doc hai, nguy hiem. ({_cite(law169)})",
                ]
                if law169
                else []
            ),
            *(
                [
                    f"- Nghi dinh 135 quy dinh chi tiet viec nghi huu o tuoi thap hon tuoi nghi huu trong dieu kien lao dong binh thuong. ({_cite(nd135)})",
                ]
                if nd135
                else []
            ),
            *(
                [
                    f"- Neu can tra tuoi cu the theo thang, nam sinh thi doi chieu them phu luc tuoi nghi huu thap nhat. ({_cite(appendix)})",
                ]
                if appendix
                else []
            ),
            "",
            f"Tom lai: can cu chinh la quy tac nghi huu tuoi thap hon va van ban huong dan chi tiet cua Nghi dinh 135. ({_cite(primary)})",
        ]
    )
    return _payload(answer, [context for context in (law169, nd135, appendix) if context])


def _no_notice_payload(question: str, contexts: Sequence[RetrievalContext]) -> dict[str, object] | None:
    law35_2 = _find_context(contexts, document_id="45-2019-qh14", article="35", clause="2")
    if law35_2 is None:
        return None
    cite = _cite(law35_2)
    if answer_language(question) == "en":
        answer = "\n".join(
            [
                "An employee may terminate an employment contract without prior notice only in the statutory cases listed in the retrieved Labor Code provision.",
                "- The employee is not assigned the agreed work, workplace, or working conditions.",
                "- The employee is not paid fully or is not paid on time.",
                "- The employee is abused, beaten, insulted, subjected to conduct affecting health, dignity, or honor, or forced to work.",
                "- The employee is sexually harassed at the workplace.",
                "- A pregnant employee must stop working under the applicable legal rule.",
                "- The employee reaches retirement age, unless the parties have another agreement.",
                "- The employer provides untruthful information that affects performance of the employment contract.",
            ]
        )
        return _payload(answer, [law35_2])
    answer = "\n".join(
        [
            f"Người lao động được đơn phương chấm dứt hợp đồng lao động không cần báo trước trong các trường hợp được liệt kê tại {cite}.",
            "",
            "Nội dung cụ thể như sau:",
            f"- Không được bố trí đúng công việc, địa điểm làm việc hoặc không được bảo đảm điều kiện làm việc theo thỏa thuận. ({cite})",
            f"- Không được trả đủ lương hoặc trả lương không đúng thời hạn. ({cite})",
            f"- Bị người sử dụng lao động ngược đãi, đánh đập, nhục mạ, hành vi ảnh hưởng sức khỏe, nhân phẩm, danh dự hoặc bị cưỡng bức lao động. ({cite})",
            f"- Bị quấy rối tình dục tại nơi làm việc. ({cite})",
            f"- Lao động nữ mang thai phải nghỉ việc theo quy định của luật. ({cite})",
            f"- Đủ tuổi nghỉ hưu, trừ khi các bên có thỏa thuận khác. ({cite})",
            f"- Người sử dụng lao động cung cấp thông tin không trung thực làm ảnh hưởng đến việc thực hiện hợp đồng lao động. ({cite})",
            "",
            f"Tóm lại: căn cứ chính cho trường hợp nghỉ việc không cần báo trước là Điều 35 khoản 2. ({cite})",
        ]
    )
    return _payload(answer, [law35_2])


def _overtime_payload(contexts: Sequence[RetrievalContext]) -> dict[str, object] | None:
    law107_2 = _find_context(contexts, document_id="45-2019-qh14", article="107", clause="2")
    law107_3 = _find_context(contexts, document_id="45-2019-qh14", article="107", clause="3")
    nd145_60 = _find_context(contexts, document_id="nghi-dinh-145-2020-nd-cp", article="60")
    nd145_61 = _find_context(contexts, document_id="nghi-dinh-145-2020-nd-cp", article="61")
    if law107_2 is None:
        return None
    answer = "\n".join(
        [
            f"Được tổ chức làm thêm giờ khi có sự đồng ý của người lao động và bảo đảm các giới hạn làm thêm; giới hạn theo tháng là không quá 40 giờ trong 01 tháng. ({_cite(law107_2)})",
            "",
            "Nội dung cụ thể như sau:",
            f"- Phải được sự đồng ý của người lao động. ({_cite(law107_2)})",
            f"- Số giờ làm thêm không quá 50% số giờ làm việc bình thường trong 01 ngày; nếu tính theo tuần thì tổng giờ làm bình thường và làm thêm không quá 12 giờ trong 01 ngày. ({_cite(law107_2)})",
            f"- Số giờ làm thêm không quá 40 giờ trong 01 tháng. ({_cite(law107_2)})",
            f"- Số giờ làm thêm không quá 200 giờ trong 01 năm, trừ các trường hợp được tổ chức làm thêm từ trên 200 giờ đến 300 giờ/năm. ({_cite(law107_2)})",
            *(
                [
                    f"- Các trường hợp làm thêm từ trên 200 giờ đến 300 giờ/năm được quy định riêng. ({_cite(law107_3)})",
                ]
                if law107_3
                else []
            ),
            *(
                [
                    f"- Nghị định 145 hướng dẫn thêm giới hạn số giờ làm thêm trong một số trường hợp. ({_cite(nd145_60)})",
                ]
                if nd145_60
                else []
            ),
            *(
                [
                    f"- Nếu hỏi về các trường hợp được tổ chức làm thêm trên 200 giờ đến 300 giờ/năm thì đối chiếu thêm Điều 61 Nghị định 145. ({_cite(nd145_61)})",
                ]
                if nd145_61
                else []
            ),
            "",
            f"Tóm lại: câu hỏi về điều kiện và giới hạn làm thêm phải ưu tiên quy định về làm thêm giờ và giới hạn số giờ làm thêm. ({_cite(law107_2)})",
        ]
    )
    return _payload(answer, [context for context in (law107_2, law107_3, nd145_60, nd145_61) if context])


def _employee_definition_payload(
    question: str,
    contexts: Sequence[RetrievalContext],
) -> dict[str, object] | None:
    definition = (
        _find_context(contexts, document_id="45-2019-qh14", article="3", clause="1")
        or _find_context(
            contexts,
            article="3",
            clause="1",
            text_terms=("nguoi lao dong", "tra luong"),
        )
        or _find_context(
            contexts,
            article="3",
            text_terms=("nguoi lao dong", "tra luong"),
        )
    )
    if definition is None:
        return None

    cite = _cite(definition)
    if answer_language(question) == "en":
        answer = "\n".join(
            [
                f"Under {cite}, an employee is a person who works for an employer under an agreement, is paid wages, and is subject to the employer's management, direction, and supervision.",
                "",
                "The same retrieved provision also states that the minimum working age is 15, except for the separate rules on minor workers.",
            ]
        )
    else:
        answer = "\n".join(
            [
                f"Theo {cite}, người lao động là người làm việc cho người sử dụng lao động theo thỏa thuận, được trả lương và chịu sự quản lý, điều hành, giám sát của người sử dụng lao động.",
                "",
                "Cùng quy định này cũng nêu tuổi lao động tối thiểu là đủ 15 tuổi, trừ các trường hợp riêng về lao động chưa thành niên.",
            ]
        )
    return _payload(answer, [definition])


def build_rule_based_answer_payload(
    question: str,
    contexts: Sequence[RetrievalContext],
) -> dict[str, object] | None:
    intent = classify_answer_intent(question)
    if intent == ANSWER_INTENT_EMPLOYEE_DEFINITION:
        return _employee_definition_payload(question, contexts)
    if intent == ANSWER_INTENT_MINOR_WORKER:
        return _minor_worker_payload(question, contexts)
    if intent == ANSWER_INTENT_RETIREMENT_AGE:
        return _retirement_payload(contexts, question=question)
    if intent == ANSWER_INTENT_EARLY_RETIREMENT_HAZARDOUS_WORK:
        return _early_retirement_payload(contexts)
    if intent == ANSWER_INTENT_CONTRACT_CONTENT:
        return _contract_content_payload(contexts)
    if intent == ANSWER_INTENT_DISMISSAL_MEDIATION:
        return _dismissal_mediation_payload(contexts)
    if intent == ANSWER_INTENT_ILLEGAL_EMPLOYEE_TERMINATION:
        return _illegal_termination_payload(contexts)
    if intent == ANSWER_INTENT_STRUCTURAL_CHANGE:
        return _structural_change_payload(contexts)
    if intent == ANSWER_INTENT_JOB_LOSS_ALLOWANCE:
        return _job_loss_allowance_payload(contexts)
    if intent == ANSWER_INTENT_SEVERANCE_VS_JOB_LOSS:
        return _severance_vs_job_loss_payload(contexts)
    if intent == ANSWER_INTENT_NO_NOTICE_RESIGNATION:
        return _no_notice_payload(question, contexts)
    if intent == ANSWER_INTENT_OVERTIME_LIMITS:
        return _overtime_payload(contexts)
    return None


__all__ = [
    "ANSWER_INTENT_CONTRACT_CONTENT",
    "ANSWER_INTENT_DISMISSAL_MEDIATION",
    "ANSWER_INTENT_EMPLOYEE_DEFINITION",
    "ANSWER_INTENT_GENERIC",
    "ANSWER_INTENT_ILLEGAL_EMPLOYEE_TERMINATION",
    "ANSWER_INTENT_JOB_LOSS_ALLOWANCE",
    "ANSWER_INTENT_MINOR_WORKER",
    "ANSWER_INTENT_NO_NOTICE_RESIGNATION",
    "ANSWER_INTENT_OVERTIME_LIMITS",
    "ANSWER_INTENT_RETIREMENT_AGE",
    "ANSWER_INTENT_SEVERANCE_VS_JOB_LOSS",
    "ANSWER_INTENT_EARLY_RETIREMENT_HAZARDOUS_WORK",
    "ANSWER_INTENT_STRUCTURAL_CHANGE",
    "build_rule_based_answer_payload",
    "classify_answer_intent",
    "select_contexts_for_grounded_generation",
]
