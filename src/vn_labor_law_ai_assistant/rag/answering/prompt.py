from __future__ import annotations

from typing import Sequence

from ...corpus_pipeline import normalize_for_matching
from ...retriever import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_CONTEXT_TOKENS,
    RetrievalContext,
    dedupe_preserve_order,
    select_contexts_for_prompt,
)

SYSTEM_PROMPT = """You are a legal research assistant for Vietnamese labor law.

Mandatory grounding rules:
1. Answer only from the provided CONTEXT.
1a. Do not use outside legal knowledge, even if you believe it is correct.
1b. Before making a legal conclusion, verify that the conclusion is directly supported by wording in CONTEXT.
1c. If CONTEXT conflicts with background knowledge, follow CONTEXT.
1d. If sources conflict, apply the legal hierarchy: Code/Law > Decree > Circular.
1e. If a lower-level document explains a higher-level document, cite the higher-level source first, then the guidance source.
2. Do not invent article numbers, clauses, points, document names, or statutory conditions.
3. Set insufficient_context = true only when no retrieved context directly supports a reliable answer, or when a required statutory condition is missing.
3a. If CONTEXT supports only part of the answer, answer that part and clearly state the limitation.
3b. Do not use system-error language or boilerplate refusal text.
4. legal_basis must contain only exact citation strings from ALLOWED_CITATIONS.
4a. Copy citations exactly. Do not shorten or rewrite them.
4b. Prefer the most specific available citation, such as a point or clause citation.
5. legal_basis and evidence_quotes are for validation/trace only. They must not be copied as a long evidence section in the answer field.
6. Use the same language as the user question:
   - English question -> English answer.
   - Vietnamese question -> Vietnamese answer.
   Keep official Vietnamese document identifiers as written.
7. If insufficient_context = true, legal_basis and evidence_quotes must be empty arrays.
8. The answer field is the user-facing legal answer body:
   - Answer the user's question first.
   - When stating a legal rule, introduce it with the exact legal reference in prose, e.g. "Theo khoản 1 Điều 21 Bộ luật Lao động 2019..." or "Căn cứ điểm a khoản 2 Điều ...".
   - Then explain, synthesize, or apply the rule in clear language. Do not merely say there is a legal basis.
   - For list questions ("gồm những gì", "các trường hợp nào", "điều kiện gì", "nội dung gì"), use a structured numbered/bulleted answer. Cite the governing article/clause/point at the start of each group or item when different rules apply.
   - Descriptive headings are allowed when they help a list answer.
   - Do not add a final "Căn cứ pháp lý" / "Legal basis" section inside the answer field; the application formats that separately from legal_basis.
   - Do not include headings such as "Answer" or "Trả lời".
   - Do not paste long raw legal quotes into the answer body; use evidence_quotes for exact supporting passages.
9. Every important legal conclusion must have supporting evidence_quotes copied from CONTEXT.
9a. If you cannot quote supporting wording from CONTEXT, do not make a strong conclusion.

Return valid JSON only:
{
  "answer": "direct answer in the same language as the question",
  "legal_basis": ["citation_text 1", "citation_text 2"],
  "evidence_quotes": [
    {"citation": "citation_text 1", "quote": "short exact passage from CONTEXT"}
  ],
  "insufficient_context": false,
  "notes": "short internal note if needed, otherwise empty string"
}
"""

DOCUMENT_TYPE_ORDER = {
    "bo_luat": 1,
    "luat": 1,
    "nghi_dinh": 2,
    "thong_tu": 3,
}


def _context_normative_rank(context: RetrievalContext) -> int:
    try:
        rank = int(context.payload.get("normative_rank") or 0)
    except (TypeError, ValueError):
        rank = 0
    if rank > 0:
        return rank
    document_type = str(context.payload.get("document_type") or "").strip()
    return DOCUMENT_TYPE_ORDER.get(document_type, 99)


def _context_article_number(context: RetrievalContext) -> int:
    try:
        return int(str(context.payload.get("article_number") or "999999"))
    except ValueError:
        return 999999


def _context_clause_number(context: RetrievalContext) -> int:
    try:
        return int(str(context.payload.get("clause_ref") or "999999"))
    except ValueError:
        return 999999


def _is_appendix_context(context: RetrievalContext) -> bool:
    level = normalize_for_matching(str(context.payload.get("level") or ""))
    chunk_type = normalize_for_matching(str(context.payload.get("chunk_type") or ""))
    appendix_id = str(context.payload.get("appendix_id") or "").strip()
    return bool(appendix_id) or "appendix" in {level, chunk_type} or "phu luc" in level or "phu luc" in chunk_type


def order_contexts_for_answer(contexts: Sequence[RetrievalContext]) -> tuple[RetrievalContext, ...]:
    indexed_contexts = list(enumerate(contexts))
    indexed_contexts.sort(
        key=lambda item: (
            _context_normative_rank(item[1]),
            1 if _is_appendix_context(item[1]) and not item[1].payload.get("retrieval_force_include") else 0,
            item[0],
            _context_article_number(item[1]),
            _context_clause_number(item[1]),
            -float(item[1].score),
        )
    )
    return tuple(context for _, context in indexed_contexts)


def build_answer_context_block(context: RetrievalContext, index: int) -> str:
    payload = context.payload
    point_refs = payload.get("point_refs") or ()
    graph_path = payload.get("graph_path") or payload.get("graph_paths") or ()
    metadata_lines = [
        f"[CONTEXT {index}]",
        f"Legal basis: {context.citation_text}",
        f"document_id: {payload.get('document_id') or ''}",
        f"document_type: {payload.get('document_type') or ''}",
        f"normative_rank: {payload.get('normative_rank') or ''}",
        f"article_number: {payload.get('article_number') or ''}",
        f"clause_ref: {payload.get('clause_ref') or ''}",
        f"point_refs: {', '.join(str(value) for value in point_refs) if isinstance(point_refs, (list, tuple)) else point_refs}",
        f"graph_path: {graph_path}",
    ]

    unique_matched_citations = dedupe_preserve_order(context.matched_citations)
    if unique_matched_citations and unique_matched_citations != (context.citation_text,):
        metadata_lines.append("Matched citations:")
        metadata_lines.extend(f"- {citation}" for citation in unique_matched_citations)

    metadata_lines.extend(
        [
            "Text:",
            str(payload.get("retrieval_text") or context.text).strip(),
        ]
    )
    return "\n".join(metadata_lines).strip()


def format_answer_context_for_prompt(
    contexts: Sequence[RetrievalContext],
    *,
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_tokens: int | None = DEFAULT_MAX_CONTEXT_TOKENS,
) -> str:
    blocks = [
        build_answer_context_block(context, index)
        for index, context in enumerate(contexts, start=1)
    ]
    text = "\n\n".join(blocks).strip()
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip()
    return text


ANSWER_FEW_SHOT_PROMPT = """FORMAT EXAMPLES:

Example 1 - English supported answer:
- Question: Can an employee terminate an employment contract without prior notice?
- Good JSON:
{
  "answer": "An employee may terminate the employment contract without prior notice only in the statutory situations supported by the retrieved context, such as not being assigned the agreed work or workplace, not being paid fully or on time, being abused or forced to work, being sexually harassed at work, having to stop work during pregnancy under the law, reaching retirement age unless otherwise agreed, or receiving untruthful employer information that affects contract performance.",
  "legal_basis": ["Labor Code 2019, Article 35, Clause 2"],
  "evidence_quotes": [
    {
      "citation": "Labor Code 2019, Article 35, Clause 2",
      "quote": "The employee has the right to unilaterally terminate the employment contract without prior notice in the following cases"
    }
  ],
  "insufficient_context": false,
  "notes": ""
}

Example 2 - Vietnamese supported answer:
- Question: Người lao động chưa đủ 15 tuổi làm việc cần điều kiện gì?
- Good JSON:
{
  "answer": "Theo khoản 1 Điều 145 Bộ luật Lao động 2019, người chưa đủ 15 tuổi chỉ được làm việc khi người sử dụng lao động đáp ứng các điều kiện riêng:\n\n1. Giao kết hợp đồng lao động bằng văn bản với người chưa đủ 15 tuổi và người đại diện theo pháp luật của người đó.\n2. Bố trí giờ làm việc không ảnh hưởng đến thời gian học tập.\n3. Có giấy khám sức khỏe phù hợp với công việc và tổ chức kiểm tra sức khỏe định kỳ.\n4. Bảo đảm điều kiện làm việc, an toàn, vệ sinh lao động phù hợp với lứa tuổi.",
  "legal_basis": ["Bộ luật Lao động 2019, Điều 145, khoản 1"],
  "evidence_quotes": [
    {
      "citation": "Bộ luật Lao động 2019, Điều 145, khoản 1",
      "quote": "người sử dụng lao động phải giao kết hợp đồng lao động bằng văn bản với người chưa đủ 15 tuổi và người đại diện theo pháp luật"
    }
  ],
  "insufficient_context": false,
  "notes": ""
}

Example 3 - Vietnamese list answer:
- Question: Hợp đồng lao động phải có những nội dung gì?
- Good JSON:
{
  "answer": "Nội dung bắt buộc của hợp đồng lao động\n\nTheo khoản 1 Điều 21 Bộ luật Lao động 2019, hợp đồng lao động phải có các nội dung chủ yếu sau:\n\n1. Thông tin về người sử dụng lao động: tên, địa chỉ; họ tên và chức danh của người giao kết hợp đồng bên phía người sử dụng lao động.\n2. Thông tin về người lao động: họ tên, ngày tháng năm sinh, giới tính, nơi cư trú và giấy tờ pháp lý.\n3. Công việc và địa điểm làm việc.\n4. Thời hạn của hợp đồng lao động.\n5. Mức lương, hình thức trả lương, thời hạn trả lương, phụ cấp lương và các khoản bổ sung khác.\n6. Chế độ nâng bậc, nâng lương; thời giờ làm việc, thời giờ nghỉ ngơi; trang bị bảo hộ lao động.\n7. Bảo hiểm xã hội, bảo hiểm y tế, bảo hiểm thất nghiệp; đào tạo, bồi dưỡng, nâng cao trình độ, kỹ năng nghề.\n\nNếu có văn bản hướng dẫn được truy xuất, hãy giải thích phần hướng dẫn sau khi đã nêu quy định chính của Bộ luật Lao động.",
  "legal_basis": ["Bộ luật Lao động 2019, Điều 21, khoản 1"],
  "evidence_quotes": [
    {
      "citation": "Bộ luật Lao động 2019, Điều 21, khoản 1",
      "quote": "Hợp đồng lao động phải có những nội dung chủ yếu sau đây"
    }
  ],
  "insufficient_context": false,
  "notes": ""
}

Example 4 - insufficient context:
- Question: Can I rely on this answer as final legal advice?
- Good JSON:
{
  "answer": "I could not find enough legal context in the indexed sources to answer that reliably. The retrieved context does not address whether the assistant's output may be treated as final legal advice.",
  "legal_basis": [],
  "evidence_quotes": [],
  "insufficient_context": true,
  "notes": "The question asks about reliance on advice, not a retrieved labor-law provision."
}
"""


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


def build_messages(
    question: str,
    contexts: Sequence[RetrievalContext],
    *,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_context_tokens: int | None = DEFAULT_MAX_CONTEXT_TOKENS,
) -> list[dict[str, str]]:
    ordered_contexts = order_contexts_for_answer(contexts)
    selected_contexts = select_contexts_for_prompt(
        ordered_contexts,
        max_chars=max_context_chars,
        max_tokens=max_context_tokens,
    )
    allowed_citations = build_allowed_citations(selected_contexts)
    context_text = format_answer_context_for_prompt(
        selected_contexts,
        max_chars=max_context_chars,
        max_tokens=max_context_tokens,
    )
    user_prompt = "\n\n".join(
        [
            ANSWER_FEW_SHOT_PROMPT,
            f"Question:\n{question.strip()}",
            (
                "LEGAL HIERARCHY RULE: If sources conflict, Code/Law prevails over Decree; "
                "Decree prevails over Circular. If using guidance from a lower-level document, "
                "cite the higher-level source first."
            ),
            "ALLOWED_CITATIONS:",
            "\n".join(f"- {citation}" for citation in allowed_citations),
            f"CONTEXT:\n{context_text.strip()}",
        ]
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


__all__ = [
    "ANSWER_FEW_SHOT_PROMPT",
    "SYSTEM_PROMPT",
    "build_allowed_citations",
    "build_answer_context_block",
    "build_messages",
    "format_answer_context_for_prompt",
    "order_contexts_for_answer",
]
