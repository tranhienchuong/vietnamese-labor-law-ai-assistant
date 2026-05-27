from __future__ import annotations

import unittest

from vn_labor_law_ai_assistant.answering import (
    ANSWER_JSON_SCHEMA,
    SYSTEM_PROMPT,
    EvidenceQuote,
    ParsedAnswer,
    build_allowed_citations,
    generate_grounded_answer,
    build_messages,
    canonicalize_citation,
    citation_overlap_matches,
    extract_json_candidate,
    format_answer_for_user,
    order_contexts_for_answer,
    parse_answer_payload,
    sanitize_evidence_quotes,
    sanitize_legal_basis,
    is_low_information_quote,
    validate_answer_quality,
    validate_grounded_answer,
)
from vn_labor_law_ai_assistant.corpus_pipeline import normalize_for_matching
from vn_labor_law_ai_assistant.retriever import RetrievalContext


def make_context(citation_text: str) -> RetrievalContext:
    return RetrievalContext(
        chunk_id=citation_text,
        citation_text=citation_text,
        text="Noi dung phap ly",
        payload={"level": "clause"},
        score=1.0,
        matched_chunk_ids=(citation_text,),
        matched_citations=(citation_text,),
    )


class AnsweringTests(unittest.TestCase):
    def test_answering_compat_exports(self) -> None:
        self.assertTrue(callable(build_messages))
        self.assertTrue(callable(parse_answer_payload))
        self.assertTrue(callable(format_answer_for_user))
        self.assertEqual(EvidenceQuote.__name__, "EvidenceQuote")
        self.assertEqual(ParsedAnswer.__name__, "ParsedAnswer")
        self.assertEqual(ANSWER_JSON_SCHEMA["type"], "object")

    def test_prompt_unchanged_smoke(self) -> None:
        contexts = (make_context("Bo luat so 45/2019/QH 14, Dieu 36, khoan 1"),)

        messages = build_messages("Cau hoi mau?", contexts)

        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("ALLOWED_CITATIONS:", messages[1]["content"])
        self.assertIn("CONTEXT:", messages[1]["content"])
        self.assertIn("Bo luat so 45/2019/QH 14, Dieu 36, khoan 1", messages[1]["content"])

    def test_format_answer_for_user_wraps_short_answer_like_sample(self) -> None:
        parsed = parse_answer_payload(
            """
            {
              "answer": "Bạn có thể thỏa thuận với người sử dụng lao động để dùng phép năm trong thời gian báo trước.",
              "legal_basis": ["Bo luat so 45/2019/QH14, Dieu 113, khoan 3"],
              "evidence_quotes": [
                {
                  "citation": "Bo luat so 45/2019/QH14, Dieu 113, khoan 3",
                  "quote": "Trường hợp do thôi việc, bị mất việc làm mà chưa nghỉ hằng năm thì được thanh toán tiền lương cho những ngày chưa nghỉ"
                }
              ],
              "insufficient_context": false,
              "notes": "Nên trao đổi rõ ràng với người sử dụng lao động trước khi nghỉ sớm."
            }
            """,
            (
                RetrievalContext(
                    chunk_id="ctx-113",
                    citation_text="Bo luat so 45/2019/QH14, Dieu 113, khoan 3",
                    text=(
                        "Trường hợp do thôi việc, bị mất việc làm mà chưa nghỉ hằng năm "
                        "thì được thanh toán tiền lương cho những ngày chưa nghỉ"
                    ),
                    payload={},
                    score=1.0,
                    matched_chunk_ids=("ctx-113",),
                    matched_citations=("Bo luat so 45/2019/QH14, Dieu 113, khoan 3",),
                ),
            ),
        )

        formatted = format_answer_for_user(
            parsed,
            question="Tôi có thể cấn trừ phép năm để nghỉ sớm hơn không?",
        )

        self.assertNotIn("Câu hỏi:", formatted)
        self.assertIn("Câu trả lời:", formatted)
        self.assertIn("Căn cứ pháp lý:", formatted)
        self.assertIn("Nội dung cụ thể như sau:", formatted)
        self.assertIn("Tóm lại:", formatted)
        self.assertIn("Khuyến nghị:", formatted)

    def test_format_answer_for_user_does_not_duplicate_styled_answer_sections(self) -> None:
        parsed = parse_answer_payload(
            """
            {
              "answer": "Câu trả lời:\\nCăn cứ vào Bo luat so 45/2019/QH14, Dieu 35, khoan 1 thì người lao động phải báo trước.\\n\\nNội dung cụ thể như sau:\\nPhải báo trước theo loại hợp đồng.\\n\\nTóm lại:\\n- Cần báo trước.\\n\\nKhuyến nghị: Nên thông báo bằng văn bản.",
              "legal_basis": ["Bo luat so 45/2019/QH14, Dieu 35, khoan 1"],
              "evidence_quotes": [
                {
                  "citation": "Bo luat so 45/2019/QH14, Dieu 35, khoan 1",
                  "quote": "người lao động phải báo trước"
                }
              ],
              "insufficient_context": false,
              "notes": ""
            }
            """,
            (
                RetrievalContext(
                    chunk_id="ctx-35",
                    citation_text="Bo luat so 45/2019/QH14, Dieu 35, khoan 1",
                    text="người lao động phải báo trước theo quy định",
                    payload={},
                    score=1.0,
                    matched_chunk_ids=("ctx-35",),
                    matched_citations=("Bo luat so 45/2019/QH14, Dieu 35, khoan 1",),
                ),
            ),
        )

        formatted = format_answer_for_user(parsed, question="Tôi nghỉ việc phải báo trước không?")

        self.assertEqual(formatted.count("Câu trả lời:"), 1)
        self.assertNotIn("Câu hỏi:", formatted)
        self.assertNotIn("Căn cứ pháp lý:", formatted)
        self.assertEqual(formatted.count("Nội dung cụ thể như sau:"), 1)

    def test_format_answer_for_user_strips_model_question_section(self) -> None:
        parsed = parse_answer_payload(
            """
            {
              "answer": "Câu hỏi: Tôi nghỉ việc phải báo trước không?\\nCâu trả lời:\\nCăn cứ vào Bo luat so 45/2019/QH14, Dieu 35, khoan 1 thì người lao động phải báo trước.\\n\\nTóm lại:\\n- Cần báo trước.",
              "legal_basis": ["Bo luat so 45/2019/QH14, Dieu 35, khoan 1"],
              "evidence_quotes": [],
              "insufficient_context": false,
              "notes": ""
            }
            """,
            (make_context("Bo luat so 45/2019/QH14, Dieu 35, khoan 1"),),
        )

        formatted = format_answer_for_user(parsed, question="Tôi nghỉ việc phải báo trước không?")

        self.assertFalse(formatted.startswith("Câu hỏi:"))
        self.assertTrue(formatted.startswith("Câu trả lời:"))

    def test_parse_answer_payload_accepts_valid_evidence_quotes(self) -> None:
        contexts = (
            RetrievalContext(
                chunk_id="ctx-20",
                citation_text="Bo luat so 45/2019/QH14, Dieu 20, khoan 1",
                text=(
                    "Hop dong lao dong phai duoc giao ket theo mot trong cac loai sau day: "
                    "hop dong lao dong khong xac dinh thoi han va hop dong lao dong xac dinh thoi han."
                ),
                payload={},
                score=1.0,
                matched_chunk_ids=("ctx-20",),
                matched_citations=("Bo luat so 45/2019/QH14, Dieu 20, khoan 1",),
            ),
        )
        raw_content = """
        {
          "answer": "Co 2 loai hop dong lao dong.",
          "legal_basis": ["Bo luat so 45/2019/QH14, Dieu 20, khoan 1"],
          "evidence_quotes": [
            {
              "citation": "Bo luat so 45/2019/QH14, Dieu 20, khoan 1",
              "quote": "hop dong lao dong khong xac dinh thoi han va hop dong lao dong xac dinh thoi han"
            }
          ],
          "insufficient_context": false,
          "notes": ""
        }
        """

        parsed = parse_answer_payload(raw_content, contexts)

        self.assertEqual(
            parsed.evidence_quotes,
            (
                EvidenceQuote(
                    citation="Bo luat so 45/2019/QH14, Dieu 20, khoan 1",
                    quote="hop dong lao dong khong xac dinh thoi han va hop dong lao dong xac dinh thoi han",
                ),
            ),
        )

    def test_parse_answer_payload_applies_direct_numeric_context_guardrail(self) -> None:
        contexts = (
            RetrievalContext(
                chunk_id="ctx-20",
                citation_text="Bo luat so 45/2019/QH14, Dieu 20, khoan 1",
                text=(
                    "Hop dong lao dong phai duoc giao ket theo mot trong cac loai sau day: "
                    "hop dong lao dong khong xac dinh thoi han va hop dong lao dong xac dinh thoi han."
                ),
                payload={},
                score=1.0,
                matched_chunk_ids=("ctx-20",),
                matched_citations=("Bo luat so 45/2019/QH14, Dieu 20, khoan 1",),
            ),
        )
        raw_content = """
        {
          "answer": "Co 3 loai hop dong lao dong, gom ca hop dong mua vu.",
          "legal_basis": ["Bo luat so 45/2019/QH14, Dieu 20, khoan 1"],
          "evidence_quotes": [],
          "insufficient_context": false,
          "notes": ""
        }
        """

        parsed = parse_answer_payload(
            raw_content,
            contexts,
            question="Hien nay Bo luat Lao dong 2019 quy dinh co bao nhieu loai hop dong lao dong?",
        )

        self.assertIn("2 loai", parsed.answer.lower())
        self.assertNotIn("3 loai", parsed.answer.lower())
        self.assertFalse(parsed.insufficient_context)
        self.assertEqual(parsed.legal_basis, ("Bo luat so 45/2019/QH14, Dieu 20, khoan 1",))
        self.assertTrue(parsed.evidence_quotes)

    def test_parse_answer_payload_overrides_false_insufficient_notice_period(self) -> None:
        contexts = (
            RetrievalContext(
                chunk_id="ctx-35",
                citation_text="Bo luat so 45/2019/QH14, Dieu 35, khoan 1",
                text=(
                    "Dieu 35. Quyen don phuong cham dut hop dong lao dong cua nguoi lao dong\n"
                    "1. Nguoi lao dong co quyen don phuong cham dut hop dong lao dong "
                    "nhung phai bao truoc cho nguoi su dung lao dong nhu sau:\n"
                    "a) It nhat 45 ngay neu lam viec theo hop dong lao dong khong xac dinh thoi han;"
                ),
                payload={},
                score=1.0,
                matched_chunk_ids=("ctx-35-a",),
                matched_citations=(
                    "Bo luat so 45/2019/QH14, Dieu 35, khoan 1, diem a",
                ),
            ),
        )
        raw_content = """
        {
          "answer": "Chua du can cu de xac dinh.",
          "legal_basis": [],
          "evidence_quotes": [],
          "insufficient_context": true,
          "notes": ""
        }
        """

        parsed = parse_answer_payload(
            raw_content,
            contexts,
            question=(
                "Nguoi lao dong ky hop dong khong xac dinh thoi han muon nghi viec "
                "thi phai bao truoc bao lau?"
            ),
        )

        self.assertFalse(parsed.insufficient_context)
        self.assertIn("45 ngay", parsed.answer.lower())
        self.assertEqual(
            parsed.legal_basis,
            ("Bo luat so 45/2019/QH14, Dieu 35, khoan 1, diem a",),
        )
        self.assertTrue(parsed.evidence_quotes)
        self.assertIn("45 ngay", parsed.evidence_quotes[0].quote.lower())

    def test_sanitize_legal_basis_keeps_only_allowed_citations(self) -> None:
        contexts = (
            make_context("Bo luat so 45/2019/QH 14, Dieu 46, khoan 1"),
            make_context("Bo luat so 45/2019/QH 14, Dieu 46, khoan 2"),
        )

        legal_basis = sanitize_legal_basis(
            [
                "Bo luat so 45/2019/QH 14, Dieu 46, khoan 2",
                "Noi dung tro cap thoi viec",
            ],
            contexts,
        )

        self.assertEqual(legal_basis, ("Bo luat so 45/2019/QH 14, Dieu 46, khoan 2",))

    def test_evidence_quote_must_appear_in_context(self) -> None:
        contexts = (make_context("Bo luat so 45/2019/QH 14, Dieu 46, khoan 1"),)

        evidence_quotes = sanitize_evidence_quotes(
            [
                {
                    "citation": "Bo luat so 45/2019/QH 14, Dieu 46, khoan 1",
                    "quote": "Noi dung khong co trong context",
                }
            ],
            contexts,
        )

        self.assertEqual(evidence_quotes, ())

    def test_evidence_quote_can_match_retrieval_text_payload(self) -> None:
        contexts = (
            RetrievalContext(
                chunk_id="ctx-3",
                citation_text="Bo luat Lao dong 2019, Dieu 3",
                text="Prompt text assembled from the retrieved context.",
                payload={
                    "retrieval_text": (
                        "Trong Bo luat nay, nguoi lao dong la nguoi lam viec cho nguoi su dung lao dong "
                        "theo thoa thuan, duoc tra luong va chiu su quan ly."
                    )
                },
                score=1.0,
                matched_chunk_ids=("ctx-3",),
                matched_citations=("Bo luat Lao dong 2019, Dieu 3",),
            ),
        )

        evidence_quotes = sanitize_evidence_quotes(
            [
                {
                    "citation": "Bo luat Lao dong 2019, Dieu 3",
                    "quote": "nguoi lao dong la nguoi lam viec cho nguoi su dung lao dong theo thoa thuan",
                }
            ],
            contexts,
        )

        self.assertEqual(len(evidence_quotes), 1)

    def test_sanitize_legal_basis_respects_empty_response(self) -> None:
        contexts = (
            make_context("Bo luat so 45/2019/QH 14, Dieu 36, khoan 1"),
            make_context("Bo luat so 45/2019/QH 14, Dieu 36, khoan 2"),
        )

        self.assertEqual(sanitize_legal_basis([], contexts), ())
        self.assertEqual(sanitize_legal_basis(None, contexts), ())

    def test_build_allowed_citations_includes_matched_child_citations(self) -> None:
        contexts = (
            RetrievalContext(
                chunk_id="ctx-35",
                citation_text="Bo luat so 45/2019/QH 14, Dieu 35, khoan 1",
                text="Noi dung phap ly",
                payload={},
                score=1.0,
                matched_chunk_ids=("ctx-35-point-a",),
                matched_citations=("Bo luat so 45/2019/QH 14, Dieu 35, khoan 1, diem a",),
            ),
        )

        allowed_citations = build_allowed_citations(contexts)

        self.assertEqual(
            allowed_citations,
            (
                "Bo luat so 45/2019/QH 14, Dieu 35, khoan 1, diem a",
                "Bo luat so 45/2019/QH 14, Dieu 35, khoan 1",
            ),
        )

    def test_citation_overlap_matches_parent_and_child_citation(self) -> None:
        self.assertTrue(
            citation_overlap_matches(
                "Bo luat so 45/2019/QH 14, Dieu 35, khoan 1, diem a",
                "Bo luat so 45/2019/QH 14, Dieu 35, khoan 1",
            )
        )
        self.assertFalse(
            citation_overlap_matches(
                "Bo luat so 45/2019/QH 14, Dieu 35, khoan 1, diem a",
                "Bo luat so 45/2019/QH 14, Dieu 35, khoan 1, diem b",
            )
        )

    def test_citation_overlap_matches_when_model_omits_article_title(self) -> None:
        self.assertTrue(
            citation_overlap_matches(
                "Bo luat so 45/2019/QH14, Dieu 40 (Nghia vu cua nguoi lao dong khi don phuong cham dut hop dong lao dong trai phap luat), khoan 2",
                "Bo luat so 45/2019/QH14, Dieu 40, khoan 2",
            )
        )

    def test_canonicalize_citation_prefers_closest_allowed_match(self) -> None:
        allowed_citations = (
            "Bo luat so 45/2019/QH 14, Dieu 35, khoan 1, diem a",
            "Bo luat so 45/2019/QH 14, Dieu 35, khoan 1",
        )

        self.assertEqual(
            canonicalize_citation(
                "Bo luat so 45/2019/QH 14, Dieu 35, khoan 1",
                allowed_citations,
            ),
            "Bo luat so 45/2019/QH 14, Dieu 35, khoan 1",
        )
        self.assertEqual(
            canonicalize_citation(
                "Bo luat so 45/2019/QH 14, Dieu 35, khoan 1, diem a",
                allowed_citations,
            ),
            "Bo luat so 45/2019/QH 14, Dieu 35, khoan 1, diem a",
        )

    def test_canonicalize_citation_accepts_same_reference_without_article_title(self) -> None:
        allowed_citations = (
            "Bo luat so 45/2019/QH14, Dieu 40 (Nghia vu cua nguoi lao dong khi don phuong cham dut hop dong lao dong trai phap luat), khoan 2",
        )

        self.assertEqual(
            canonicalize_citation(
                "Bo luat so 45/2019/QH14, Dieu 40, khoan 2",
                allowed_citations,
            ),
            "Bo luat so 45/2019/QH14, Dieu 40 (Nghia vu cua nguoi lao dong khi don phuong cham dut hop dong lao dong trai phap luat), khoan 2",
        )

    def test_sanitize_legal_basis_accepts_specific_child_citation_from_matched_hits(self) -> None:
        contexts = (
            RetrievalContext(
                chunk_id="ctx-35",
                citation_text="Bo luat so 45/2019/QH 14, Dieu 35, khoan 1",
                text="Noi dung phap ly",
                payload={},
                score=1.0,
                matched_chunk_ids=("ctx-35-point-a",),
                matched_citations=("Bo luat so 45/2019/QH 14, Dieu 35, khoan 1, diem a",),
            ),
        )

        legal_basis = sanitize_legal_basis(
            ["Bo luat so 45/2019/QH 14, Dieu 35, khoan 1, diem a"],
            contexts,
        )

        self.assertEqual(
            legal_basis,
            ("Bo luat so 45/2019/QH 14, Dieu 35, khoan 1, diem a",),
        )

    def test_parse_answer_payload_preserves_answer_when_model_returns_invalid_basis(self) -> None:
        contexts = (
            make_context("Bo luat so 45/2019/QH 14, Dieu 46, khoan 1"),
            make_context("Bo luat so 45/2019/QH 14, Dieu 46, khoan 2"),
        )
        raw_content = """
        {
          "answer": "Tro cap thoi viec duoc tinh theo thoi gian lam viec va tien luong binh quan.",
          "legal_basis": [
            "moi nam lam viec duoc tro cap mot nua thang tien luong"
          ],
          "insufficient_context": false,
          "notes": ""
        }
        """

        parsed = parse_answer_payload(raw_content, contexts)

        self.assertEqual(parsed.legal_basis, ())
        self.assertFalse(parsed.insufficient_context)
        self.assertIn("tro cap thoi viec", parsed.answer.lower())
        self.assertIn("legal_basis", parsed.notes.lower())
        self.assertIn("allowed_citations", parsed.notes.lower())

    def test_parse_answer_payload_falls_back_naturally_when_answer_is_blank_and_basis_invalid(self) -> None:
        contexts = (make_context("Bo luat so 45/2019/QH 14, Dieu 46, khoan 1"),)
        raw_content = """
        {
          "answer": "",
          "legal_basis": ["Dieu 99 khoan 9"],
          "insufficient_context": false,
          "notes": ""
        }
        """

        parsed = parse_answer_payload(raw_content, contexts)

        self.assertTrue(parsed.insufficient_context)
        self.assertEqual(parsed.legal_basis, ())
        self.assertIn("chua du can cu", parsed.answer.lower())

    def test_parse_answer_payload_respects_insufficient_context(self) -> None:
        contexts = (
            make_context("Bo luat so 45/2019/QH 14, Dieu 36, khoan 1"),
            make_context("Bo luat so 45/2019/QH 14, Dieu 36, khoan 2"),
        )
        raw_content = """
        {
          "answer": "Chua du can cu de ket luan.",
          "legal_basis": [
            "Bo luat so 45/2019/QH 14, Dieu 36, khoan 1"
          ],
          "insufficient_context": true,
          "notes": ""
        }
        """

        parsed = parse_answer_payload(raw_content, contexts)

        self.assertTrue(parsed.insufficient_context)
        self.assertEqual(parsed.legal_basis, ())

    def test_parse_answer_payload_fills_blank_insufficient_answer(self) -> None:
        contexts = (make_context("Bo luat so 45/2019/QH 14, Dieu 36, khoan 1"),)
        raw_content = """
        {
          "answer": "",
          "legal_basis": [],
          "insufficient_context": true,
          "notes": ""
        }
        """

        parsed = parse_answer_payload(raw_content, contexts)

        self.assertTrue(parsed.insufficient_context)
        self.assertIn("chua du can cu", parsed.answer.lower())

    def test_parse_answer_payload_marks_invalid_json_as_insufficient(self) -> None:
        contexts = (make_context("Bo luat so 45/2019/QH 14, Dieu 36, khoan 1"),)

        parsed = parse_answer_payload("khong phai json hop le", contexts)

        self.assertTrue(parsed.insufficient_context)
        self.assertEqual(parsed.legal_basis, ())

    def test_parse_answer_payload_accepts_list_with_dict_item(self) -> None:
        contexts = (make_context("Bo luat so 45/2019/QH 14, Dieu 36, khoan 1"),)
        raw_content = """
        [
          {
            "answer": "Co.",
            "legal_basis": ["Bo luat so 45/2019/QH 14, Dieu 36, khoan 1"],
            "insufficient_context": false,
            "notes": ""
          }
        ]
        """

        parsed = parse_answer_payload(raw_content, contexts)

        self.assertFalse(parsed.insufficient_context)
        self.assertEqual(parsed.legal_basis, ("Bo luat so 45/2019/QH 14, Dieu 36, khoan 1",))

    def test_parse_answer_payload_accepts_child_citation_listed_in_matched_citations(self) -> None:
        contexts = (
            RetrievalContext(
                chunk_id="ctx-35",
                citation_text="Bo luat so 45/2019/QH 14, Dieu 35, khoan 1",
                text="Noi dung phap ly",
                payload={},
                score=1.0,
                matched_chunk_ids=("ctx-35-point-a",),
                matched_citations=("Bo luat so 45/2019/QH 14, Dieu 35, khoan 1, diem a",),
            ),
        )
        raw_content = """
        {
          "answer": "Khong. Nguoi lao dong khong can ly do chinh dang khi don phuong cham dut hop dong trong truong hop nay.",
          "legal_basis": ["Bo luat so 45/2019/QH 14, Dieu 35, khoan 1, diem a"],
          "insufficient_context": false,
          "notes": ""
        }
        """

        parsed = parse_answer_payload(raw_content, contexts)

        self.assertFalse(parsed.insufficient_context)
        self.assertEqual(
            parsed.legal_basis,
            ("Bo luat so 45/2019/QH 14, Dieu 35, khoan 1, diem a",),
        )

    def test_parse_answer_payload_rejects_valid_json_with_wrong_top_level_type(self) -> None:
        contexts = (make_context("Bo luat so 45/2019/QH 14, Dieu 36, khoan 1"),)

        parsed = parse_answer_payload('["chi la mot mang chuoi"]', contexts)

        self.assertTrue(parsed.insufficient_context)
        self.assertEqual(parsed.legal_basis, ())
        self.assertIn("khong the xac nhan", parsed.notes.lower())

    def test_extract_json_candidate_handles_markdown_fence(self) -> None:
        raw_content = """
        ```json
        {
          "answer": "Co",
          "legal_basis": ["Bo luat so 45/2019/QH 14, Dieu 36, khoan 1"],
          "insufficient_context": false,
          "notes": ""
        }
        ```
        """

        cleaned = extract_json_candidate(raw_content)

        self.assertTrue(cleaned.startswith("{"))
        self.assertTrue(cleaned.endswith("}"))
        self.assertIn('"answer": "Co"', cleaned)

    def test_extract_json_candidate_preserves_top_level_array(self) -> None:
        raw_content = """
        [
          {
            "answer": "Co",
            "legal_basis": ["Bo luat so 45/2019/QH 14, Dieu 36, khoan 1"],
            "insufficient_context": false,
            "notes": ""
          }
        ]
        """

        cleaned = extract_json_candidate(raw_content)

        self.assertTrue(cleaned.startswith("["))
        self.assertTrue(cleaned.endswith("]"))

    def test_parse_answer_payload_accepts_markdown_wrapped_json(self) -> None:
        contexts = (
            make_context("Bo luat so 45/2019/QH 14, Dieu 36, khoan 1"),
            make_context("Bo luat so 45/2019/QH 14, Dieu 36, khoan 2"),
        )
        raw_content = """
        ```json
        {
          "answer": "Co, nguoi lao dong co the duoc bao ve trong truong hop nay.",
          "legal_basis": [
            "Bo luat so 45/2019/QH 14, Dieu 36, khoan 1"
          ],
          "insufficient_context": false,
          "notes": ""
        }
        ```
        """

        parsed = parse_answer_payload(raw_content, contexts)

        self.assertFalse(parsed.insufficient_context)
        self.assertEqual(
            parsed.legal_basis,
            ("Bo luat so 45/2019/QH 14, Dieu 36, khoan 1",),
        )
        self.assertIn("nguoi lao dong", parsed.answer.lower())

    def test_parse_answer_payload_accepts_json_embedded_in_prose(self) -> None:
        contexts = (make_context("Bo luat so 45/2019/QH 14, Dieu 36, khoan 1"),)
        raw_content = """
        Day la ket qua da duoc dinh dang:
        {
          "answer": "Co.",
          "legal_basis": ["Bo luat so 45/2019/QH 14, Dieu 36, khoan 1"],
          "insufficient_context": false,
          "notes": ""
        }
        """

        parsed = parse_answer_payload(raw_content, contexts)

        self.assertFalse(parsed.insufficient_context)
        self.assertEqual(parsed.legal_basis, ("Bo luat so 45/2019/QH 14, Dieu 36, khoan 1",))

    def test_build_messages_drops_lower_ranked_context_blocks_to_fit_budget(self) -> None:
        contexts = (
            RetrievalContext(
                chunk_id="chunk-1",
                citation_text="Bo luat so 45/2019/QH 14, Dieu 36, khoan 1",
                text="A" * 80,
                payload={},
                score=1.0,
                matched_chunk_ids=("chunk-1",),
                matched_citations=("Bo luat so 45/2019/QH 14, Dieu 36, khoan 1",),
            ),
            RetrievalContext(
                chunk_id="chunk-2",
                citation_text="Bo luat so 45/2019/QH 14, Dieu 36, khoan 2",
                text="B" * 80,
                payload={},
                score=0.9,
                matched_chunk_ids=("chunk-2",),
                matched_citations=("Bo luat so 45/2019/QH 14, Dieu 36, khoan 2",),
            ),
        )

        messages = build_messages(
            "Cau hoi mau?",
            contexts,
            max_context_chars=180,
            max_context_tokens=200,
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Cau hoi:\nCau hoi mau?", messages[1]["content"])
        self.assertIn("CONTEXT:\n[NGU CANH 1]", messages[1]["content"])
        self.assertIn("Dieu 36, khoan 1", messages[1]["content"])
        self.assertNotIn("Dieu 36, khoan 2", messages[1]["content"])
        context_text = messages[1]["content"].split("CONTEXT:\n", 1)[1]
        self.assertNotIn("...", context_text)

    def test_order_contexts_for_answer_prefers_higher_normative_rank(self) -> None:
        circular = RetrievalContext(
            chunk_id="tt10-3",
            citation_text="Thong tu 10/2020/TT-BLDTBXH, Dieu 3",
            text="Huong dan noi dung hop dong",
            payload={"document_type": "thong_tu", "normative_rank": 3, "article_number": "3"},
            score=2.0,
            matched_chunk_ids=("tt10-3",),
            matched_citations=("Thong tu 10/2020/TT-BLDTBXH, Dieu 3",),
        )
        law = RetrievalContext(
            chunk_id="bll-21",
            citation_text="Bo luat Lao dong 2019, Dieu 21",
            text="Noi dung hop dong lao dong",
            payload={"document_type": "bo_luat", "normative_rank": 1, "article_number": "21"},
            score=1.0,
            matched_chunk_ids=("bll-21",),
            matched_citations=("Bo luat Lao dong 2019, Dieu 21",),
        )

        ordered = order_contexts_for_answer((circular, law))

        self.assertEqual([context.chunk_id for context in ordered], ["bll-21", "tt10-3"])

    def test_validate_grounded_answer_rejects_unretrieved_article_number(self) -> None:
        contexts = (make_context("Bo luat Lao dong 2019, Dieu 35, khoan 2"),)
        parsed = ParsedAnswer(
            answer="Can cu Bo luat Lao dong 2019, Dieu 99 thi duoc nghi.",
            legal_basis=("Bo luat Lao dong 2019, Dieu 35, khoan 2",),
            evidence_quotes=(),
            insufficient_context=False,
            notes="",
            raw_content="",
        )

        validation = validate_grounded_answer(parsed, contexts)

        self.assertFalse(validation.passed)
        self.assertEqual(validation.unsupported_article_numbers, ("99",))

    def test_validate_grounded_answer_allows_article_number_from_retrieved_text(self) -> None:
        contexts = (
            RetrievalContext(
                chunk_id="bll-47",
                citation_text="Bo luat Lao dong 2019, Dieu 47, khoan 1",
                text="Tro cap mat viec lam theo khoan 11 Dieu 34 cua Bo luat nay.",
                payload={"document_type": "bo_luat", "normative_rank": 1, "article_number": "47"},
                score=1.0,
                matched_chunk_ids=("bll-47",),
                matched_citations=("Bo luat Lao dong 2019, Dieu 47, khoan 1",),
            ),
        )
        parsed = ParsedAnswer(
            answer=(
                "Nguoi su dung lao dong tra tro cap mat viec lam theo khoan 11 Dieu 34; "
                "can cu Bo luat Lao dong 2019, Dieu 47, khoan 1."
            ),
            legal_basis=("Bo luat Lao dong 2019, Dieu 47, khoan 1",),
            evidence_quotes=(),
            insufficient_context=False,
            notes="",
            raw_content="",
        )

        validation = validate_grounded_answer(parsed, contexts)

        self.assertTrue(validation.passed)
        self.assertEqual(validation.unsupported_article_numbers, ())

    def test_generate_grounded_answer_extractive_uses_retrieved_citations(self) -> None:
        contexts = (
            RetrievalContext(
                chunk_id="bll-107",
                citation_text="Bo luat Lao dong 2019, Dieu 107, khoan 2",
                text="Bao dam so gio lam them khong qua 40 gio trong 01 thang.",
                payload={"document_type": "bo_luat", "normative_rank": 1, "article_number": "107", "clause_ref": "2"},
                score=1.0,
                matched_chunk_ids=("bll-107",),
                matched_citations=("Bo luat Lao dong 2019, Dieu 107, khoan 2",),
            ),
        )

        result = generate_grounded_answer(
            "Gioi han lam them theo thang la bao nhieu?",
            contexts,
            provider="extractive",
        )

        self.assertTrue(result.validation.passed)
        self.assertIn("Bo luat Lao dong 2019, Dieu 107, khoan 2", result.answer)
        self.assertEqual(result.parsed.legal_basis, ("Bo luat Lao dong 2019, Dieu 107, khoan 2",))

    def test_low_information_quote_detection_rejects_article_titles(self) -> None:
        self.assertTrue(is_low_information_quote("Dieu 143."))
        self.assertTrue(is_low_information_quote("Tuoi nghi huu"))
        self.assertFalse(
            is_low_information_quote(
                "Nguoi su dung lao dong phai giao ket hop dong lao dong bang van ban voi nguoi chua du 15 tuoi."
            )
        )

    def test_generate_contract_content_answer_lists_actual_items(self) -> None:
        contexts = (
            RetrievalContext(
                chunk_id="bll-21",
                citation_text="Bo luat Lao dong 2019, Dieu 21, khoan 1",
                text=(
                    "Dieu 21. Noi dung hop dong lao dong\n\n"
                    "1. Hop dong lao dong phai co nhung noi dung chu yeu sau day:\n\n"
                    "a) Ten, dia chi cua nguoi su dung lao dong va ho ten, chuc danh cua nguoi giao ket;\n\n"
                    "b) Ho ten, ngay thang nam sinh, gioi tinh, noi cu tru, giay to phap ly cua nguoi lao dong;\n\n"
                    "c) Cong viec va dia diem lam viec;\n\n"
                    "d) Thoi han cua hop dong lao dong;\n\n"
                    "dd) Muc luong, hinh thuc tra luong, thoi han tra luong, phu cap luong;\n\n"
                    "e) Che do nang bac, nang luong;\n\n"
                    "g) Thoi gio lam viec, thoi gio nghi ngoi;\n\n"
                    "h) Trang bi bao ho lao dong;\n\n"
                    "i) Bao hiem xa hoi, bao hiem y te va bao hiem that nghiep;\n\n"
                    "k) Dao tao, boi duong, nang cao trinh do, ky nang nghe."
                ),
                payload={"document_id": "45-2019-qh14", "document_type": "bo_luat", "normative_rank": 1, "article_number": "21", "clause_ref": "1"},
                score=1.0,
                matched_chunk_ids=("bll-21",),
                matched_citations=("Bo luat Lao dong 2019, Dieu 21, khoan 1",),
            ),
            RetrievalContext(
                chunk_id="tt10-3",
                citation_text="Thong tu 10/2020/TT-BLDTBXH, Dieu 3",
                text="Dieu 3. Noi dung chu yeu cua hop dong lao dong\n\nNoi dung chu yeu phai co cua hop dong lao dong theo khoan 1 Dieu 21 cua Bo luat Lao dong duoc quy dinh nhu sau:",
                payload={"document_id": "thong-tu-10-2020-tt-bldtbxh", "document_type": "thong_tu", "normative_rank": 3, "article_number": "3"},
                score=0.9,
                matched_chunk_ids=("tt10-3",),
                matched_citations=("Thong tu 10/2020/TT-BLDTBXH, Dieu 3",),
            ),
        )

        result = generate_grounded_answer(
            "Hop dong lao dong can co nhung noi dung gi?",
            contexts,
            provider="extractive",
        )
        quality = validate_answer_quality(
            "Hop dong lao dong can co nhung noi dung gi?",
            result.parsed,
            result.contexts,
            final_answer=result.answer,
        )

        self.assertTrue(result.validation.passed)
        self.assertTrue(quality.passed)
        self.assertEqual(quality.numeric_answer_present, "not_applicable")
        self.assertEqual(quality.yes_no_answer_present, "not_applicable")
        self.assertEqual(quality.exception_answer_present, "not_applicable")
        self.assertTrue(quality.conditions_listed)
        normalized_answer = normalize_for_matching(result.answer)
        self.assertIn("dia chi", normalized_answer)
        self.assertIn("muc luong", normalized_answer)
        self.assertIn("bao hiem xa hoi", normalized_answer)

    def test_minor_worker_generic_query_does_not_mention_14_year_old(self) -> None:
        contexts = (
            RetrievalContext(
                chunk_id="bll-145-1",
                citation_text="Bo luat Lao dong 2019, Dieu 145, khoan 1",
                text=(
                    "Dieu 145. Su dung nguoi chua du 15 tuoi lam viec\n\n"
                    "1. Khi su dung nguoi chua du 15 tuoi lam viec, nguoi su dung lao dong phai "
                    "giao ket hop dong lao dong bang van ban voi nguoi chua du 15 tuoi va nguoi dai dien theo phap luat; "
                    "bo tri gio lam viec khong anh huong den thoi gian hoc tap; co giay kham suc khoe phu hop; "
                    "kiem tra suc khoe dinh ky it nhat mot lan trong 06 thang; bao dam dieu kien an toan ve sinh lao dong."
                ),
                payload={"document_id": "45-2019-qh14", "document_type": "bo_luat", "normative_rank": 1, "article_number": "145", "clause_ref": "1"},
                score=1.0,
                matched_chunk_ids=("bll-145-1",),
                matched_citations=("Bo luat Lao dong 2019, Dieu 145, khoan 1",),
            ),
            RetrievalContext(
                chunk_id="tt09-3",
                citation_text="Thong tu 09/2020/TT-BLDTBXH, Dieu 3",
                text="Dieu 3. Nguoi su dung lao dong phai tuan thu Dieu 145 khi su dung nguoi chua du 15 tuoi lam viec.",
                payload={"document_id": "thong-tu-09-2020-tt-bldtbxh", "document_type": "thong_tu", "normative_rank": 3, "article_number": "3"},
                score=0.9,
                matched_chunk_ids=("tt09-3",),
                matched_citations=("Thong tu 09/2020/TT-BLDTBXH, Dieu 3",),
            ),
        )

        result = generate_grounded_answer(
            "Nguoi chua du 15 tuoi lam viec can dieu kien gi?",
            contexts,
            provider="extractive",
        )

        normalized_answer = normalize_for_matching(result.answer)
        self.assertNotIn("nguoi 14 tuoi", normalized_answer)
        self.assertIn("nguoi chua du 15 tuoi chi duoc lam viec neu", normalized_answer)

    def test_minor_worker_specific_14_year_old_query_may_mention_14_year_old(self) -> None:
        contexts = (
            RetrievalContext(
                chunk_id="bll-145-1",
                citation_text="Bo luat Lao dong 2019, Dieu 145, khoan 1",
                text=(
                    "Dieu 145. Su dung nguoi chua du 15 tuoi lam viec\n\n"
                    "1. Khi su dung nguoi chua du 15 tuoi lam viec, nguoi su dung lao dong phai "
                    "giao ket hop dong lao dong bang van ban voi nguoi chua du 15 tuoi va nguoi dai dien theo phap luat; "
                    "bo tri gio lam viec khong anh huong den thoi gian hoc tap; co giay kham suc khoe phu hop; "
                    "kiem tra suc khoe dinh ky it nhat mot lan trong 06 thang; bao dam dieu kien an toan ve sinh lao dong."
                ),
                payload={"document_id": "45-2019-qh14", "document_type": "bo_luat", "normative_rank": 1, "article_number": "145", "clause_ref": "1"},
                score=1.0,
                matched_chunk_ids=("bll-145-1",),
                matched_citations=("Bo luat Lao dong 2019, Dieu 145, khoan 1",),
            ),
        )

        result = generate_grounded_answer(
            "Nguoi 14 tuoi co duoc lam viec khong?",
            contexts,
            provider="extractive",
        )

        self.assertIn("nguoi 14 tuoi", normalize_for_matching(result.answer))

    def test_evidence_quote_cleanup_removes_repeated_headings_and_keeps_rules(self) -> None:
        contexts = (
            RetrievalContext(
                chunk_id="bll-21",
                citation_text="Bo luat Lao dong 2019, Dieu 21, khoan 1",
                text=(
                    "Dieu 21. Noi dung hop dong lao dong\n\n"
                    "Dieu 21. Noi dung hop dong lao dong\n\n"
                    "1. Hop dong lao dong phai co nhung noi dung chu yeu sau day:\n\n"
                    "1. Hop dong lao dong phai co nhung noi dung chu yeu sau day:\n\n"
                    "a) Ten, dia chi cua nguoi su dung lao dong va ho ten, chuc danh cua nguoi giao ket;\n\n"
                    "a) Ten, dia chi cua nguoi su dung lao dong va ho ten, chuc danh cua nguoi giao ket;\n\n"
                    "b) Ho ten, ngay thang nam sinh, gioi tinh, noi cu tru, giay to phap ly cua nguoi lao dong;\n\n"
                    "c) Cong viec va dia diem lam viec;\n\n"
                    "d) Thoi han cua hop dong lao dong;\n\n"
                    "dd) Muc luong, hinh thuc tra luong, thoi han tra luong, phu cap luong;\n\n"
                    "e) Che do nang bac, nang luong;\n\n"
                    "g) Thoi gio lam viec, thoi gio nghi ngoi;\n\n"
                    "h) Trang bi bao ho lao dong;\n\n"
                    "i) Bao hiem xa hoi, bao hiem y te va bao hiem that nghiep;\n\n"
                    "k) Dao tao, boi duong, nang cao trinh do, ky nang nghe."
                ),
                payload={"document_id": "45-2019-qh14", "document_type": "bo_luat", "normative_rank": 1, "article_number": "21", "clause_ref": "1"},
                score=1.0,
                matched_chunk_ids=("bll-21",),
                matched_citations=("Bo luat Lao dong 2019, Dieu 21, khoan 1",),
            ),
        )

        result = generate_grounded_answer(
            "Hop dong lao dong can co nhung noi dung gi?",
            contexts,
            provider="extractive",
        )

        quote = result.parsed.evidence_quotes[0].quote
        normalized_quote = normalize_for_matching(quote)
        self.assertEqual(normalized_quote.count("dieu 21 noi dung hop dong lao dong"), 0)
        self.assertEqual(
            normalized_quote.count("hop dong lao dong phai co nhung noi dung chu yeu sau day"),
            1,
        )
        self.assertIn("muc luong", normalized_quote)
        self.assertIn("bao hiem xa hoi", normalized_quote)

    def test_generic_answer_evidence_skips_heading_only_snippets(self) -> None:
        contexts = (
            RetrievalContext(
                chunk_id="bll-20",
                citation_text="Bo luat Lao dong 2019, Dieu 20, khoan 1",
                text=(
                    "HOP DONG LAO DONG;\n\n"
                    "1. Hop dong lao dong phai duoc giao ket theo mot trong cac loai sau day:\n\n"
                    "a) Hop dong lao dong khong xac dinh thoi han;\n\n"
                    "b) Hop dong lao dong xac dinh thoi han."
                ),
                payload={"document_id": "45-2019-qh14", "document_type": "bo_luat", "normative_rank": 1, "article_number": "20", "clause_ref": "1"},
                score=1.0,
                matched_chunk_ids=("bll-20",),
                matched_citations=("Bo luat Lao dong 2019, Dieu 20, khoan 1",),
            ),
        )

        result = generate_grounded_answer(
            "So sanh hop dong lao dong xac dinh thoi han va khong xac dinh thoi han?",
            contexts,
            provider="extractive",
        )
        quality = validate_answer_quality(
            "So sanh hop dong lao dong xac dinh thoi han va khong xac dinh thoi han?",
            result.parsed,
            result.contexts,
            final_answer=result.answer,
        )

        self.assertTrue(result.parsed.evidence_quotes)
        quote = normalize_for_matching(result.parsed.evidence_quotes[0].quote)
        self.assertNotEqual(quote, "hop dong lao dong")
        self.assertIn("hop dong lao dong phai duoc giao ket", quote)
        self.assertEqual(quality.low_information_quotes_count, 0)

    def test_contextual_override_answer_includes_inline_citation_after_parsing(self) -> None:
        contexts = (
            RetrievalContext(
                chunk_id="bll-26",
                citation_text="Bo luat Lao dong 2019, Dieu 26",
                text="Tien luong cua nguoi lao dong trong thoi gian thu viec it nhat phai bang 85% muc luong cua cong viec do.",
                payload={"document_id": "45-2019-qh14", "document_type": "bo_luat", "normative_rank": 1, "article_number": "26"},
                score=1.0,
                matched_chunk_ids=("bll-26",),
                matched_citations=("Bo luat Lao dong 2019, Dieu 26",),
            ),
        )

        result = generate_grounded_answer(
            "Luong thu viec toi thieu la bao nhieu?",
            contexts,
            provider="extractive",
        )
        quality = validate_answer_quality(
            "Luong thu viec toi thieu la bao nhieu?",
            result.parsed,
            result.contexts,
            final_answer=result.answer,
        )

        self.assertTrue(result.validation.passed)
        self.assertTrue(quality.passed)
        self.assertIn("Bo luat Lao dong 2019, Dieu 26", result.answer)

    def test_retirement_guidance_query_does_not_require_numeric_answer(self) -> None:
        contexts = (
            RetrievalContext(
                chunk_id="bll-169-2",
                citation_text="Bo luat Lao dong 2019, Dieu 169, khoan 2",
                text="Tuoi nghi huu cua nguoi lao dong trong dieu kien binh thuong duoc dieu chinh theo lo trinh cho den khi du 62 tuoi doi voi lao dong nam vao nam 2028 va du 60 tuoi doi voi lao dong nu vao nam 2035.",
                payload={"document_id": "45-2019-qh14", "document_type": "bo_luat", "normative_rank": 1, "article_number": "169", "clause_ref": "2"},
                score=1.0,
                matched_chunk_ids=("bll-169-2",),
                matched_citations=("Bo luat Lao dong 2019, Dieu 169, khoan 2",),
            ),
            RetrievalContext(
                chunk_id="nd135-4",
                citation_text="Nghi dinh 135/2020/ND-CP, Dieu 4",
                text="Dieu 4 quy dinh tuoi nghi huu trong dieu kien lao dong binh thuong va lo trinh tang tuoi nghi huu theo tung nam.",
                payload={"document_id": "nghi-dinh-135-2020-nd-cp", "document_type": "nghi_dinh", "normative_rank": 2, "article_number": "4"},
                score=0.9,
                matched_chunk_ids=("nd135-4",),
                matched_citations=("Nghi dinh 135/2020/ND-CP, Dieu 4",),
            ),
        )
        query = "Tuoi nghi huu theo BLLD Dieu 169 duoc Nghi dinh 135 huong dan the nao?"

        result = generate_grounded_answer(query, contexts, provider="extractive")
        quality = validate_answer_quality(
            query,
            result.parsed,
            result.contexts,
            final_answer=result.answer,
        )

        self.assertTrue(result.validation.passed)
        self.assertTrue(quality.passed)
        self.assertEqual(quality.numeric_answer_present, "not_applicable")
        self.assertIn("lo trinh", normalize_for_matching(result.answer))

    def test_generate_illegal_termination_answer_lists_article_40_obligations(self) -> None:
        contexts = (
            RetrievalContext(
                chunk_id="bll-40-1",
                citation_text="Bo luat Lao dong 2019, Dieu 40, khoan 1",
                text="Dieu 40. Nghia vu cua nguoi lao dong\n\n1. Khong duoc tro cap thoi viec.",
                payload={"document_id": "45-2019-qh14", "document_type": "bo_luat", "normative_rank": 1, "article_number": "40", "clause_ref": "1"},
                score=1.0,
                matched_chunk_ids=("bll-40-1",),
                matched_citations=("Bo luat Lao dong 2019, Dieu 40, khoan 1",),
            ),
            RetrievalContext(
                chunk_id="bll-40-2",
                citation_text="Bo luat Lao dong 2019, Dieu 40, khoan 2",
                text=(
                    "Dieu 40. Nghia vu cua nguoi lao dong\n\n"
                    "2. Phai boi thuong cho nguoi su dung lao dong nua thang tien luong theo hop dong lao dong "
                    "va mot khoan tien tuong ung voi tien luong theo hop dong lao dong trong nhung ngay khong bao truoc."
                ),
                payload={"document_id": "45-2019-qh14", "document_type": "bo_luat", "normative_rank": 1, "article_number": "40", "clause_ref": "2"},
                score=0.9,
                matched_chunk_ids=("bll-40-2",),
                matched_citations=("Bo luat Lao dong 2019, Dieu 40, khoan 2",),
            ),
            RetrievalContext(
                chunk_id="bll-40-3",
                citation_text="Bo luat Lao dong 2019, Dieu 40, khoan 3",
                text="Dieu 40. Nghia vu cua nguoi lao dong\n\n3. Phai hoan tra cho nguoi su dung lao dong chi phi dao tao quy dinh tai Dieu 62.",
                payload={"document_id": "45-2019-qh14", "document_type": "bo_luat", "normative_rank": 1, "article_number": "40", "clause_ref": "3"},
                score=0.8,
                matched_chunk_ids=("bll-40-3",),
                matched_citations=("Bo luat Lao dong 2019, Dieu 40, khoan 3",),
            ),
        )

        result = generate_grounded_answer(
            "Nguoi lao dong don phuong cham dut hop dong trai luat thi phai boi thuong gi?",
            contexts,
            provider="extractive",
        )
        quality = validate_answer_quality(
            "Nguoi lao dong don phuong cham dut hop dong trai luat thi phai boi thuong gi?",
            result.parsed,
            result.contexts,
            final_answer=result.answer,
        )

        self.assertTrue(result.validation.passed)
        self.assertTrue(quality.passed)
        normalized_answer = normalize_for_matching(result.answer)
        self.assertIn("khong duoc tro cap thoi viec", normalized_answer)
        self.assertIn("nua thang tien luong", normalized_answer)
        self.assertIn("chi phi dao tao", normalized_answer)


if __name__ == "__main__":
    unittest.main()
