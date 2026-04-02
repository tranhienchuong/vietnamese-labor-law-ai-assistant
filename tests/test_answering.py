from __future__ import annotations

import unittest

from vn_labor_law_ai_assistant.answering import (
    build_messages,
    extract_json_candidate,
    parse_answer_payload,
    sanitize_legal_basis,
)
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

    def test_sanitize_legal_basis_respects_empty_response(self) -> None:
        contexts = (
            make_context("Bo luat so 45/2019/QH 14, Dieu 36, khoan 1"),
            make_context("Bo luat so 45/2019/QH 14, Dieu 36, khoan 2"),
        )

        self.assertEqual(sanitize_legal_basis([], contexts), ())
        self.assertEqual(sanitize_legal_basis(None, contexts), ())

    def test_parse_answer_payload_does_not_invent_citations_when_model_returns_invalid_basis(self) -> None:
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
        self.assertTrue(parsed.insufficient_context)
        self.assertIn("khong the xac nhan", parsed.answer.lower())
        self.assertIn("vo hieu hoa", parsed.notes.lower())

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

    def test_build_messages_respects_context_char_budget(self) -> None:
        contexts = (
            RetrievalContext(
                chunk_id="chunk-1",
                citation_text="Bo luat so 45/2019/QH 14, Dieu 36, khoan 1",
                text="A" * 500,
                payload={},
                score=1.0,
                matched_chunk_ids=("chunk-1",),
                matched_citations=("Bo luat so 45/2019/QH 14, Dieu 36, khoan 1",),
            ),
        )

        messages = build_messages(
            "Cau hoi mau?",
            contexts,
            max_context_chars=180,
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Cau hoi:\nCau hoi mau?", messages[1]["content"])
        self.assertIn("CONTEXT:\n[NGU CANH 1]", messages[1]["content"])
        context_text = messages[1]["content"].split("CONTEXT:\n", 1)[1]
        self.assertLessEqual(len(context_text), 180)


if __name__ == "__main__":
    unittest.main()
