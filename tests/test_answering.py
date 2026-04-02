from __future__ import annotations

import unittest

from vn_labor_law_ai_assistant.answering import parse_answer_payload, sanitize_legal_basis
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
            make_context("Bộ luật số 45/2019/QH 14, Điều 46, khoản 1"),
            make_context("Bộ luật số 45/2019/QH 14, Điều 46, khoản 2"),
        )

        legal_basis = sanitize_legal_basis(
            [
                "Bộ luật số 45/2019/QH 14, Điều 46, khoản 2",
                "Noi dung tro cap thoi viec",
            ],
            contexts,
        )

        self.assertEqual(legal_basis, ("Bộ luật số 45/2019/QH 14, Điều 46, khoản 2",))

    def test_parse_answer_payload_falls_back_to_context_citations(self) -> None:
        contexts = (
            make_context("Bộ luật số 45/2019/QH 14, Điều 46, khoản 1"),
            make_context("Bộ luật số 45/2019/QH 14, Điều 46, khoản 2"),
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

        self.assertEqual(
            parsed.legal_basis,
            (
                "Bộ luật số 45/2019/QH 14, Điều 46, khoản 1",
                "Bộ luật số 45/2019/QH 14, Điều 46, khoản 2",
            ),
        )


if __name__ == "__main__":
    unittest.main()
