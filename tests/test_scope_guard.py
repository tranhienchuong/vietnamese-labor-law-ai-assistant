from __future__ import annotations

import json
from pathlib import Path
import unittest

from vn_labor_law_ai_assistant.rag.answering import generate_grounded_answer, validate_answer_quality
from vn_labor_law_ai_assistant.rag.scope_guard import assess_question_domain, assess_scope
from vn_labor_law_ai_assistant.retriever import RetrievalContext


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = REPO_ROOT / "artifacts" / "evaluation" / "golden_benchmark_100_extended.jsonl"


def make_context(
    citation_text: str = "Bo luat Lao dong 2019, Dieu 46",
    text: str = "Tro cap thoi viec duoc tinh theo thoi gian lam viec va tien luong.",
    *,
    score: float = 5.0,
    payload: dict[str, object] | None = None,
) -> RetrievalContext:
    return RetrievalContext(
        chunk_id=citation_text,
        citation_text=citation_text,
        text=text,
        payload={
            "document_id": "45-2019-qh14",
            "document_type": "bo_luat",
            "normative_rank": 1,
            "article_number": "46",
            "retrieval_text": text,
            "final_score": score,
            **(payload or {}),
        },
        score=score,
        matched_chunk_ids=(citation_text,),
        matched_citations=(citation_text,),
    )


class ScopeGuardTests(unittest.TestCase):
    def test_domain_guard_rejects_non_legal_chatter(self) -> None:
        for question in ("alo", "co tft", "hom nay an gi"):
            with self.subTest(question=question):
                decision = assess_question_domain(question)
                self.assertTrue(decision.out_of_domain)
                self.assertFalse(decision.matched_signals)

    def test_domain_guard_allows_configured_labor_questions(self) -> None:
        for question in (
            "Nguoi duoi 15 tuoi co duoc lam viec khong?",
            "What rules apply to workers under 15?",
            "Cong ty sa thai toi co dung luat khong?",
            "Dieu 35 quy dinh gi?",
        ):
            with self.subTest(question=question):
                decision = assess_question_domain(question)
                self.assertFalse(decision.out_of_domain)

    def test_frozen_out_of_corpus_items_are_detected_without_ids(self) -> None:
        if not BENCHMARK_PATH.exists():
            raise unittest.SkipTest(
                "Artifact not available in checkout: artifacts/evaluation/golden_benchmark_100_extended.jsonl"
            )

        items = []
        for line in BENCHMARK_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("category") == "out_of_corpus_qa" and not item.get("required_citations"):
                items.append(item)

        self.assertEqual(len(items), 6)
        unrelated_contexts = (
            make_context(),
            make_context(
                "Nghi dinh 145/2020/ND-CP, Dieu 89",
                "Nguoi su dung lao dong giup viec gia dinh phai tra tien luong dung han.",
                score=0.8,
                payload={"document_id": "nghi-dinh-145-2020-nd-cp", "article_number": "89"},
            ),
        )

        for item in items:
            with self.subTest(query=item["query"]):
                decision = assess_scope(str(item["query"]), unrelated_contexts)
                self.assertTrue(decision.out_of_scope)
                self.assertEqual(decision.topic_context_matches, 0)

    def test_generation_refuses_without_unrelated_legal_basis(self) -> None:
        result = generate_grounded_answer(
            "Tro cap thoi viec co tinh thue thu nhap ca nhan khong?",
            (
                make_context(
                    "Bo luat Lao dong 2019, Dieu 46",
                    "Dieu 46. Tro cap thoi viec. Moi nam lam viec duoc tro cap mot nua thang tien luong.",
                ),
            ),
            provider="extractive",
        )
        quality = validate_answer_quality(
            "Tro cap thoi viec co tinh thue thu nhap ca nhan khong?",
            result.parsed,
            result.contexts,
            final_answer=result.answer,
        )

        self.assertEqual(result.generation_method, "scope_guard")
        self.assertTrue(result.parsed.insufficient_context)
        self.assertEqual(result.parsed.legal_basis, ())
        self.assertEqual(result.parsed.evidence_quotes, ())
        self.assertTrue(result.validation.passed)
        self.assertTrue(quality.passed)

    def test_supported_adjacent_topic_is_not_blocked(self) -> None:
        decision = assess_scope(
            "Hop dong lao dong co noi dung ve bao hiem xa hoi khong?",
            (
                make_context(
                    "Bo luat Lao dong 2019, Dieu 21",
                    "Hop dong lao dong phai co noi dung ve bao hiem xa hoi, bao hiem y te va bao hiem that nghiep.",
                    payload={"article_number": "21"},
                ),
            ),
        )

        self.assertFalse(decision.out_of_scope)

    def test_direct_topic_evidence_prevents_refusal(self) -> None:
        decision = assess_scope(
            "Ty le dong BHXH cua cong ty la bao nhieu?",
            (
                make_context(
                    "Van ban gia dinh, Dieu 1",
                    "Ty le dong bao hiem xa hoi cua cong ty duoc quy dinh la 1%.",
                ),
            ),
        )

        self.assertFalse(decision.out_of_scope)
        self.assertGreater(decision.topic_context_matches, 0)

    def test_related_but_incomplete_context_still_refuses(self) -> None:
        wage_decision = assess_scope(
            "Nam 2026 luong toi thieu vung I, II, III, IV la bao nhieu?",
            (
                make_context(
                    "Nghi dinh 145/2020/ND-CP, Dieu 89",
                    "Tien luong cua nguoi lao dong giup viec gia dinh khong thap hon muc luong toi thieu vung do Chinh phu cong bo.",
                    payload={"document_id": "nghi-dinh-145-2020-nd-cp", "article_number": "89"},
                ),
            ),
        )
        permit_decision = assess_scope(
            "Ho so xin giay phep lao dong cho nguoi nuoc ngoai nam 2026 gom nhung gi?",
            (
                make_context(
                    "Bo luat Lao dong 2019, Dieu 154",
                    "Nguoi lao dong nuoc ngoai lam viec tai Viet Nam khong thuoc dien cap giay phep lao dong trong mot so truong hop.",
                    payload={"article_number": "154"},
                ),
            ),
        )

        self.assertTrue(wage_decision.out_of_scope)
        self.assertTrue(permit_decision.out_of_scope)


if __name__ == "__main__":
    unittest.main()
