from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from openpyxl import Workbook

from vn_labor_law_ai_assistant.evaluation import (
    BenchmarkCase,
    citation_matches_expected,
    load_benchmark_workbook,
    parse_benchmark_rows,
    retrieval_hit_at_k,
    score_citation_correctness,
)


class EvaluationTests(unittest.TestCase):
    def test_parse_benchmark_rows_skips_intro_rows_and_parses_cases(self) -> None:
        rows = [
            ("Golden Benchmark Template", None, None),
            (None, None, None),
            (
                "id",
                "category",
                "subtopic",
                "difficulty",
                "question_type",
                "question",
                "scenario",
                "gold_issue",
                "gold_citation_primary",
                "gold_citation_secondary",
                "gold_answer_short",
                "gold_answer_full",
                "abstain_required",
                "missing_information",
                "source_document",
                "source_url",
                "annotator",
                "review_status",
                "notes",
            ),
            (
                "LBR_001",
                "employee_unilateral_termination",
                "indefinite-term resignation grounds",
                "easy",
                "direct_qa",
                "Cau hoi?",
                "Tinh huong",
                "Gold issue",
                "Điều 35 khoản 1 điểm a Bộ luật Lao động 2019",
                None,
                "Tra loi ngan",
                "Tra loi day du",
                "No",
                None,
                "Bộ luật Lao động 2019",
                "https://example.test",
                "annotator",
                "answered_draft",
                "notes",
            ),
        ]

        cases = parse_benchmark_rows(rows)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].id, "LBR_001")
        self.assertFalse(cases[0].abstain_required)
        self.assertEqual(cases[0].gold_citation_primary, "Điều 35 khoản 1 điểm a Bộ luật Lao động 2019")

    def test_load_benchmark_workbook_reads_real_xlsx(self) -> None:
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = "golden_benchmark"
        worksheet.append(["Golden Benchmark Template"])
        worksheet.append([None])
        worksheet.append(
            [
                "id",
                "category",
                "subtopic",
                "difficulty",
                "question_type",
                "question",
                "scenario",
                "gold_issue",
                "gold_citation_primary",
                "gold_citation_secondary",
                "gold_answer_short",
                "gold_answer_full",
                "abstain_required",
                "missing_information",
                "source_document",
                "source_url",
                "annotator",
                "review_status",
                "notes",
            ]
        )
        worksheet.append(
            [
                "LBR_002",
                "notice_period",
                "fixed-term notice period",
                "medium",
                "time_limit",
                "Bao truoc bao lau?",
                "Tinh huong",
                "Gold issue",
                "Điều 35 khoản 1 điểm b Bộ luật Lao động 2019",
                None,
                "Tra loi ngan",
                "Tra loi day du",
                "Yes",
                "Can biet thoi han hop dong",
                "Bộ luật Lao động 2019",
                "https://example.test",
                "annotator",
                "answered_draft",
                "notes",
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = Path(tmpdir) / "benchmark.xlsx"
            workbook.save(workbook_path)
            cases = load_benchmark_workbook(workbook_path)

        self.assertEqual(len(cases), 1)
        self.assertTrue(cases[0].abstain_required)
        self.assertEqual(cases[0].id, "LBR_002")

    def test_citation_matches_expected_uses_legal_signature(self) -> None:
        expected = "Điều 35 khoản 1 điểm a Bộ luật Lao động 2019"
        observed = "Bộ luật số 45/2019/QH 14, Điều 35 (Quyền đơn phương chấm dứt hợp đồng lao động của người lao động), khoản 1, điểm a"

        self.assertTrue(citation_matches_expected(expected, observed))

    def test_retrieval_hit_and_citation_scoring(self) -> None:
        case = BenchmarkCase(
            id="LBR_003",
            category="employee_unilateral_termination",
            subtopic="sexual harassment and immediate resignation",
            difficulty="medium",
            question_type="direct_qa",
            question="Cau hoi?",
            scenario="Tinh huong",
            gold_issue="Gold issue",
            gold_citation_primary="Điều 35 khoản 2 điểm d Bộ luật Lao động 2019",
            gold_citation_secondary="Điều 35 khoản 1 điểm a Bộ luật Lao động 2019",
            gold_answer_short="Tra loi ngan",
            gold_answer_full="Tra loi day du",
            abstain_required=False,
            missing_information=None,
            source_document="Bộ luật Lao động 2019",
            source_url=None,
            annotator=None,
            review_status=None,
            notes=None,
        )
        observed = [
            "Bộ luật số 45/2019/QH 14, Điều 35, khoản 2, điểm d",
            "Bộ luật số 45/2019/QH 14, Điều 34, khoản 1",
        ]

        self.assertTrue(retrieval_hit_at_k(case, observed, k=5))
        self.assertEqual(score_citation_correctness(case, observed), "partial")


if __name__ == "__main__":
    unittest.main()
