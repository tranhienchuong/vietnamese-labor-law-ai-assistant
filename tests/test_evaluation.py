from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from openpyxl import Workbook

from vn_labor_law_ai_assistant.evaluation import (
    BenchmarkCase,
    compute_final_score_10,
    citation_matches_expected,
    score_citation_document_correctness_for_scope,
    expected_citation_scope,
    expected_citations,
    expected_citations_in_scope,
    expected_citations_out_of_scope,
    first_relevant_rank,
    load_benchmark_workbook,
    mean_reciprocal_rank,
    parse_judge_payload,
    parse_benchmark_rows,
    reciprocal_rank,
    result_columns,
    retrieval_hit_at_k,
    score_citation_article_correctness_for_scope,
    score_citation_correctness,
    score_citation_correctness_for_scope,
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
        self.assertEqual(
            expected_citations(cases[0]),
            ("Điều 35 khoản 1 điểm a Bộ luật Lao động 2019",),
        )
        self.assertEqual(cases[0].skill_tag, "legal_classification")

    def test_parse_benchmark_rows_skips_placeholder_cases(self) -> None:
        rows = [
            ("Golden Benchmark Template", None, None),
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
                "notice_period",
                "employee resignation",
                "easy",
                "direct_qa",
                "Can an employee resign without prior notice?",
                "Placeholder scenario",
                "Placeholder issue",
                "Labor Code 2019 - [Article], [Clause]",
                None,
                "Placeholder short answer",
                "Placeholder full answer",
                "No",
                None,
                "Labor Code 2019",
                "https://example.com/source",
                "initials",
                "draft",
                "Replace the placeholder citation with the exact article/clause.",
            ),
            (
                "LBR_002",
                "labor_contract",
                "contract types",
                "easy",
                "direct_qa",
                "Co may loai hop dong?",
                "Tinh huong",
                "Gold issue",
                "Điều 20 khoản 1 Bộ luật Lao động 2019",
                None,
                "Tra loi ngan",
                "Tra loi day du",
                "No",
                None,
                "Bộ luật Lao động 2019",
                None,
                None,
                None,
                None,
            ),
        ]

        cases = parse_benchmark_rows(rows)

        self.assertEqual([case.id for case in cases], ["LBR_002"])

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

    def test_expected_citations_expand_compound_references(self) -> None:
        case = BenchmarkCase(
            id="LBR_004",
            category="notice_period",
            subtopic="compound citations",
            difficulty="medium",
            question_type="direct_qa",
            question="Cau hoi?",
            scenario="Tinh huong",
            gold_issue="Gold issue",
            gold_citation_primary="Điều 35 khoản 1 điểm b, c Bộ luật Lao động 2019",
            gold_citation_secondary="Điều 97 khoản 4 Bộ luật Lao động 2019",
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

        self.assertEqual(
            expected_citations(case),
            (
                "Điều 35 khoản 1 điểm b Bộ luật Lao động 2019",
                "Điều 35 khoản 1 điểm c Bộ luật Lao động 2019",
                "Điều 97 khoản 4 Bộ luật Lao động 2019",
            ),
        )

    def test_expected_citations_preserve_e_transaction_law_family(self) -> None:
        law = "Lu\u1eadt Giao d\u1ecbch \u0111i\u1ec7n t\u1eed 2023"
        case = BenchmarkCase(
            id="LBR_050",
            category="out_of_scope",
            subtopic="electronic transaction law",
            difficulty="medium",
            question_type="direct_qa",
            question="Cau hoi?",
            scenario="Tinh huong",
            gold_issue="Gold issue",
            gold_citation_primary=f"\u0110i\u1ec1u 9 {law}",
            gold_citation_secondary=f"\u0110i\u1ec1u 11 {law}; \u0110i\u1ec1u 16 {law}",
            gold_answer_short="Tra loi ngan",
            gold_answer_full="Tra loi day du",
            abstain_required=True,
            missing_information=None,
            source_document=law,
            source_url=None,
            annotator=None,
            review_status=None,
            notes=None,
        )

        self.assertEqual(
            expected_citations(case),
            (
                f"\u0110i\u1ec1u 9 {law}",
                f"\u0110i\u1ec1u 11 {law}",
                f"\u0110i\u1ec1u 16 {law}",
            ),
        )

    def test_scope_partition_keeps_in_scope_and_excludes_out_of_scope(self) -> None:
        case = BenchmarkCase(
            id="LBR_005",
            category="mixed_scope",
            subtopic="mixed scope",
            difficulty="medium",
            question_type="direct_qa",
            question="Cau hoi?",
            scenario="Tinh huong",
            gold_issue="Gold issue",
            gold_citation_primary="Điều 35 Bộ luật Lao động 2019; Điều 39 Nghị định 12/2022/NĐ-CP",
            gold_citation_secondary=None,
            gold_answer_short="Tra loi ngan",
            gold_answer_full="Tra loi day du",
            abstain_required=False,
            missing_information=None,
            source_document="Bo luat Lao dong 2019",
            source_url=None,
            annotator=None,
            review_status=None,
            notes=None,
        )
        allowed_families = ("bo_luat_2019", "nghi_dinh_145")
        observed = ["Bộ luật số 45/2019/QH14, Điều 35"]

        self.assertEqual(
            expected_citations_in_scope(case, allowed_document_families=allowed_families),
            ("Điều 35 Bộ luật Lao động 2019",),
        )
        self.assertEqual(
            expected_citations_out_of_scope(case, allowed_document_families=allowed_families),
            ("Điều 39 Nghị định 12/2022/NĐ-CP",),
        )
        self.assertEqual(
            expected_citation_scope(case, allowed_document_families=allowed_families),
            "mixed_scope",
        )
        self.assertTrue(
            retrieval_hit_at_k(
                case,
                observed,
                k=5,
                allowed_document_families=allowed_families,
            )
        )
        self.assertEqual(
            score_citation_correctness_for_scope(
                case,
                observed,
                allowed_document_families=allowed_families,
            ),
            "exact",
        )

    def test_scope_partition_marks_fully_out_of_scope_cases_as_na(self) -> None:
        case = BenchmarkCase(
            id="LBR_006",
            category="out_of_scope",
            subtopic="out of scope",
            difficulty="medium",
            question_type="direct_qa",
            question="Cau hoi?",
            scenario="Tinh huong",
            gold_issue="Gold issue",
            gold_citation_primary="Luật Công đoàn 2024",
            gold_citation_secondary=None,
            gold_answer_short="Tra loi ngan",
            gold_answer_full="Tra loi day du",
            abstain_required=False,
            missing_information=None,
            source_document="Luat Cong doan 2024",
            source_url=None,
            annotator=None,
            review_status=None,
            notes=None,
        )
        allowed_families = ("bo_luat_2019", "nghi_dinh_145")

        self.assertIsNone(
            retrieval_hit_at_k(
                case,
                ["Bộ luật số 45/2019/QH14, Điều 63"],
                k=5,
                allowed_document_families=allowed_families,
            )
        )
        self.assertEqual(
            expected_citation_scope(case, allowed_document_families=allowed_families),
            "out_of_scope",
        )
        self.assertEqual(
            score_citation_correctness_for_scope(
                case,
                ["Bộ luật số 45/2019/QH14, Điều 63"],
                allowed_document_families=allowed_families,
            ),
            "na",
        )

    def test_rank_metrics_capture_first_match_and_reciprocal_rank(self) -> None:
        case = BenchmarkCase(
            id="LBR_007",
            category="ranking",
            subtopic="mrr",
            difficulty="easy",
            question_type="direct_qa",
            question="Cau hoi?",
            scenario="Tinh huong",
            gold_issue="Gold issue",
            gold_citation_primary="Điều 35 khoản 2 điểm d Bộ luật Lao động 2019",
            gold_citation_secondary=None,
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
            "Bộ luật số 45/2019/QH14, Điều 34, khoản 1",
            "Bộ luật số 45/2019/QH14, Điều 35, khoản 2, điểm d",
            "Bộ luật số 45/2019/QH14, Điều 41, khoản 1",
        ]

        self.assertEqual(first_relevant_rank(case, observed, k=5), 2)
        self.assertEqual(reciprocal_rank(case, observed, k=5), 0.5)
        self.assertEqual(mean_reciprocal_rank([1.0, 0.5, 0.0, None]), 0.5)

    def test_citation_document_and_article_scores_are_separate(self) -> None:
        case = BenchmarkCase(
            id="LBR_008",
            category="labor_discipline",
            subtopic="dismissal",
            difficulty="medium",
            question_type="direct_qa",
            question="Cau hoi?",
            scenario="Tinh huong",
            gold_issue="Gold issue",
            gold_citation_primary="Điều 125 Bộ luật Lao động 2019",
            gold_citation_secondary=None,
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
        observed = ["Bộ luật số 45/2019/QH14, Điều 36"]

        self.assertEqual(
            score_citation_document_correctness_for_scope(case, observed),
            "exact",
        )
        self.assertEqual(
            score_citation_article_correctness_for_scope(case, observed),
            "no",
        )

    def test_compute_final_score_uses_missing_information_formula(self) -> None:
        case = BenchmarkCase(
            id="LBR_009",
            category="notice_period",
            subtopic="missing contract term",
            difficulty="medium",
            question_type="time_limit",
            question="Bao truoc bao lau?",
            scenario="Tinh huong",
            gold_issue="Gold issue",
            gold_citation_primary="Điều 35 Bộ luật Lao động 2019",
            gold_citation_secondary=None,
            gold_answer_short="Tra loi ngan",
            gold_answer_full="Tra loi day du",
            abstain_required=False,
            missing_information="Can biet loai hop dong.",
            source_document="Bộ luật Lao động 2019",
            source_url=None,
            annotator=None,
            review_status=None,
            notes=None,
        )

        self.assertEqual(
            compute_final_score_10(
                case=case,
                answer_correct="yes",
                legal_issue_classification_correct="yes",
                legal_reasoning_score_1_5=5,
                missing_information_score_0_2=2,
                citation_article_correct="exact",
                citation_supports_answer="yes",
                groundedness_score_1_5=5,
                clarity_score_1_5=5,
                format_score_1_5=5,
            ),
            10,
        )
        self.assertEqual(
            compute_final_score_10(
                case=case,
                answer_correct="yes",
                legal_issue_classification_correct="yes",
                legal_reasoning_score_1_5=5,
                missing_information_score_0_2=0,
                citation_article_correct="exact",
                citation_supports_answer="yes",
                groundedness_score_1_5=5,
                clarity_score_1_5=5,
                format_score_1_5=5,
            ),
            8,
        )

    def test_result_columns_renames_retrieval_hit_column_for_top_k(self) -> None:
        columns = result_columns("retrieval_hit_at_10")

        self.assertIn("retrieval_hit_at_10", columns)
        self.assertNotIn("retrieval_hit_at_5", columns)
        self.assertIn("citation_provision_correct", columns)
        self.assertIn("skill_tag", columns)
        self.assertEqual(columns[2], "retrieval_hit_at_10")

    def test_parse_judge_payload_reads_valid_json(self) -> None:
        score = parse_judge_payload(
            """
            {
              "answer_correct": "partial",
              "legal_issue_classification_correct": "yes",
              "legal_reasoning_score_1_5": 3,
              "missing_information_score_0_2": 2,
              "citation_supports_answer": "partial",
              "groundedness_score_1_5": 2,
              "clarity_score_1_5": 4,
              "format_score_1_5": 5,
              "hallucination_types": ["rule"],
              "comments": "Thiếu một ý quan trọng."
            }
            """
        )

        self.assertIsNotNone(score)
        assert score is not None
        self.assertEqual(score.answer_correct, "partial")
        self.assertEqual(score.legal_issue_classification_correct, "yes")
        self.assertEqual(score.legal_reasoning_score_1_5, 3)
        self.assertEqual(score.missing_information_score_0_2, 2)
        self.assertEqual(score.citation_supports_answer, "partial")
        self.assertEqual(score.groundedness_score_1_5, 2)
        self.assertEqual(score.clarity_score_1_5, 4)
        self.assertEqual(score.format_score_1_5, 5)
        self.assertEqual(score.hallucination_types, ("rule",))

    def test_parse_judge_payload_rejects_invalid_json(self) -> None:
        self.assertIsNone(parse_judge_payload("khong phai json"))


if __name__ == "__main__":
    unittest.main()
