from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
LLM_AS_JUDGE_ROOT = REPO_ROOT / "llm-as-judge"
sys.path.insert(0, str(LLM_AS_JUDGE_ROOT))

from ragas_eval.dataset_loader import (  # noqa: E402
    load_benchmark_samples,
    read_benchmark_records,
    validate_benchmark_schema,
)
from ragas_eval.legal_judge_prompt import (  # noqa: E402
    LegalJudgeParseError,
    parse_legal_judge_response,
)
from ragas_eval.summarize_results import build_summary, render_summary_markdown  # noqa: E402


class RagasDatasetLoaderTests(unittest.TestCase):
    def test_loader_maps_legacy_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "benchmark.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "LBR_001",
                        "question": "Cau hoi?",
                        "gold_answer": "Dap an chuan",
                        "generated_answer": "Cau tra loi",
                        "contexts": ["chunk 1"],
                        "gold_citations": ["Dieu 1"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            samples = load_benchmark_samples(
                path,
                require_response=True,
                require_retrieved_contexts=True,
            )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].user_input, "Cau hoi?")
        self.assertEqual(samples[0].reference, "Dap an chuan")
        self.assertEqual(samples[0].response, "Cau tra loi")
        self.assertEqual(samples[0].retrieved_contexts, ("chunk 1",))
        self.assertEqual(samples[0].gold_citation, "Dieu 1")

    def test_validation_reports_missing_required_fields_by_sample_id(self) -> None:
        records = [{"id": "LBR_BAD", "question": "Cau hoi?"}]

        report = validate_benchmark_schema(records)

        self.assertEqual(report.valid_count, 0)
        self.assertEqual(report.error_count, 1)
        self.assertEqual(report.errors[0].sample_id, "LBR_BAD")
        self.assertEqual(report.errors[0].field, "reference")

    def test_validate_existing_project_benchmark_without_response(self) -> None:
        path = REPO_ROOT / "eval" / "data" / "golden_benchmark_100_answered_v2.jsonl"
        records = read_benchmark_records(path)

        report = validate_benchmark_schema(records[:2])

        self.assertEqual(report.total_samples, 2)
        self.assertEqual(report.valid_count, 2)
        self.assertEqual(report.error_count, 0)


class LegalJudgeParserTests(unittest.TestCase):
    def test_parse_valid_legal_judge_json(self) -> None:
        score = parse_legal_judge_response(
            json.dumps(
                {
                    "legal_correctness": 0.8,
                    "citation_correctness": 1.0,
                    "legal_completeness": 0.7,
                    "legal_safety": 0.9,
                    "legal_overall_score": 0.85,
                    "error_type": "none",
                    "explanation": "Hop ly.",
                }
            )
        )

        self.assertEqual(score.error_type, "none")
        self.assertAlmostEqual(score.legal_overall_score, 0.85)

    def test_parser_rejects_out_of_range_scores(self) -> None:
        with self.assertRaises(LegalJudgeParseError):
            parse_legal_judge_response(
                json.dumps(
                    {
                        "legal_correctness": 1.2,
                        "citation_correctness": 1.0,
                        "legal_completeness": 0.7,
                        "legal_safety": 0.9,
                        "legal_overall_score": 0.85,
                        "error_type": "none",
                        "explanation": "Hop ly.",
                    }
                )
            )


class SummaryTests(unittest.TestCase):
    def test_summary_generation(self) -> None:
        rows = [
            {
                "id": "LBR_001",
                "user_input": "Q1",
                "faithfulness": "0.5",
                "legal_overall_score": "0.7",
                "overall_avg": "0.6",
                "error_type": "missing_condition",
                "judge_model": "gpt-5.4-pro",
            },
            {
                "id": "LBR_002",
                "user_input": "Q2",
                "faithfulness": "1.0",
                "legal_overall_score": "0.9",
                "overall_avg": "0.95",
                "error_type": "none",
                "judge_model": "gpt-5.4-pro",
            },
        ]

        summary = build_summary(rows, dataset="benchmark.jsonl")
        markdown = render_summary_markdown(summary)

        self.assertAlmostEqual(summary["overall_metrics"]["faithfulness"], 0.75)
        self.assertEqual(summary["error_type_distribution"]["missing_condition"], 1)
        self.assertIn("RAGAS Evaluation Summary", markdown)
        self.assertEqual(summary["worst_samples"][0]["id"], "LBR_001")


if __name__ == "__main__":
    unittest.main()
