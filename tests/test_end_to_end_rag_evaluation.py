from __future__ import annotations

import argparse
import unittest

from scripts.evaluate_end_to_end_rag import (
    aggregate_by_field,
    aggregate_by_category,
    answer_completeness_score,
    build_output,
    classify_failure_reasons,
    end_to_end_metrics_for_query,
    render_report,
    status_passed,
)
from scripts.evaluate_retrieval_modes import item_from_json


class EndToEndRagEvaluationTests(unittest.TestCase):
    def test_not_applicable_quality_status_counts_as_passed(self) -> None:
        quality = {
            "direct_answer_present": True,
            "required_legal_rule_present": True,
            "numeric_answer_present": "not_applicable",
            "yes_no_answer_present": "not_applicable",
            "conditions_listed": True,
            "exception_answer_present": "not_applicable",
            "no_article_title_only_answer": True,
            "all_legal_claims_have_citations": True,
        }

        self.assertTrue(status_passed("not_applicable"))
        self.assertEqual(answer_completeness_score(quality), 1.0)

    def test_failure_classification_distinguishes_retrieval_and_answer_causes(self) -> None:
        retrieval = {
            "missing_required_citations": ["BLLD Dieu 40"],
            "forbidden_citation_violations": ["BLLD Dieu 98"],
        }
        answer = {
            "direct_answer_present": False,
            "required_legal_rule_present": False,
            "citation_grounding_passed": False,
            "unsupported_article_numbers": ["99"],
            "low_information_quotes_count": 1,
            "ignores_higher_rank_context": True,
            "insufficient_context_handled": False,
        }

        reasons = classify_failure_reasons(retrieval_metrics=retrieval, answer_metrics=answer)

        self.assertIn("retrieval_missing_required_context", reasons)
        self.assertIn("retrieval_over_expansion", reasons)
        self.assertIn("answer_not_direct", reasons)
        self.assertIn("answer_missing_required_rule", reasons)
        self.assertIn("hallucinated_citation", reasons)
        self.assertIn("unsupported_article_number", reasons)
        self.assertIn("low_information_answer", reasons)
        self.assertIn("wrong_normative_priority", reasons)
        self.assertIn("insufficient_context_not_reported", reasons)

    def test_end_to_end_metrics_require_retrieval_answer_and_grounding(self) -> None:
        retrieval = {
            "retrieval_passed": True,
            "required_citation_coverage": 1.0,
            "forbidden_citation_violations": [],
            "missing_required_citations": [],
        }
        answer = {
            "answer_passed": True,
            "citation_grounding_passed": True,
            "answer_faithfulness_passed": True,
            "answer_completeness_score": 1.0,
            "direct_answer_present": True,
            "required_legal_rule_present": True,
            "unsupported_article_numbers": [],
            "low_information_quotes_count": 0,
            "ignores_higher_rank_context": False,
            "insufficient_context_handled": True,
        }

        metrics = end_to_end_metrics_for_query(
            retrieval_metrics=retrieval,
            answer_metrics=answer,
        )

        self.assertTrue(metrics["end_to_end_passed"])
        self.assertEqual(metrics["failure_reasons"], [])
        self.assertEqual(metrics["final_quality_score"], 100.0)

    def test_category_aggregation_reports_rates_and_average_score(self) -> None:
        results = [
            {
                "category": "direct_qa",
                "end_to_end_metrics": {
                    "retrieval_passed": True,
                    "answer_passed": True,
                    "citation_grounding_passed": True,
                    "end_to_end_passed": True,
                    "final_quality_score": 100.0,
                },
            },
            {
                "category": "direct_qa",
                "end_to_end_metrics": {
                    "retrieval_passed": False,
                    "answer_passed": True,
                    "citation_grounding_passed": True,
                    "end_to_end_passed": False,
                    "final_quality_score": 80.0,
                },
            },
        ]

        metrics = aggregate_by_category(results)

        self.assertEqual(metrics["direct_qa"]["query_count"], 2)
        self.assertEqual(metrics["direct_qa"]["retrieval_pass_rate"], 0.5)
        self.assertEqual(metrics["direct_qa"]["end_to_end_pass_rate"], 0.5)
        self.assertEqual(metrics["direct_qa"]["average_quality_score"], 90.0)

    def test_expanded_benchmark_metadata_is_parsed(self) -> None:
        item = item_from_json(
            {
                "id": "expanded_case",
                "query": "Nguoi lao dong la gi?",
                "category": "definition_qa",
                "topic": "khai niem nguoi lao dong",
                "expected_answer_points": ["dinh nghia", "co tra luong"],
                "difficulty": "easy",
                "requires_graph": True,
                "requires_normative_hierarchy": False,
                "expected_documents": ["45-2019-qh14"],
                "required_citations": [
                    {
                        "label": "BLLD Dieu 3 khoan 1",
                        "document_id": "45-2019-qh14",
                        "article_number": "3",
                        "clause_ref": "1",
                        "top_n": 5,
                    }
                ],
            }
        )

        self.assertEqual(item.topic, "khai niem nguoi lao dong")
        self.assertEqual(item.expected_answer_points, ("dinh nghia", "co tra luong"))
        self.assertEqual(item.difficulty, "easy")
        self.assertTrue(item.requires_graph)
        self.assertFalse(item.requires_normative_hierarchy)
        self.assertEqual(item.required_citations[0].clause_ref, "1")

    def test_field_aggregation_reports_topic_and_difficulty_rates(self) -> None:
        results = [
            {
                "topic": "lam them gio",
                "answer_metrics": {"quality_validation_passed": True},
                "end_to_end_metrics": {
                    "retrieval_passed": True,
                    "answer_passed": True,
                    "citation_grounding_passed": True,
                    "end_to_end_passed": True,
                    "final_quality_score": 100.0,
                },
            },
            {
                "topic": "lam them gio",
                "answer_metrics": {"quality_validation_passed": False},
                "end_to_end_metrics": {
                    "retrieval_passed": False,
                    "answer_passed": True,
                    "citation_grounding_passed": True,
                    "end_to_end_passed": False,
                    "final_quality_score": 70.0,
                },
            },
        ]

        metrics = aggregate_by_field(results, "topic")

        self.assertEqual(metrics["lam them gio"]["query_count"], 2)
        self.assertEqual(metrics["lam them gio"]["retrieval_pass_rate"], 0.5)
        self.assertEqual(metrics["lam them gio"]["quality_pass_rate"], 0.5)
        self.assertEqual(metrics["lam them gio"]["average_quality_score"], 85.0)

    def test_build_output_surfaces_no_hallucinated_citations(self) -> None:
        args = argparse.Namespace(
            benchmark_path="benchmark.jsonl",
            top_k=10,
            provider="extractive",
            model="",
        )
        results = [
            {
                "category": "direct_qa",
                "end_to_end_metrics": {
                    "end_to_end_passed": True,
                    "retrieval_passed": True,
                    "answer_passed": True,
                    "citation_grounding_passed": True,
                    "final_quality_score": 100.0,
                    "failure_reasons": [],
                },
                "answer_metrics": {
                    "quality_validation_passed": True,
                    "low_information_quotes_count": 0,
                    "unsupported_article_numbers": [],
                    "unretrieved_citations": [],
                },
                "retrieval_metrics": {
                    "graph_expansion_used": True,
                    "average_graph_depth": 2.0,
                    "missing_required_citations": [],
                    "forbidden_citation_violations": [],
                },
            }
        ]

        output = build_output(args=args, results=results, benchmark_count=1)

        self.assertTrue(output["passed"])
        self.assertEqual(output["overall"]["citation_validation_pass_rate"], 1.0)
        self.assertEqual(output["overall"]["low_information_quotes_count"], 0)
        self.assertEqual(output["overall"]["unsupported_article_numbers"], [])
        self.assertEqual(output["overall"]["unretrieved_citations"], [])
        self.assertIn("topic_metrics", output)
        self.assertIn("difficulty_metrics", output)

    def test_report_includes_expanded_evaluation_sections(self) -> None:
        args = argparse.Namespace(
            benchmark_path="expanded.jsonl",
            top_k=10,
            provider="extractive",
            model="",
        )
        output = build_output(
            args=args,
            benchmark_count=1,
            results=[
                {
                    "id": "case_1",
                    "query": "Nguoi lao dong la gi?",
                    "category": "definition_qa",
                    "topic": "khai niem nguoi lao dong",
                    "difficulty": "easy",
                    "requires_graph": False,
                    "requires_normative_hierarchy": False,
                    "answer": "Nguoi lao dong...",
                    "end_to_end_metrics": {
                        "end_to_end_passed": True,
                        "retrieval_passed": True,
                        "answer_passed": True,
                        "citation_grounding_passed": True,
                        "answer_faithfulness_passed": True,
                        "answer_completeness_score": 1.0,
                        "final_quality_score": 100.0,
                        "failure_reasons": [],
                    },
                    "answer_metrics": {
                        "citation_validation_passed": True,
                        "quality_validation_passed": True,
                        "low_information_quotes_count": 0,
                        "unsupported_article_numbers": [],
                        "unretrieved_citations": [],
                        "legal_claim_citation_coverage": 1.0,
                        "no_article_title_only_answer": True,
                        "insufficient_context_handled": True,
                    },
                    "retrieval_metrics": {
                        "required_citation_coverage": 1.0,
                        "required_citations_found": ["BLLD Dieu 3 khoan 1"],
                        "missing_required_citations": [],
                        "forbidden_citation_violations": [],
                        "retrieval_source_distribution": {"vector": 1},
                        "graph_expansion_used": False,
                        "average_graph_depth": 0.0,
                        "graph_depth": 0,
                        "graph_edge_types": {},
                        "top_k_contexts": [
                            {
                                "citation_text": "Bo luat Lao dong 2019, Dieu 3, khoan 1"
                            }
                        ],
                    },
                }
            ],
        )
        output["mode_comparison"] = {
            "overall_metrics": {
                "vector_only": {"recall_at_10": 0.5, "required_citation_coverage": 0.5},
                "hybrid": {"recall_at_10": 0.7, "required_citation_coverage": 0.7},
                "graph_augmented": {"recall_at_10": 1.0, "required_citation_coverage": 1.0},
            }
        }

        report = render_report(output)

        self.assertIn("## Per-Topic Results", report)
        self.assertIn("## Per-Difficulty Results", report)
        self.assertIn("## Graph-Required Results", report)
        self.assertIn("## Retrieval Mode And End-To-End Comparison", report)
        self.assertIn("end_to_end_graph_rag", report)


if __name__ == "__main__":
    unittest.main()
