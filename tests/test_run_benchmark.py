from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
LLM_AS_JUDGE_ROOT = REPO_ROOT / "llm-as-judge"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LLM_AS_JUDGE_ROOT))

from ragas_eval.dataset_loader import load_benchmark_samples  # noqa: E402
from scripts.run_benchmark import build_ragas_export_row  # noqa: E402
from vn_labor_law_ai_assistant.evaluation import BenchmarkCase, load_benchmark_jsonl  # noqa: E402
from vn_labor_law_ai_assistant.rag.retrieval.models import RetrievalContext  # noqa: E402


class RunBenchmarkRagasExportTests(unittest.TestCase):
    def test_build_ragas_export_row_matches_loader_schema(self) -> None:
        case = BenchmarkCase(
            id="LBR_001",
            category="contract",
            subtopic="definition",
            difficulty="easy",
            question_type="direct_qa",
            question="Cau hoi?",
            scenario="Tinh huong",
            gold_issue="Van de",
            gold_citation_primary="Dieu 1 Bo luat Lao dong 2019",
            gold_citation_secondary=None,
            gold_answer_short="Tra loi ngan",
            gold_answer_full="Tra loi day du",
            abstain_required=False,
            missing_information=None,
            source_document=None,
            source_url=None,
            annotator=None,
            review_status=None,
            notes=None,
            reference_contexts=("Ngu canh chuan",),
        )
        context = RetrievalContext(
            chunk_id="chunk-1",
            citation_text="Dieu 1",
            text="Ngu canh da dua vao prompt",
            payload={},
            score=1.0,
            matched_chunk_ids=("chunk-1",),
            matched_citations=("Dieu 1",),
        )

        row = build_ragas_export_row(
            case=case,
            generated_answer="Cau tra loi sinh ra",
            prompt_contexts=(context,),
            expected_citations_scoped=(),
            expected_citations_all=("Dieu 1 Bo luat Lao dong 2019",),
        )

        self.assertEqual(row["id"], "LBR_001")
        self.assertEqual(row["user_input"], "Cau hoi?")
        self.assertEqual(row["response"], "Cau tra loi sinh ra")
        self.assertEqual(row["retrieved_contexts"], ["Ngu canh da dua vao prompt"])
        self.assertTrue(all(isinstance(value, str) for value in row["retrieved_contexts"]))
        self.assertEqual(row["reference"], "Tra loi day du")
        self.assertEqual(row["reference_contexts"], ["Ngu canh chuan"])
        self.assertEqual(row["gold_citation"], "Dieu 1 Bo luat Lao dong 2019")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ragas_export.jsonl"
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            samples = load_benchmark_samples(
                path,
                require_response=True,
                require_retrieved_contexts=True,
            )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].id, "LBR_001")

    def test_ragas_template_jsonl_can_be_used_as_benchmark_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ragas_template.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "LBR_002",
                        "user_input": "Cau hoi template?",
                        "reference": "Dap an chuan",
                        "reference_contexts": ["Ngu canh chuan"],
                        "gold_citation": "Dieu 2 | Dieu 3",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            cases = load_benchmark_jsonl(path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].question, "Cau hoi template?")
        self.assertEqual(cases[0].gold_answer_full, "Dap an chuan")
        self.assertEqual(cases[0].reference_contexts, ("Ngu canh chuan",))
        self.assertEqual(cases[0].gold_citations, ("\u0110i\u1ec1u 2", "\u0110i\u1ec1u 3"))


if __name__ == "__main__":
    unittest.main()
