from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import LLM_AS_JUDGE_ROOT


RAGAS_METRIC_COLUMNS = (
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "factual_correctness",
    "context_precision",
    "context_recall",
)
LEGAL_METRIC_COLUMNS = (
    "legal_correctness",
    "citation_correctness",
    "legal_completeness",
    "legal_safety",
    "legal_overall_score",
)


@dataclass(frozen=True)
class SummaryArtifacts:
    json_path: Path
    markdown_path: Path


def load_result_rows(input_path: Path) -> list[dict[str, Any]]:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in input_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return list(payload["rows"])
        if isinstance(payload, list):
            return list(payload)
    raise ValueError("Unsupported result format. Expected .csv, .jsonl, or .json.")


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str = "",
    judge_model: str = "",
    provider: str = "TheSparkDaily",
    run_name: str = "",
) -> dict[str, Any]:
    metric_columns = tuple(
        column
        for column in (*RAGAS_METRIC_COLUMNS, *LEGAL_METRIC_COLUMNS, "overall_avg")
        if any(_numeric_value(row.get(column)) is not None for row in rows)
    )
    averages = {
        column: _mean(_numeric_value(row.get(column)) for row in rows)
        for column in metric_columns
    }
    error_distribution = Counter(str(row.get("error_type") or "none") for row in rows)
    worst_samples = sorted(
        rows,
        key=lambda row: _numeric_value(row.get("overall_avg"))
        if _numeric_value(row.get("overall_avg")) is not None
        else math.inf,
    )[:10]

    return {
        "configuration": {
            "judge_provider": provider,
            "judge_model": judge_model or _unique_join(row.get("judge_model") for row in rows),
            "dataset": dataset,
            "run_name": run_name,
            "number_of_samples": len(rows),
        },
        "overall_metrics": {
            key: value
            for key, value in averages.items()
            if key in RAGAS_METRIC_COLUMNS or key == "overall_avg"
        },
        "legal_judge_metrics": {
            key: value for key, value in averages.items() if key in LEGAL_METRIC_COLUMNS
        },
        "error_type_distribution": dict(sorted(error_distribution.items())),
        "worst_samples": [
            {
                "id": row.get("id", ""),
                "question": row.get("user_input", ""),
                "overall_score": _numeric_value(row.get("overall_avg")),
                "error_type": row.get("error_type", "none") or "none",
            }
            for row in worst_samples
        ],
    }


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    config = summary["configuration"]
    lines = [
        "# RAGAS Evaluation Summary",
        "",
        "## Configuration",
        f"- Judge provider: {config.get('judge_provider', '')}",
        f"- Judge model: {config.get('judge_model', '')}",
        f"- Dataset: {config.get('dataset', '')}",
        f"- Number of samples: {config.get('number_of_samples', 0)}",
        "",
        "## Overall Metrics",
        "| Metric | Score |",
        "|---|---:|",
    ]
    lines.extend(_metric_rows(summary.get("overall_metrics", {})))
    lines.extend(
        [
            "",
            "## Legal Judge Metrics",
            "| Metric | Score |",
            "|---|---:|",
        ]
    )
    lines.extend(_metric_rows(summary.get("legal_judge_metrics", {})))
    lines.extend(
        [
            "",
            "## Error Type Distribution",
            "| Error Type | Count |",
            "|---|---:|",
        ]
    )
    for error_type, count in dict(summary.get("error_type_distribution", {})).items():
        lines.append(f"| {error_type} | {count} |")
    lines.extend(
        [
            "",
            "## Worst Samples",
            "| ID | Question | Overall Score | Error Type |",
            "|---|---|---:|---|",
        ]
    )
    for row in summary.get("worst_samples", []):
        question = str(row.get("question", "")).replace("|", "\\|")
        if len(question) > 120:
            question = question[:117] + "..."
        score = row.get("overall_score")
        score_text = "" if score is None else f"{float(score):.4f}"
        lines.append(
            f"| {row.get('id', '')} | {question} | {score_text} | {row.get('error_type', '')} |"
        )
    return "\n".join(lines) + "\n"


def write_summary_files(
    rows: Sequence[Mapping[str, Any]],
    *,
    result_path: Path,
    output_dir: Path | None = None,
    dataset: str = "",
    judge_model: str = "",
    provider: str = "TheSparkDaily",
) -> SummaryArtifacts:
    resolved_output_dir = output_dir or LLM_AS_JUDGE_ROOT / "outputs"
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    summary = build_summary(
        rows,
        dataset=dataset,
        judge_model=judge_model,
        provider=provider,
        run_name=result_path.stem,
    )
    json_path = resolved_output_dir / f"ragas_summary_{timestamp}.json"
    markdown_path = resolved_output_dir / f"ragas_summary_{timestamp}.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_summary_markdown(summary), encoding="utf-8")
    return SummaryArtifacts(json_path=json_path, markdown_path=markdown_path)


def build_comparison(paths: Sequence[Path]) -> list[dict[str, Any]]:
    comparison: list[dict[str, Any]] = []
    for path in paths:
        rows = load_result_rows(path)
        summary = build_summary(rows, run_name=path.stem)
        comparison.append(
            {
                "run": path.stem,
                "samples": summary["configuration"]["number_of_samples"],
                **summary["overall_metrics"],
                **summary["legal_judge_metrics"],
            }
        )
    return comparison


def _metric_rows(metrics: Mapping[str, Any]) -> list[str]:
    if not metrics:
        return ["| No metric in this run | |"]
    return [f"| {name} | {float(value):.4f} |" for name, value in metrics.items()]


def _numeric_value(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _mean(values: Sequence[float | None] | Any) -> float:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def _unique_join(values: Any) -> str:
    unique = sorted({str(value) for value in values if value not in (None, "")})
    return " | ".join(unique)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize RAGAS evaluation results.")
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=LLM_AS_JUDGE_ROOT / "outputs")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--judge-model", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.input) > 1:
        comparison = build_comparison(args.input)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / "ragas_run_comparison.json"
        output_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote comparison: {output_path}")

    rows = load_result_rows(args.input[-1])
    artifacts = write_summary_files(
        rows,
        result_path=args.input[-1],
        output_dir=args.output_dir,
        dataset=args.dataset,
        judge_model=args.judge_model,
    )
    print(f"Wrote summary JSON: {artifacts.json_path}")
    print(f"Wrote summary Markdown: {artifacts.markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
