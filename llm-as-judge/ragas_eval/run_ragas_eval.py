from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import LLM_AS_JUDGE_ROOT, load_thesparkdaily_config, resolve_output_path
from .dataset_loader import BenchmarkSample, read_benchmark_records, validate_benchmark_schema
from .embeddings import build_ragas_embeddings
from .legal_judge_prompt import (
    LEGAL_JUDGE_SCORE_FIELDS,
    build_legal_judge_messages,
    parse_legal_judge_response,
)
from .metrics import metric_output_name, metrics_require_embeddings, resolve_ragas_metrics
from .summarize_results import write_summary_files
from .thesparkdaily_llm import build_chat_openai, build_judge_llm, invoke_chat_model


RAGAS_OUTPUT_COLUMNS = (
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "factual_correctness",
    "context_precision",
    "context_recall",
)
LEGAL_OUTPUT_COLUMNS = (
    *LEGAL_JUDGE_SCORE_FIELDS,
    "error_type",
    "explanation",
)
BASE_OUTPUT_COLUMNS = (
    "id",
    "user_input",
    *RAGAS_OUTPUT_COLUMNS,
    *LEGAL_OUTPUT_COLUMNS,
    "overall_avg",
    "judge_model",
    "ragas_judge_model",
    "legal_judge_model",
    "mode",
    "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation with TheSparkDaily judge LLM.")
    parser.add_argument("--input", required=True, type=Path, help="Benchmark .jsonl/.json/.csv path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .csv, .json, or .jsonl. Defaults to llm-as-judge/outputs.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N samples.")
    parser.add_argument(
        "--judge-model",
        default="",
        help="Accurate judge model. Defaults to THESPARKDAILY_JUDGE_MODEL.",
    )
    parser.add_argument(
        "--fast-judge-model",
        default="",
        help="Fast judge model. Defaults to THESPARKDAILY_FAST_JUDGE_MODEL.",
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "accurate", "full"),
        default="full",
        help="fast=cheap RAGAS subset, accurate=legal judge, full=RAGAS + legal judge.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate evaluation readiness without importing ragas or calling the judge LLM.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip summary JSON/Markdown generation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_thesparkdaily_config()
    output_path = resolve_output_path(args.output)
    needs_ragas = args.mode in {"fast", "full"}

    all_records = read_benchmark_records(args.input)
    records = all_records[: args.limit] if args.limit > 0 else all_records
    report = validate_benchmark_schema(
        records,
        require_response=True,
        require_retrieved_contexts=needs_ragas,
    )
    samples = list(report.valid_samples)

    print(f"Total source samples: {len(all_records)}")
    print(f"Selected samples: {report.total_samples}")
    print(f"Evaluation-ready samples: {len(samples)}")
    print(f"Schema errors: {report.error_count}")
    for error in report.errors:
        print(f"- {error.format()}")
    if report.errors:
        return 1
    if args.dry_run:
        print("Dry run complete: schema is evaluation-ready and no judge call was made.")
        return 0

    config.require_api_key()
    rows = run_evaluation(
        samples,
        mode=args.mode,
        judge_model=args.judge_model or config.judge_model,
        fast_judge_model=args.fast_judge_model or config.fast_judge_model,
    )
    write_rows(rows, output_path)
    print(f"Wrote results: {output_path}")

    if not args.no_summary:
        legal_model = args.judge_model or config.judge_model
        artifacts = write_summary_files(
            rows,
            result_path=output_path,
            output_dir=LLM_AS_JUDGE_ROOT / "outputs",
            dataset=str(args.input),
            judge_model=legal_model,
        )
        print(f"Wrote summary JSON: {artifacts.json_path}")
        print(f"Wrote summary Markdown: {artifacts.markdown_path}")
    return 0


def run_evaluation(
    samples: Sequence[BenchmarkSample],
    *,
    mode: str,
    judge_model: str,
    fast_judge_model: str,
) -> list[dict[str, Any]]:
    rows = [_empty_output_row(sample, mode=mode, judge_model=judge_model) for sample in samples]

    if mode in {"fast", "full"}:
        ragas_model = fast_judge_model
        ragas_rows = run_ragas_metrics(samples, mode=mode, judge_model=ragas_model)
        for row, ragas_row in zip(rows, ragas_rows, strict=False):
            row.update(ragas_row)
            row["ragas_judge_model"] = ragas_model
            if mode == "fast":
                row["judge_model"] = ragas_model

    if mode in {"accurate", "full"}:
        legal_rows = run_legal_judge(samples, judge_model=judge_model)
        for row, legal_row in zip(rows, legal_rows, strict=False):
            row.update(legal_row)
            row["judge_model"] = judge_model
            row["legal_judge_model"] = judge_model

    for row in rows:
        row["overall_avg"] = _overall_average(row)
    return rows


def run_ragas_metrics(
    samples: Sequence[BenchmarkSample],
    *,
    mode: str,
    judge_model: str,
) -> list[dict[str, Any]]:
    from ragas import evaluate

    resolved_metrics = resolve_ragas_metrics(mode)
    metrics = [resolved.metric for resolved in resolved_metrics]
    llm = build_judge_llm(model=judge_model)
    embeddings = build_ragas_embeddings() if metrics_require_embeddings(resolved_metrics) else None
    dataset = _build_ragas_dataset(samples)
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        show_progress=True,
    )
    result_rows = _ragas_result_rows(result)
    output_rows: list[dict[str, Any]] = []
    expected_names = {
        metric_output_name(resolved): resolved.name for resolved in resolved_metrics
    }
    for index, sample in enumerate(samples):
        raw_row = result_rows[index] if index < len(result_rows) else {}
        output = {column: "" for column in RAGAS_OUTPUT_COLUMNS}
        for raw_name, stable_name in expected_names.items():
            value = raw_row.get(raw_name, raw_row.get(stable_name, ""))
            if stable_name == "answer_correctness" and raw_name == "factual_correctness":
                output["factual_correctness"] = _clean_metric_value(value)
            else:
                output[stable_name] = _clean_metric_value(value)
        output_rows.append(output)
    return output_rows


def run_legal_judge(
    samples: Sequence[BenchmarkSample],
    *,
    judge_model: str,
) -> list[dict[str, Any]]:
    chat_model = build_chat_openai(model=judge_model, json_mode=True)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        row = {column: "" for column in LEGAL_OUTPUT_COLUMNS}
        try:
            messages = build_legal_judge_messages(sample.to_legal_judge_payload())
            content = invoke_chat_model(chat_model, messages)
            score = parse_legal_judge_response(content)
            row.update(score.to_row())
        except Exception as exc:
            row["error"] = f"legal_judge_failed: {exc}"
        rows.append(row)
    return rows


def write_rows(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(BASE_OUTPUT_COLUMNS))
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in BASE_OUTPUT_COLUMNS})
        return
    if suffix == ".jsonl":
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        return
    if suffix == ".json":
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "rows": list(rows),
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    raise ValueError("Unsupported output format. Expected .csv, .json, or .jsonl.")


def _build_ragas_dataset(samples: Sequence[BenchmarkSample]) -> Any:
    records = [sample.to_ragas_record() for sample in samples]
    try:
        from ragas import EvaluationDataset
    except ImportError:
        from datasets import Dataset

        legacy_records = [
            {
                "question": sample.user_input,
                "answer": sample.response,
                "contexts": list(sample.retrieved_contexts),
                "ground_truth": sample.reference,
            }
            for sample in samples
        ]
        return Dataset.from_list(legacy_records)
    return EvaluationDataset.from_list(records)


def _ragas_result_rows(result: Any) -> list[dict[str, Any]]:
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        return [dict(row) for row in frame.to_dict(orient="records")]
    if hasattr(result, "scores"):
        scores = result.scores
        if isinstance(scores, list):
            return [dict(row) for row in scores]
    if isinstance(result, list):
        return [dict(row) for row in result]
    if isinstance(result, dict):
        return [dict(result)]
    return []


def _empty_output_row(sample: BenchmarkSample, *, mode: str, judge_model: str) -> dict[str, Any]:
    row = {column: "" for column in BASE_OUTPUT_COLUMNS}
    row.update(sample.to_output_base())
    row["judge_model"] = judge_model
    row["legal_judge_model"] = judge_model if mode in {"accurate", "full"} else ""
    row["mode"] = mode
    return row


def _clean_metric_value(value: Any) -> Any:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(number):
        return ""
    return number


def _overall_average(row: Mapping[str, Any]) -> float | str:
    metric_values: list[float] = []
    for column in (*RAGAS_OUTPUT_COLUMNS, *LEGAL_JUDGE_SCORE_FIELDS):
        value = _clean_metric_value(row.get(column))
        if value == "":
            continue
        try:
            metric_values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not metric_values:
        return ""
    return round(sum(metric_values) / len(metric_values), 6)


if __name__ == "__main__":
    raise SystemExit(main())
