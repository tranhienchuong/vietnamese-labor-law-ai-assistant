from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from vn_labor_law_ai_assistant.retriever import format_intent_summary, route_query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect retrieval or answer failures from benchmark result files."
    )
    parser.add_argument(
        "results_path",
        type=Path,
        help="Path to benchmark results in CSV or JSONL format.",
    )
    parser.add_argument(
        "--failure-type",
        choices=("retrieval_miss", "hallucination", "abstention_mismatch"),
        default="retrieval_miss",
        help="Which failure slice to inspect.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of failures to print.",
    )
    return parser.parse_args()


def load_rows(results_path: Path) -> list[dict[str, str]]:
    suffix = results_path.suffix.lower()
    if suffix == ".csv":
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        rows: list[dict[str, str]] = []
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append({str(key): str(value) for key, value in payload.items()})
        return rows
    raise ValueError(f"Unsupported results format: {results_path.suffix}")


def retrieval_hit_value(row: dict[str, str]) -> str:
    hit_columns = sorted(
        column
        for column in row
        if column.startswith("retrieval_hit_at_")
    )
    if not hit_columns:
        return ""
    return row.get(hit_columns[0], "")


def is_failure(row: dict[str, str], failure_type: str) -> bool:
    if failure_type == "retrieval_miss":
        return retrieval_hit_value(row).strip().lower() == "no"
    if failure_type == "hallucination":
        return row.get("hallucination_flag", "").strip().lower() == "yes"
    return row.get("abstention_correct", "").strip().lower() == "no"


def print_row(index: int, row: dict[str, str]) -> None:
    question = row.get("question", "").strip()
    intent = route_query(question)
    print(f"[{index}] {row.get('id', '').strip()} | {row.get('model_version', '').strip()}")
    print(f"Question: {question}")
    print(f"Route: {format_intent_summary(intent)}")
    print(f"Expected: {row.get('expected_citations_in_scope', '').strip()}")
    print(f"Retrieved: {row.get('retrieved_citations', '').strip()}")
    generated_answer = row.get("generated_answer", "").strip()
    if generated_answer:
        print(f"Generated: {generated_answer}")
    comments = row.get("comments", "").strip()
    if comments:
        print(f"Comments: {comments}")
    print("")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    rows = load_rows(args.results_path)
    failures = [row for row in rows if is_failure(row, args.failure_type)]

    print(f"Results file: {args.results_path.resolve()}")
    print(f"Failure type: {args.failure_type}")
    print(f"Matched rows: {len(failures)} / {len(rows)}")
    print("")

    for index, row in enumerate(failures[: args.limit], start=1):
        print_row(index, row)


if __name__ == "__main__":
    main()
