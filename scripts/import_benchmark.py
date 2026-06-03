from __future__ import annotations

import argparse
from pathlib import Path
import sys

from vn_labor_law_ai_assistant.evaluation import (
    BENCHMARK_JSONL_NAME,
    WORKBOOK_SHEET_NAME,
    load_benchmark_workbook,
    summarize_benchmark_cases,
    write_benchmark_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import the golden benchmark workbook into repo-native JSONL."
    )
    parser.add_argument(
        "workbook_path",
        type=Path,
        help="Path to the benchmark .xlsx workbook.",
    )
    parser.add_argument(
        "--sheet-name",
        default=WORKBOOK_SHEET_NAME,
        help=f"Worksheet to import (default: {WORKBOOK_SHEET_NAME}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("archive/legacy/eval/data") / BENCHMARK_JSONL_NAME,
        help=(
            "Legacy output JSONL path "
            f"(default: archive/legacy/eval/data/{BENCHMARK_JSONL_NAME})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    cases = load_benchmark_workbook(args.workbook_path, sheet_name=args.sheet_name)
    write_benchmark_jsonl(cases, args.output)
    summary = summarize_benchmark_cases(cases)

    print(f"Imported benchmark cases: {summary['case_count']}")
    print(f"Output JSONL: {args.output.resolve()}")
    print(f"Difficulties: {summary['difficulty_distribution']}")
    print(f"Categories: {summary['category_distribution']}")


if __name__ == "__main__":
    main()
