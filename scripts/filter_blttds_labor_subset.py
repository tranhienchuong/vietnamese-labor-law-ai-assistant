from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vn_labor_law_ai_assistant.blttds_labor_filter import (  # noqa: E402
    filter_blttds_labor_subset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the labor-related BLTTDS 2015 subset for later RAG steps."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT
        / "corpus"
        / "data"
        / "curated"
        / "bo_luat_92_2015_qh13_to_tung_dan_su_2015_clean.txt",
        help="Curated full Bộ luật Tố tụng dân sự 2015 text file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "corpus" / "data" / "curated" / "92_2015_QH13_labor_only.txt",
        help="Filtered labor-only output text file.",
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=REPO_ROOT / "artifacts" / "validation" / "curated_text_validation.json",
        help="Optional curated validation report used only for suggestion metadata.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "validation",
        help="Directory for JSON and Markdown filter reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = filter_blttds_labor_subset(
        args.input,
        args.output,
        args.report_dir,
        validation_report_path=args.validation_report,
    )

    print(f"Strict articles parsed: {report['total_article_count']}")
    print(f"Kept labor-related articles: {report['kept_article_count']}")
    print(f"Removed articles: {report['removed_article_count']}")
    print(f"Filtered text: {args.output}")
    print(f"JSON report: {report['json_report_path']}")
    print(f"Markdown report: {report['markdown_report_path']}")


if __name__ == "__main__":
    main()
