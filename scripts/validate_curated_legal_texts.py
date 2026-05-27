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

from vn_labor_law_ai_assistant.curated_text_validation import (  # noqa: E402
    validate_curated_directory,
    write_validation_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate curated UTF-8 legal text files before chunking."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "corpus" / "data" / "curated",
        help="Directory containing curated .txt legal documents.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "validation",
        help="Directory for validation JSON and Markdown reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_curated_directory(args.input_dir)
    json_path, markdown_path = write_validation_artifacts(report, args.output_dir)

    print(f"Validated {report['document_count']} curated text file(s).")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    if report.get("warnings"):
        print("Run warnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
