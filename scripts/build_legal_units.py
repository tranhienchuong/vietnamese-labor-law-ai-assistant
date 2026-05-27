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

from vn_labor_law_ai_assistant.legal_unit_parser import (  # noqa: E402
    build_legal_units_from_directory,
    write_legal_unit_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse curated Vietnamese legal texts into structured legal units."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "corpus" / "data" / "curated",
        help="Directory containing curated legal text files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "legal_units",
        help="Directory for legal_units.jsonl and summary artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents, units = build_legal_units_from_directory(args.input_dir)
    jsonl_path, summary_json_path, summary_md_path = write_legal_unit_artifacts(
        documents,
        units,
        args.output_dir,
        input_dir=args.input_dir,
    )

    print(f"Parsed {len(documents)} curated legal document(s).")
    print(f"Wrote {len(units)} legal unit records.")
    print(f"JSONL units: {jsonl_path}")
    print(f"JSON summary: {summary_json_path}")
    print(f"Markdown summary: {summary_md_path}")


if __name__ == "__main__":
    main()
