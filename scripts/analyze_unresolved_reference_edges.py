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

from vn_labor_law_ai_assistant.legal_reference_edges import read_jsonl  # noqa: E402
from vn_labor_law_ai_assistant.unresolved_reference_report import (  # noqa: E402
    build_unresolved_reference_report,
    write_unresolved_reference_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze unresolved legal reference edges before graph import."
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=REPO_ROOT / "artifacts" / "graph" / "reference_edges.jsonl",
        help="Reference edge JSONL path.",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=REPO_ROOT / "artifacts" / "chunks" / "legal_chunks_enriched.jsonl",
        help="Enriched legal chunk JSONL path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "graph",
        help="Directory for unresolved reference report artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path, markdown_path = write_unresolved_reference_report(
        edges_path=args.edges,
        chunks_path=args.chunks,
        output_dir=args.output_dir,
    )
    report = build_unresolved_reference_report(read_jsonl(args.edges), read_jsonl(args.chunks))

    print(f"Unresolved edges: {report['total_unresolved_edges']}")
    print(f"Critical unresolved references: {report['critical_unresolved_references']}")
    print(f"Report JSON: {json_path}")
    print(f"Report Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
