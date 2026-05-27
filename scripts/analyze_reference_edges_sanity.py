from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vn_labor_law_ai_assistant.reference_edges_sanity import (  # noqa: E402
    write_reference_edges_sanity_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run final sanity checks on legal reference edge artifacts."
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
        help="Enriched legal chunks JSONL path used for source-text context.",
    )
    parser.add_argument(
        "--unresolved-report",
        type=Path,
        default=REPO_ROOT / "artifacts" / "graph" / "unresolved_reference_edges_report.json",
        help="Unresolved reference report JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "graph",
        help="Directory for sanity report artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path, markdown_path = write_reference_edges_sanity_report(
        edges_path=args.edges,
        chunks_path=args.chunks,
        unresolved_report_path=args.unresolved_report,
        output_dir=args.output_dir,
    )
    report = json.loads(json_path.read_text(encoding="utf-8"))

    print(f"Suspicious resolved edges: {report['suspicious_resolved_edges_count']}")
    print(f"Suspicious unresolved edges: {report['suspicious_unresolved_edges_count']}")
    print(f"Dual-target duplicates: {report['dual_target_duplicate_count']}")
    print(f"Critical unresolved references: {report['critical_unresolved_references']}")
    print(
        "External classification issues: "
        f"{report['external_reference_classification_issues_count']}"
    )
    print(f"Passed: {report['passed']}")
    print(f"Report JSON: {json_path}")
    print(f"Report Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
