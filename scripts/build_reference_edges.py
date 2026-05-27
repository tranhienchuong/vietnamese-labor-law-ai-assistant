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

from vn_labor_law_ai_assistant.legal_reference_edges import (  # noqa: E402
    build_reference_edge_records,
    read_jsonl,
    summarize_reference_edges,
    write_reference_edge_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build legal cross-reference edge artifacts from enriched legal chunks."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "artifacts" / "chunks" / "legal_chunks_enriched.jsonl",
        help="Input enriched legal chunks JSONL path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "graph",
        help="Directory for reference edge artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = read_jsonl(args.input)
    edges, duplicate_edges_removed = build_reference_edge_records(chunks)
    edges_path, summary_json_path, summary_md_path = write_reference_edge_artifacts(
        edges,
        duplicate_edges_removed,
        args.output_dir,
    )
    summary = summarize_reference_edges(edges, duplicate_edges_removed)

    print(f"Built {summary['total_reference_edges']} reference edge(s).")
    print(f"Resolved edges: {summary['resolved_edges']}")
    print(f"Unresolved edges: {summary['unresolved_edges']}")
    print(f"Duplicate edges removed: {summary['duplicate_edges_removed']}")
    print(f"Edges JSONL: {edges_path}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary Markdown: {summary_md_path}")


if __name__ == "__main__":
    main()
