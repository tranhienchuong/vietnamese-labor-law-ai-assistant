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

from vn_labor_law_ai_assistant.legal_chunk_enrichment import (  # noqa: E402
    enrich_legal_chunks,
    read_jsonl,
    summarize_enriched_chunks,
    write_enriched_chunk_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich validated hierarchy-aware legal chunks with metadata and taxonomy."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "artifacts" / "chunks" / "legal_chunks.jsonl",
        help="Input legal chunks JSONL path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "chunks",
        help="Directory for enriched JSONL and summary artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = read_jsonl(args.input)
    enriched_chunks = enrich_legal_chunks(chunks)
    chunks_path, summary_json_path, summary_md_path = write_enriched_chunk_artifacts(
        enriched_chunks,
        args.output_dir,
    )
    summary = summarize_enriched_chunks(enriched_chunks)

    print(f"Enriched {summary['chunk_count']} legal chunk(s).")
    print(f"Chunks JSONL: {chunks_path}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary Markdown: {summary_md_path}")
    print(f"Chunks missing topic: {len(summary['chunks_missing_topic'])}")
    print(f"Chunks missing actor: {len(summary['chunks_missing_actor'])}")
    print(f"Chunks missing issue_type: {len(summary['chunks_missing_issue_type'])}")


if __name__ == "__main__":
    main()
