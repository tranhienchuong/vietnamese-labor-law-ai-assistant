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

from vn_labor_law_ai_assistant.corpus_pipeline import (  # noqa: E402
    build_curated_chunk_records,
    resolve_curated_legal_chunk_paths,
    write_legal_chunk_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build hierarchy-aware chunks from curated legal text files only."
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
        default=REPO_ROOT / "artifacts" / "chunks",
        help="Directory for legal_chunks.jsonl and summary artifacts.",
    )
    parser.add_argument("--max-chars", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text_paths = resolve_curated_legal_chunk_paths(args.input_dir)
    chunks, warnings = build_curated_chunk_records(text_paths, max_chars=args.max_chars)
    chunks_path, summary_json_path, summary_md_path = write_legal_chunk_artifacts(
        chunks,
        args.output_dir,
    )

    print(f"Built {len(chunks)} legal chunk(s).")
    print(f"Chunks JSONL: {chunks_path}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary Markdown: {summary_md_path}")
    for warning in warnings:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
