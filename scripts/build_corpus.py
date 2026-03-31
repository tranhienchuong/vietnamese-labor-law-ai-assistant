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

from vn_labor_law_ai_assistant.corpus_pipeline import build_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaned legal corpus and chunk files from raw PDFs.")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "corpus" / "raw")
    parser.add_argument("--cleaned-dir", type=Path, default=REPO_ROOT / "corpus" / "cleaned")
    parser.add_argument("--chunks-dir", type=Path, default=REPO_ROOT / "corpus" / "chunks")
    parser.add_argument("--metadata-dir", type=Path, default=REPO_ROOT / "corpus" / "metadata")
    parser.add_argument(
        "--curated-text",
        type=Path,
        nargs="*",
        default=[],
        help="Optional UTF-8 cleaned text files to include as curated sources.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_corpus(
        raw_dir=args.raw_dir,
        cleaned_dir=args.cleaned_dir,
        chunks_dir=args.chunks_dir,
        metadata_dir=args.metadata_dir,
        curated_text_paths=args.curated_text,
    )

    print(f"Processed {manifest['document_count']} document(s).")
    print(f"Ready for indexing: {manifest['ready_documents']}")
    print(f"Needs OCR: {manifest['needs_ocr_documents']}")

    for doc in manifest["documents"]:
        print(
            f"- {doc['document_title']}: {doc['status']} "
            f"(pages={doc['page_count']}, text_pages={doc['text_page_count']}, chunks={doc['chunk_count']})"
        )


if __name__ == "__main__":
    main()
