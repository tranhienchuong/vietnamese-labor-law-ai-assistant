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

from vn_labor_law_ai_assistant.corpus_pipeline import (
    build_cleaned_text,
    build_page_records_from_text,
    chunk_sections,
    enrich_chunk,
    infer_document_title,
    select_curated_text_sources,
    slugify_text,
    split_sections,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rechunk cleaned Vietnamese labor-law text into hierarchical clause-level JSONL."
    )
    parser.add_argument(
        "--cleaned-dir",
        type=Path,
        default=REPO_ROOT / "corpus" / "cleaned",
        help="Directory containing cleaned UTF-8 legal text files.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        nargs="*",
        default=[],
        help="Optional explicit cleaned text files. Defaults to all .txt files in --cleaned-dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "labor_law_rechunked_hierarchical.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--max-chars", type=int, default=1200, help="Maximum characters per chunk.")
    return parser.parse_args()


def build_records(text_paths: list[Path], max_chars: int) -> tuple[list[dict[str, object]], list[str]]:
    selected_paths, warnings = select_curated_text_sources(text_paths)
    records: list[dict[str, object]] = []

    for text_path in selected_paths:
        source_text = text_path.read_text(encoding="utf-8")
        page_records = build_page_records_from_text(source_text)
        cleaned_text = build_cleaned_text(page_records)
        document_title = infer_document_title(cleaned_text, fallback_title=text_path.stem)
        document_id = slugify_text(text_path.stem)
        sections = split_sections(
            page_records=page_records,
            document_id=document_id,
            document_title=document_title,
        )
        chunks = chunk_sections(sections, max_chars=max_chars)

        for chunk in chunks:
            records.append(
                {
                    "document_id": document_id,
                    "document_title": document_title,
                    "source_kind": "curated_text",
                    "source_path": str(text_path.resolve().as_posix()),
                    **enrich_chunk(
                        chunk=chunk,
                        document_title=document_title,
                        source_kind="curated_text",
                    ),
                }
            )

    return records, warnings


def main() -> None:
    args = parse_args()
    text_paths = [path.resolve() for path in args.source]
    if not text_paths:
        text_paths = sorted(args.cleaned_dir.resolve().glob("*.txt"))
    if not text_paths:
        raise SystemExit("No cleaned text files were found to rechunk.")

    records, warnings = build_records(text_paths, max_chars=args.max_chars)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} chunk(s) to {args.output.resolve().as_posix()}")
    for warning in warnings:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
