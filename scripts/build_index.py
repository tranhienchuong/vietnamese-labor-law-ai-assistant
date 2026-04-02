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

from vn_labor_law_ai_assistant.indexing import build_hybrid_index, resolve_chunk_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dense+sparse hybrid index artifacts for legal retrieval.")
    parser.add_argument("--chunks-dir", type=Path, default=REPO_ROOT / "corpus" / "chunks")
    parser.add_argument(
        "--chunk-file",
        type=Path,
        nargs="*",
        default=[],
        help="Optional explicit chunk JSONL files. Defaults to all JSONL files in --chunks-dir.",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=REPO_ROOT / "artifacts" / "index")
    parser.add_argument(
        "--dense-model",
        type=str,
        default="keepitreal/vietnamese-sbert",
        help="Sentence-transformers model used for dense embeddings.",
    )
    parser.add_argument("--collection-name", type=str, default="labor_law_hybrid")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None, help="Optional torch device override, e.g. cpu or cuda.")
    parser.add_argument("--build-id", type=str, default=None, help="Optional explicit build id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunk_paths = resolve_chunk_paths(args.chunks_dir, args.chunk_file)

    if not chunk_paths:
        raise SystemExit("No chunk JSONL files were found to index.")

    manifest = build_hybrid_index(
        chunk_paths=chunk_paths,
        artifacts_dir=args.artifacts_dir,
        dense_model_name=args.dense_model,
        collection_name=args.collection_name,
        batch_size=args.batch_size,
        device=args.device,
        build_id=args.build_id,
    )

    print(f"Hybrid index build complete: {manifest['build_id']}")
    print(f"Collection: {manifest['collection_name']}")
    print(f"Records: {manifest['record_count']}")
    print(f"Dense model: {manifest['dense_model_name']}")
    print(f"Build directory: {manifest['build_dir']}")
    print(f"Current pointer: {(args.artifacts_dir / 'current.json').resolve().as_posix()}")


if __name__ == "__main__":
    main()
