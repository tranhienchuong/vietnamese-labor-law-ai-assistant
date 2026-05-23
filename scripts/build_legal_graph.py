from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vn_labor_law_ai_assistant.rag.graph import LegalGraphBuilder, Neo4jLegalGraphStore
from vn_labor_law_ai_assistant.rag.graph.structural_parser import load_index_records
from vn_labor_law_ai_assistant.rag.retrieval.manifest import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Neo4j legal knowledge graph.")
    parser.add_argument("--index-path", type=Path, default=REPO_ROOT / "artifacts" / "index")
    parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", type=str, default="neo4j")
    parser.add_argument("--neo4j-password", type=str, default="password")
    parser.add_argument("--neo4j-database", type=str, default="neo4j")
    parser.add_argument("--reset", action="store_true", help="Delete existing LegalNode graph before rebuild.")
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="Build only document/article/clause/point/chunk structure.",
    )
    parser.add_argument(
        "--with-concepts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable dictionary/rule concept links.",
    )
    parser.add_argument(
        "--with-references",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable regex cross-reference links.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=REPO_ROOT / "artifacts" / "graph" / "legal_graph_build_summary.json",
    )
    return parser.parse_args()


def manifest_path_for_index(index_path: Path) -> Path:
    return index_path / "current.json" if index_path.is_dir() else index_path


def build_metadata(index_path: Path) -> dict[str, object]:
    manifest_path = manifest_path_for_index(index_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest = load_manifest(index_path)
    return {
        "index_path": str(index_path),
        "manifest_path": str(manifest_path),
        "manifest_hash": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_build_id": str(manifest.get("build_id") or ""),
        "manifest_record_count": int(manifest.get("record_count") or 0),
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    args = parse_args()
    records = load_index_records(args.index_path)
    include_concepts = bool(args.with_concepts)
    include_references = bool(args.with_references)
    if args.structural_only:
        include_concepts = False
        include_references = False

    store = Neo4jLegalGraphStore(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
        database=args.neo4j_database,
    )
    try:
        store.setup_schema()
        if args.reset:
            store.reset_graph()
        result = LegalGraphBuilder(
            with_concepts=include_concepts,
            with_references=include_references,
        ).build_and_upsert(
            records,
            store,
            build_metadata=build_metadata(args.index_path),
        )
    finally:
        store.close()

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for key in (
        "documents",
        "articles",
        "clauses",
        "points",
        "evidence_chunks",
        "edges",
        "concept_nodes",
        "reference_edges",
    ):
        print(f"{key}: {result.summary.get(key, 0)}")
    print(f"summary_path: {args.summary_path.resolve()}")


if __name__ == "__main__":
    main()
