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

from vn_labor_law_ai_assistant.rag.graph import LegalGraphBuilder, Neo4jLegalGraphStore
from vn_labor_law_ai_assistant.rag.graph.structural_parser import load_index_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Neo4j legal knowledge graph.")
    parser.add_argument("--index-path", type=Path, default=REPO_ROOT / "artifacts" / "index")
    parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", type=str, default="neo4j")
    parser.add_argument("--neo4j-password", type=str, default="password")
    parser.add_argument("--neo4j-database", type=str, default="neo4j")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=REPO_ROOT / "artifacts" / "graph" / "legal_graph_build_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_index_records(args.index_path)
    store = Neo4jLegalGraphStore(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
        database=args.neo4j_database,
    )
    try:
        store.setup_schema()
        result = LegalGraphBuilder().build_and_upsert(records, store)
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
