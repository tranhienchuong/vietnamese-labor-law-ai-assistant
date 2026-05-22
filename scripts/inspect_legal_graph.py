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

from vn_labor_law_ai_assistant.rag.graph import Neo4jLegalGraphStore
from vn_labor_law_ai_assistant.rag.graph.ontology import GRAPH_EXPANSION_EDGE_TYPES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the Neo4j legal graph.")
    parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", type=str, default="neo4j")
    parser.add_argument("--neo4j-password", type=str, default="password")
    parser.add_argument("--neo4j-database", type=str, default="neo4j")
    parser.add_argument("--chunk-id", type=str, default="")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--limit", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = Neo4jLegalGraphStore(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
        database=args.neo4j_database,
    )
    try:
        if args.chunk_id:
            result = store.expand_from_chunk_ids(
                (args.chunk_id,),
                depth=args.depth,
                limit=args.limit,
                min_confidence=0.0,
                edge_types=GRAPH_EXPANSION_EDGE_TYPES,
            )
            print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
            return

        with store._session() as session:
            node_counts = session.run(
                """
                MATCH (n:LegalNode)
                RETURN n.node_type AS node_type, count(*) AS count
                ORDER BY node_type
                """
            )
            edge_counts = session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) AS edge_type, count(*) AS count
                ORDER BY edge_type
                """
            )
            print("Node counts:")
            for record in node_counts:
                print(f"- {record['node_type']}: {record['count']}")
            print("Edge counts:")
            for record in edge_counts:
                print(f"- {record['edge_type']}: {record['count']}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
