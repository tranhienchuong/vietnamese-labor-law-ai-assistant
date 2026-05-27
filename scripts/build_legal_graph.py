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
from vn_labor_law_ai_assistant.rag.graph.builder import EXPECTED_DOCUMENT_NORMATIVE_RANKS
from vn_labor_law_ai_assistant.rag.retrieval.manifest import load_manifest


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Neo4j legal knowledge graph.")
    parser.add_argument("--index-path", type=Path, default=REPO_ROOT / "artifacts" / "index")
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=REPO_ROOT / "artifacts" / "chunks" / "legal_chunks_enriched.jsonl",
        help="Enriched legal chunks JSONL used as the graph source of truth.",
    )
    parser.add_argument(
        "--reference-edges-path",
        type=Path,
        default=REPO_ROOT / "artifacts" / "graph" / "reference_edges.jsonl",
        help="Reference edge artifact JSONL. Only resolved edges are loaded.",
    )
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
        help="Load resolved cross-reference links from --reference-edges-path.",
    )
    parser.add_argument(
        "--with-normative-hierarchy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable document-level normative hierarchy links.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=REPO_ROOT / "artifacts" / "graph" / "legal_graph_build_summary.json",
    )
    return parser.parse_args()


def manifest_path_for_index(index_path: Path) -> Path:
    return index_path / "current.json" if index_path.is_dir() else index_path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_metadata(index_path: Path, chunks_path: Path, reference_edges_path: Path) -> dict[str, object]:
    manifest_path = manifest_path_for_index(index_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest = load_manifest(index_path)
    validation = manifest.get("validation") if isinstance(manifest.get("validation"), dict) else {}
    return {
        "index_path": str(index_path),
        "manifest_path": str(manifest_path),
        "manifest_hash": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_build_id": str(manifest.get("build_id") or ""),
        "manifest_record_count": int(manifest.get("record_count") or 0),
        "manifest_chunk_count": int(manifest.get("chunk_count") or validation.get("chunk_count") or 0),
        "manifest_document_count": int(manifest.get("document_count") or 0),
        "manifest_validation_passed": bool(validation.get("passed")),
        "indexed_chunk_count": int(validation.get("indexed_chunk_count") or 0),
        "all_chunks_indexed": bool(validation.get("all_chunks_indexed")),
        "vector_dimension": manifest.get("vector_dimension")
        or (validation.get("dense_vector_dimensions") or [None])[0],
        "chunks_path": str(chunks_path),
        "chunks_hash": file_sha256(chunks_path),
        "reference_edges_path": str(reference_edges_path),
        "reference_edges_hash": file_sha256(reference_edges_path),
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def render_summary_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Legal Graph Build Summary",
        "",
        f"- Documents: {summary.get('documents')}",
        f"- Articles: {summary.get('articles')}",
        f"- Clauses: {summary.get('clauses')}",
        f"- Points: {summary.get('points')}",
        f"- Appendices: {summary.get('appendices')}",
        f"- Evidence chunks: {summary.get('evidence_chunks')}",
        f"- Topic nodes: {summary.get('topic_nodes')}",
        f"- Actor nodes: {summary.get('actor_nodes')}",
        f"- Issue type nodes: {summary.get('issue_type_nodes')}",
        f"- Reference edges: {summary.get('reference_edges')}",
        f"- DETAILS edges: {summary.get('details_edges')}",
        f"- GUIDED_BY edges: {summary.get('guided_by_edges')}",
        f"- Taxonomy edges: {summary.get('taxonomy_edges')}",
        f"- Normative hierarchy edges: {summary.get('normative_hierarchy_edges')}",
        f"- Unresolved edges skipped: {summary.get('unresolved_edges_skipped')}",
        f"- Duplicate edges skipped: {summary.get('duplicate_edges_skipped')}",
        "",
        "## Validation",
        "",
    ]
    validation = summary.get("validation") or {}
    if isinstance(validation, dict):
        for key, value in validation.items():
            lines.append(f"- {key}: {value}")
    neo4j_validation = summary.get("neo4j_validation") or {}
    if isinstance(neo4j_validation, dict):
        lines.extend(["", "## Neo4j Validation", ""])
        for key, value in neo4j_validation.items():
            lines.append(f"- {key}: {value}")
    warnings = summary.get("warnings") or []
    lines.extend(["", "## Warnings", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.chunks_path)
    reference_edges = read_jsonl(args.reference_edges_path) if args.with_references else []
    include_concepts = bool(args.with_concepts)
    include_references = bool(args.with_references)
    include_normative_hierarchy = bool(args.with_normative_hierarchy)
    if args.structural_only:
        include_concepts = False
        include_references = False
        include_normative_hierarchy = False

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
            with_normative_hierarchy=include_normative_hierarchy,
            reference_edges=reference_edges,
        ).build_and_upsert(
            records,
            store,
            build_metadata=build_metadata(args.index_path, args.chunks_path, args.reference_edges_path),
        )
        result.summary["neo4j_validation"] = store.validate_loaded_graph(
            expected_chunk_count=len(records),
            expected_document_count=6,
            expected_normative_ranks=EXPECTED_DOCUMENT_NORMATIVE_RANKS,
        )
    finally:
        store.close()

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md_path = args.summary_path.with_suffix(".md")
    summary_md_path.write_text(render_summary_markdown(result.summary), encoding="utf-8")

    for key in (
        "documents",
        "articles",
        "clauses",
        "points",
        "appendices",
        "evidence_chunks",
        "topic_nodes",
        "actor_nodes",
        "issue_type_nodes",
        "reference_edges",
        "details_edges",
        "guided_by_edges",
        "taxonomy_edges",
        "normative_hierarchy_edges",
        "unresolved_edges_skipped",
    ):
        print(f"{key}: {result.summary.get(key, 0)}")
    print(f"summary_path: {args.summary_path.resolve()}")
    print(f"summary_markdown_path: {summary_md_path.resolve()}")


if __name__ == "__main__":
    main()
