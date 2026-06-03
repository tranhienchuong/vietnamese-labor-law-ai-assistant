from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vn_labor_law_ai_assistant.corpus_pipeline import normalize_for_matching
from vn_labor_law_ai_assistant.rag.retrieval import HybridRetriever


ARTICLE_RE = re.compile(r"dieu\s+(?P<article>\d+[a-z]?)", re.IGNORECASE)
CLAUSE_RE = re.compile(r"khoan\s+(?P<clause>\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exploratory hybrid vs Neo4j graph retrieval check over the official benchmark."
    )
    parser.add_argument(
        "--golden-path",
        type=Path,
        default=REPO_ROOT / "artifacts" / "evaluation" / "golden_benchmark_100_extended.jsonl",
    )
    parser.add_argument("--index-path", type=Path, default=REPO_ROOT / "artifacts" / "index")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reranker-model", type=str, default=os.getenv("RERANKER_MODEL", ""))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "evaluation")
    return parser.parse_args()


def load_cases(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cases.append(json.loads(line))
        if limit and len(cases) >= limit:
            break
    return cases


def normalized_citations(case: dict[str, Any]) -> tuple[str, ...]:
    citations = list(case.get("gold_citations") or [])
    for spec in case.get("required_citations") or []:
        if isinstance(spec, dict) and spec.get("label"):
            citations.append(str(spec["label"]))
    for key in ("gold_citation_primary", "gold_citation_secondary"):
        if case.get(key):
            citations.append(str(case[key]))
    return tuple(
        normalize_for_matching(citation)
        for citation in citations
        if normalize_for_matching(str(citation))
    )


def extract_articles(texts: Iterable[str]) -> set[str]:
    values: set[str] = set()
    for text in texts:
        values.update(match.group("article").lower() for match in ARTICLE_RE.finditer(text))
    return values


def extract_clauses(texts: Iterable[str]) -> set[str]:
    values: set[str] = set()
    for text in texts:
        values.update(match.group("clause").lower() for match in CLAUSE_RE.finditer(text))
    return values


def is_relevant(citation: str, expected: tuple[str, ...]) -> bool:
    normalized = normalize_for_matching(citation)
    return any(gold in normalized or normalized in gold for gold in expected)


def dcg(relevance: list[int]) -> float:
    return sum(value / math.log2(index + 2) for index, value in enumerate(relevance))


def metrics_for_case(case: dict[str, Any], result: Any, *, top_k: int) -> dict[str, Any]:
    expected = normalized_citations(case)
    hit_citations = [hit.citation_text for hit in result.hits[:top_k]]
    relevance = [1 if is_relevant(citation, expected) else 0 for citation in hit_citations]
    relevant_count = sum(relevance)
    expected_count = max(1, len(expected))
    first_relevant_rank = next((idx + 1 for idx, value in enumerate(relevance) if value), 0)
    ideal = sorted(relevance, reverse=True)

    expected_articles = extract_articles(expected)
    retrieved_articles = extract_articles(hit_citations)
    expected_clauses = extract_clauses(expected)
    retrieved_clauses = extract_clauses(hit_citations)
    graph_hit_count = sum(
        1
        for hit in result.hits[:top_k]
        if hit.payload.get("retrieval_source") in {"graph", "hybrid"}
        or hit.payload.get("retrieval_method") == "neo4j_graph_expansion"
    )
    relevant_graph_hit_count = sum(
        1
        for hit in result.hits[:top_k]
        if (
            hit.payload.get("retrieval_source") in {"graph", "hybrid"}
            or hit.payload.get("retrieval_method") == "neo4j_graph_expansion"
        )
        and is_relevant(hit.citation_text, expected)
    )

    return {
        "id": case.get("id", ""),
        "question": case.get("question") or case.get("query", ""),
        "recall_at_k": min(1.0, relevant_count / expected_count),
        "precision_at_k": relevant_count / max(1, top_k),
        "mrr": (1.0 / first_relevant_rank) if first_relevant_rank else 0.0,
        "ndcg": dcg(relevance) / max(dcg(ideal), 1e-9),
        "article_recall": (
            len(expected_articles & retrieved_articles) / len(expected_articles)
            if expected_articles
            else 0.0
        ),
        "clause_recall": (
            len(expected_clauses & retrieved_clauses) / len(expected_clauses)
            if expected_clauses
            else 0.0
        ),
        "citation_recall": min(1.0, relevant_count / expected_count),
        "multi_hop_evidence_recall": 1.0 if relevant_count >= 2 else 0.0,
        "graph_expansion_utility": relevant_graph_hit_count / max(1, graph_hit_count),
        "graph_expanded_hits": graph_hit_count,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_variant(
    *,
    variant: str,
    cases: list[dict[str, Any]],
    index_path: Path,
    top_k: int,
    env_overrides: dict[str, str],
    reranker_model: str,
) -> list[dict[str, Any]]:
    previous_env = {key: os.environ.get(key) for key in env_overrides}
    os.environ.update(env_overrides)
    rows: list[dict[str, Any]] = []
    retriever = HybridRetriever(index_path=index_path, reranker_model=reranker_model)
    try:
        for case in cases:
            query = str(case.get("question") or case.get("query") or "")
            result = retriever.retrieve(query, top_k=top_k)
            row = metrics_for_case(case, result, top_k=top_k)
            row["variant"] = variant
            rows.append(row)
    finally:
        retriever.close()
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metric_keys = (
        "recall_at_k",
        "precision_at_k",
        "mrr",
        "ndcg",
        "article_recall",
        "clause_recall",
        "citation_recall",
        "multi_hop_evidence_recall",
        "graph_expansion_utility",
    )
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    return {
        variant: {
            key: sum(float(row[key]) for row in variant_rows) / max(1, len(variant_rows))
            for key in metric_keys
        }
        for variant, variant_rows in by_variant.items()
    }


def main() -> None:
    args = parse_args()
    cases = load_cases(args.golden_path, limit=args.limit)
    baseline_rows = run_variant(
        variant="hybrid_baseline",
        cases=cases,
        index_path=args.index_path,
        top_k=args.top_k,
        env_overrides={"LEGAL_GRAPH_ENABLED": "false"},
        reranker_model="",
    )
    semantic_rows = run_variant(
        variant="hybrid_semantic_reranker",
        cases=cases,
        index_path=args.index_path,
        top_k=args.top_k,
        env_overrides={"LEGAL_GRAPH_ENABLED": "false"},
        reranker_model=args.reranker_model,
    )
    graph_rows = run_variant(
        variant="hybrid_neo4j_graph_expansion",
        cases=cases,
        index_path=args.index_path,
        top_k=args.top_k,
        env_overrides={"LEGAL_GRAPH_ENABLED": "true"},
        reranker_model=args.reranker_model,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "baseline_hybrid_results.csv", baseline_rows + semantic_rows)
    write_csv(args.output_dir / "neo4j_graph_results.csv", graph_rows)
    summary = summarize(baseline_rows + semantic_rows + graph_rows)
    (args.output_dir / "graph_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
