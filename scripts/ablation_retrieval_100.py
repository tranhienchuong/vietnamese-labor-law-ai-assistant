from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate_retrieval_modes import (
    MODE_GRAPH,
    MODE_HYBRID,
    BenchmarkItem,
    CitationSpec,
    ModeResult,
    load_or_create_benchmark,
    markdown_table,
    matches_spec,
    run_mode,
    write_csv,
)


DEFAULT_BENCHMARK_PATH = REPO_ROOT / "artifacts" / "evaluation" / "golden_benchmark_100_extended.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "evaluation"
DEFAULT_OUTPUT_PREFIX = "ablation_retrieval_100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval ablation on the frozen 100-query benchmark.")
    parser.add_argument("--index-path", type=Path, default=REPO_ROOT / "artifacts" / "index")
    parser.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", type=str, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--prefetch-limit", type=int, default=24)
    parser.add_argument("--reranker-model", type=str, default=os.getenv("RERANKER_MODEL", ""))
    parser.add_argument(
        "--embedding-provider",
        type=str,
        default=os.getenv("GRAPH_RETRIEVAL_TEST_EMBEDDING_PROVIDER", "sentence_transformers"),
    )
    parser.add_argument("--device", type=str, default=os.getenv("GRAPH_RETRIEVAL_TEST_DEVICE", "cpu"))
    return parser.parse_args()


def in_corpus_items(items: Sequence[BenchmarkItem]) -> list[BenchmarkItem]:
    return [item for item in items if item.required_citations]


def citation_found(result: ModeResult, citation: CitationSpec, *, top_k: int) -> bool:
    limit = min(top_k, citation.top_n) if citation.top_n else top_k
    return any(matches_spec(context, citation) for context in result.contexts[:limit])


def missing_required_labels(result: ModeResult, *, top_k: int) -> list[str]:
    return [citation.label for citation in result.item.required_citations if not citation_found(result, citation, top_k=top_k)]


def forbidden_violation_labels(result: ModeResult) -> list[str]:
    labels = str(result.metrics.get("forbidden_citation_labels") or "").strip()
    if not labels:
        return []
    return [value.strip() for value in labels.split(";") if value.strip()]


def retrieval_passed(result: ModeResult, *, top_k: int) -> bool:
    return not missing_required_labels(result, top_k=top_k) and not forbidden_violation_labels(result)


def bool_rate(values: Sequence[bool]) -> float:
    return sum(1 for value in values if value) / len(values) if values else 0.0


def mode_display_name(mode: str) -> str:
    if mode == MODE_HYBRID:
        return "hybrid_only"
    if mode == MODE_GRAPH:
        return "graph_augmented"
    return mode


def aggregate_mode(results: Sequence[ModeResult], *, top_k: int) -> dict[str, object]:
    missing_cases: list[dict[str, object]] = []
    violation_cases: list[dict[str, object]] = []
    per_query_rows: list[dict[str, object]] = []

    for result in results:
        missing = missing_required_labels(result, top_k=top_k)
        violations = forbidden_violation_labels(result)
        passed = not missing and not violations
        row = {
            "mode": mode_display_name(result.mode),
            "id": result.item.id,
            "query": result.item.query,
            "category": result.item.category,
            "topic": result.item.topic,
            "recall_at_10": float(result.metrics["recall_at_10"]),
            "required_citation_coverage": float(result.metrics["required_citation_coverage"]),
            "forbidden_citation_violation": bool(result.metrics["forbidden_citation_violation"]),
            "retrieval_passed": passed,
            "missing_required_citations": " | ".join(missing),
            "forbidden_citation_violations": " | ".join(violations),
            "top10_citations": " || ".join(context.citation_text for context in result.contexts[:top_k]),
            "top10_chunk_ids": " || ".join(context.chunk_id for context in result.contexts[:top_k]),
        }
        per_query_rows.append(row)

        if missing:
            missing_cases.append(
                {
                    "id": result.item.id,
                    "query": result.item.query,
                    "category": result.item.category,
                    "missing_required_citations": missing,
                    "top10_citations": [context.citation_text for context in result.contexts[:top_k]],
                }
            )
        if violations:
            violation_cases.append(
                {
                    "id": result.item.id,
                    "query": result.item.query,
                    "category": result.item.category,
                    "forbidden_citation_violations": violations,
                    "top10_citations": [context.citation_text for context in result.contexts[:top_k]],
                }
            )

    return {
        "query_count": len(results),
        "recall_at_10": sum(float(result.metrics["recall_at_10"]) for result in results) / len(results) if results else 0.0,
        "required_citation_coverage": sum(float(result.metrics["required_citation_coverage"]) for result in results) / len(results)
        if results
        else 0.0,
        "forbidden_citation_violation_rate": bool_rate(
            [bool(result.metrics["forbidden_citation_violation"]) for result in results]
        ),
        "retrieval_pass_rate": bool_rate([retrieval_passed(result, top_k=top_k) for result in results]),
        "missing_required_citation_cases": missing_cases,
        "forbidden_citation_violations": violation_cases,
        "rows": per_query_rows,
    }


def render_report(summary: dict[str, object]) -> str:
    lines: list[str] = []
    lines.append("# Retrieval Ablation on Frozen 100-Query Benchmark")
    lines.append("")
    lines.append(f"- Generated at: {summary['generated_at']}")
    lines.append(f"- Benchmark: `{summary['benchmark_path']}`")
    lines.append(f"- In-corpus queries used for retrieval metrics: {summary['in_corpus_query_count']}")
    lines.append(f"- Top-k window: {summary['top_k']}")
    lines.append("")

    metric_rows: list[tuple[object, ...]] = []
    for mode_key in ("hybrid_only", "graph_augmented"):
        metrics = summary["modes"][mode_key]
        metric_rows.append(
            (
                mode_key,
                metrics["query_count"],
                f"{float(metrics['recall_at_10']):.3f}",
                f"{float(metrics['required_citation_coverage']):.3f}",
                f"{float(metrics['forbidden_citation_violation_rate']):.3f}",
                f"{float(metrics['retrieval_pass_rate']):.3f}",
            )
        )
    lines.extend(
        markdown_table(
            (
                "Mode",
                "Queries",
                "Recall@10",
                "Required Citation Coverage",
                "Forbidden Citation Violation Rate",
                "Retrieval Pass Rate",
            ),
            metric_rows,
        )
    )

    for mode_key in ("hybrid_only", "graph_augmented"):
        metrics = summary["modes"][mode_key]
        lines.extend(["", f"## {mode_key}", ""])
        lines.append(f"- Missing required citation cases: {len(metrics['missing_required_citation_cases'])}")
        if metrics["missing_required_citation_cases"]:
            for item in metrics["missing_required_citation_cases"]:
                lines.append(
                    f"- `{item['id']}` ({item['category']}): {', '.join(item['missing_required_citations'])}"
                )
        else:
            lines.append("- None")

        lines.append("")
        lines.append(f"- Forbidden citation violations: {len(metrics['forbidden_citation_violations'])}")
        if metrics["forbidden_citation_violations"]:
            for item in metrics["forbidden_citation_violations"]:
                lines.append(
                    f"- `{item['id']}` ({item['category']}): {', '.join(item['forbidden_citation_violations'])}"
                )
        else:
            lines.append("- None")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    if args.embedding_provider:
        os.environ["EMBEDDING_PROVIDER"] = args.embedding_provider

    benchmark_items = load_or_create_benchmark(args.benchmark_path)
    items = in_corpus_items(benchmark_items)
    all_results: list[ModeResult] = []
    for mode in (MODE_HYBRID, MODE_GRAPH):
        all_results.extend(
            run_mode(
                mode=mode,
                items=items,
                index_path=args.index_path,
                top_k=args.top_k,
                prefetch_limit=args.prefetch_limit,
                reranker_model=args.reranker_model,
                device=args.device,
            )
        )

    by_mode: dict[str, list[ModeResult]] = {"hybrid_only": [], "graph_augmented": []}
    for result in all_results:
        by_mode[mode_display_name(result.mode)].append(result)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_path": str(args.benchmark_path),
        "benchmark_query_count": len(benchmark_items),
        "in_corpus_query_count": len(items),
        "top_k": args.top_k,
        "modes": {
            mode_key: aggregate_mode(results, top_k=args.top_k)
            for mode_key, results in by_mode.items()
        },
    }

    rows: list[dict[str, object]] = []
    for mode_key in ("hybrid_only", "graph_augmented"):
        rows.extend(summary["modes"][mode_key].pop("rows"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.output_prefix}_results.csv"
    json_path = args.output_dir / f"{args.output_prefix}_results.json"
    report_path = args.output_dir / f"{args.output_prefix}_report.md"

    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8")

    print(
        json.dumps(
            {
                "csv_path": str(csv_path),
                "json_path": str(json_path),
                "report_path": str(report_path),
                "in_corpus_query_count": len(items),
                "modes": {
                    mode_key: {
                        "recall_at_10": summary["modes"][mode_key]["recall_at_10"],
                        "required_citation_coverage": summary["modes"][mode_key]["required_citation_coverage"],
                        "forbidden_citation_violation_rate": summary["modes"][mode_key]["forbidden_citation_violation_rate"],
                        "retrieval_pass_rate": summary["modes"][mode_key]["retrieval_pass_rate"],
                    }
                    for mode_key in ("hybrid_only", "graph_augmented")
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
