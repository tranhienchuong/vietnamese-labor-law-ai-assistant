"""Compute adjusted split metrics for the final 100-query benchmark.

The built-in end-to-end evaluator treats empty required-citation sets as
retrieval-covered. This script separates in-corpus retrieval scoring from
out-of-corpus refusal scoring and writes the corrected adjusted metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _norm(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def _is_refusal_or_insufficient_context(result: dict[str, Any]) -> bool:
    answer_metrics = result.get("answer_metrics", {})
    if answer_metrics.get("insufficient_context") is True:
        return True

    answer_text = _norm(result.get("answer", ""))
    refusal_phrases = [
        "insufficient context",
        "not enough context",
        "cannot answer",
        "khong du thong tin",
        "khong co du thong tin",
        "khong tim thay",
        "khong the tra loi",
        "ngoai pham vi",
    ]
    return any(phrase in answer_text for phrase in refusal_phrases)


def _has_specific_ooc_value(result: dict[str, Any]) -> bool:
    text = _norm(result.get("answer", ""))
    query = _norm(result.get("query", ""))
    value_query = "ty le" in query or "phan tram" in query
    value_pattern = re.compile(r"\b\d+(?:[,.]\d+)?\s*(?:%|phan tram)")
    return value_query and bool(value_pattern.search(text))


def _unsupported_citation_count(result: dict[str, Any]) -> int:
    answer_metrics = result.get("answer_metrics", {})
    return len(answer_metrics.get("unsupported_article_numbers", [])) + len(
        answer_metrics.get("unretrieved_citations", [])
    )


def _label_list(items: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("label", "")) for item in items if item.get("label")]


def compute_metrics(results_payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results = results_payload["results"]
    in_corpus = [item for item in results if item.get("required_citations")]
    out_of_corpus = [item for item in results if not item.get("required_citations")]

    in_retrieval_coverages = [
        float(item.get("retrieval_metrics", {}).get("required_citation_coverage", 0.0))
        for item in in_corpus
    ]
    in_forbidden_violations = [
        bool(item.get("retrieval_metrics", {}).get("forbidden_citation_violations", []))
        for item in in_corpus
    ]
    in_retrieval_passes = [
        bool(item.get("retrieval_metrics", {}).get("retrieval_passed", False))
        for item in in_corpus
    ]
    in_e2e_passes = [
        bool(item.get("end_to_end_metrics", {}).get("end_to_end_passed", False))
        for item in in_corpus
    ]

    ooc_details: list[dict[str, Any]] = []
    for item in out_of_corpus:
        refusal = _is_refusal_or_insufficient_context(item)
        unsupported_count = _unsupported_citation_count(item)
        specific_value = _has_specific_ooc_value(item)
        legal_basis = item.get("legal_basis", [])
        ooc_details.append(
            {
                "id": item.get("id"),
                "query": item.get("query"),
                "insufficient_context_or_refusal": refusal,
                "parser_insufficient_context": item.get("answer_metrics", {}).get(
                    "insufficient_context", False
                ),
                "unsupported_citation_count": unsupported_count,
                "legal_basis_count": len(legal_basis),
                "legal_basis": legal_basis,
                "specific_ooc_value_flag": specific_value,
                "pass": refusal and unsupported_count == 0 and not specific_value,
                "built_in_e2e_passed": bool(
                    item.get("end_to_end_metrics", {}).get("end_to_end_passed", False)
                ),
                "built_in_failure_reasons": item.get("end_to_end_metrics", {}).get(
                    "failure_reasons", []
                ),
            }
        )

    ooc_passes = [detail["pass"] for detail in ooc_details]
    adjusted_pass_count = sum(in_e2e_passes) + sum(ooc_passes)
    total = len(results)

    answer_passes = [
        bool(item.get("answer_metrics", {}).get("answer_passed", False)) for item in results
    ]
    citation_passes = [
        bool(item.get("answer_metrics", {}).get("citation_grounding_passed", False))
        for item in results
    ]
    quality_passes = [
        bool(item.get("answer_metrics", {}).get("quality_validation_passed", False))
        for item in results
    ]

    adjusted_failed_rows: list[dict[str, Any]] = []
    for item in in_corpus:
        if bool(item.get("end_to_end_metrics", {}).get("end_to_end_passed", False)):
            continue
        retrieval_metrics = item.get("retrieval_metrics", {})
        adjusted_failed_rows.append(
            {
                "id": item.get("id"),
                "scope": "in_corpus",
                "category": item.get("category"),
                "query": item.get("query"),
                "failure_reasons": " | ".join(
                    item.get("end_to_end_metrics", {}).get("failure_reasons", [])
                ),
                "missing_required_citations": " | ".join(
                    retrieval_metrics.get("missing_required_citations", [])
                ),
                "forbidden_violations": " | ".join(
                    retrieval_metrics.get("forbidden_citation_violations", [])
                ),
            }
        )

    for detail in ooc_details:
        if detail["pass"]:
            continue
        reasons = []
        if not detail["insufficient_context_or_refusal"]:
            reasons.append("out_of_corpus_not_refused")
        if detail["legal_basis_count"]:
            reasons.append("answered_with_in_corpus_citations")
        if detail["unsupported_citation_count"]:
            reasons.append("unsupported_citations")
        if detail["specific_ooc_value_flag"]:
            reasons.append("specific_ooc_value")
        adjusted_failed_rows.append(
            {
                "id": detail["id"],
                "scope": "out_of_corpus",
                "category": "out_of_corpus_qa",
                "query": detail["query"],
                "failure_reasons": " | ".join(reasons),
                "missing_required_citations": "",
                "forbidden_violations": "",
            }
        )

    metrics = {
        "source_files": {
            "results": "",
        },
        "benchmark": {
            "query_count": total,
            "in_corpus_query_count": len(in_corpus),
            "out_of_corpus_query_count": len(out_of_corpus),
        },
        "in_corpus": {
            "retrieval": {
                "query_count": len(in_corpus),
                "recall_at_10_macro": _mean(in_retrieval_coverages),
                "required_citation_coverage_macro": _mean(in_retrieval_coverages),
                "forbidden_citation_violation_rate": _rate(
                    sum(in_forbidden_violations), len(in_corpus)
                ),
                "retrieval_pass_rate": _rate(sum(in_retrieval_passes), len(in_corpus)),
                "missing_required_cases": [
                    item.get("id")
                    for item in in_corpus
                    if item.get("retrieval_metrics", {}).get("missing_required_citations", [])
                ],
                "forbidden_violation_cases": [
                    item.get("id")
                    for item in in_corpus
                    if item.get("retrieval_metrics", {}).get(
                        "forbidden_citation_violations", []
                    )
                ],
            },
            "end_to_end_existing_logic": {
                "query_count": len(in_corpus),
                "end_to_end_pass_rate_existing_logic": _rate(
                    sum(in_e2e_passes), len(in_corpus)
                ),
                "failed_cases_existing_logic": [
                    item.get("id")
                    for item in in_corpus
                    if not item.get("end_to_end_metrics", {}).get("end_to_end_passed", False)
                ],
            },
        },
        "out_of_corpus": {
            "query_count": len(out_of_corpus),
            "insufficient_context_pass_rate": _rate(sum(ooc_passes), len(out_of_corpus)),
            "full_refusal_pass_rate": _rate(sum(ooc_passes), len(out_of_corpus)),
            "unsupported_citation_count": sum(
                detail["unsupported_citation_count"] for detail in ooc_details
            ),
            "legal_basis_citation_count": sum(
                detail["legal_basis_count"] for detail in ooc_details
            ),
            "specific_ooc_value_failure_count": sum(
                detail["specific_ooc_value_flag"] for detail in ooc_details
            ),
            "failed_cases": [detail for detail in ooc_details if not detail["pass"]],
            "details": ooc_details,
        },
        "adjusted_100": {
            "query_count": total,
            "in_corpus_pass_count": sum(in_e2e_passes),
            "out_of_corpus_pass_count": sum(ooc_passes),
            "adjusted_end_to_end_pass_count": adjusted_pass_count,
            "adjusted_end_to_end_pass_rate": _rate(adjusted_pass_count, total),
            "built_in_end_to_end_pass_rate": results_payload.get("overall", {}).get(
                "end_to_end_pass_rate"
            ),
            "answer_pass_rate": _rate(sum(answer_passes), total),
            "citation_grounding_pass_rate": _rate(sum(citation_passes), total),
            "quality_validation_pass_rate": _rate(sum(quality_passes), total),
            "low_information_quotes": sum(
                int(item.get("answer_metrics", {}).get("low_information_quotes_count", 0))
                for item in results
            ),
            "failed_cases_adjusted": [row["id"] for row in adjusted_failed_rows],
        },
    }
    return metrics, adjusted_failed_rows


def write_failed_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "scope",
                "category",
                "query",
                "failure_reasons",
                "missing_required_citations",
                "forbidden_violations",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute adjusted split metrics for the final 100-query benchmark."
    )
    parser.add_argument(
        "--results-path",
        default="artifacts/evaluation/end_to_end_100_extended_results.json",
        help="Path to end_to_end_100_extended_results.json.",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/evaluation/benchmark_100_split_metrics.json",
        help="Path for the adjusted split metrics JSON.",
    )
    parser.add_argument(
        "--failed-cases-csv",
        default="artifacts/evaluation/benchmark_100_adjusted_failed_cases.csv",
        help="Path for the adjusted failed-cases CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results_path = Path(args.results_path)
    output_json = Path(args.output_json)
    failed_csv = Path(args.failed_cases_csv)

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    metrics, failed_rows = compute_metrics(payload)

    metrics["source_files"]["results"] = str(results_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_failed_csv(failed_csv, failed_rows)

    adjusted = metrics["adjusted_100"]["adjusted_end_to_end_pass_rate"]
    print(f"Wrote {output_json}")
    print(f"Wrote {failed_csv}")
    print(f"AdjustedE2E = {adjusted:.3f}")
