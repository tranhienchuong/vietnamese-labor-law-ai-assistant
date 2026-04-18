from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import re
from typing import Iterable, Sequence

from .corpus_pipeline import normalize_for_matching
from .indexing import extract_legal_hint_tokens


WORKBOOK_SHEET_NAME = "golden_benchmark"
WORKBOOK_RESULTS_SHEET_NAME = "evaluation_results"
BENCHMARK_JSONL_NAME = "golden_benchmark_100_answered_v1.jsonl"
RESULTS_COLUMNS = (
    "id",
    "model_version",
    "retrieval_hit_at_5",
    "citation_correct",
    "answer_correct",
    "hallucination_flag",
    "abstention_correct",
    "clarity_score_1_5",
    "format_score_1_5",
    "final_score_10",
    "evaluator",
    "comments",
    "question",
    "expected_citations",
    "retrieved_citations",
    "generated_answer",
    "generated_legal_basis",
    "insufficient_context",
)


def require_openpyxl():
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("openpyxl is required for benchmark workbook import.") from exc
    return load_workbook


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    category: str
    subtopic: str
    difficulty: str
    question_type: str
    question: str
    scenario: str
    gold_issue: str
    gold_citation_primary: str | None
    gold_citation_secondary: str | None
    gold_answer_short: str
    gold_answer_full: str
    abstain_required: bool
    missing_information: str | None
    source_document: str | None
    source_url: str | None
    annotator: str | None
    review_status: str | None
    notes: str | None


def coerce_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def coerce_required_text(value: object, field_name: str) -> str:
    text = coerce_optional_text(value)
    if text is None:
        raise ValueError(f"Missing required benchmark field: {field_name}")
    return text


def parse_yes_no_flag(value: object) -> bool:
    text = normalize_for_matching(str(value or ""))
    return text in {"yes", "y", "true", "1"}


def locate_header_row(rows: Sequence[Sequence[object]]) -> tuple[int, list[str]]:
    for index, row in enumerate(rows):
        normalized = [normalize_for_matching(str(cell or "")) for cell in row]
        if normalized and normalized[0] == "id":
            return index, [str(cell or "").strip() for cell in row]
    raise ValueError("Could not locate benchmark header row.")


def parse_benchmark_rows(rows: Sequence[Sequence[object]]) -> list[BenchmarkCase]:
    header_index, headers = locate_header_row(rows)
    header_map = {header: position for position, header in enumerate(headers) if header}
    cases: list[BenchmarkCase] = []

    for row in rows[header_index + 1 :]:
        padded_row = list(row) + [None] * max(0, len(headers) - len(row))
        row_id = coerce_optional_text(padded_row[header_map["id"]])
        if row_id is None:
            continue

        def value_for(header: str) -> object:
            return padded_row[header_map[header]]

        cases.append(
            BenchmarkCase(
                id=coerce_required_text(value_for("id"), "id"),
                category=coerce_required_text(value_for("category"), "category"),
                subtopic=coerce_required_text(value_for("subtopic"), "subtopic"),
                difficulty=coerce_required_text(value_for("difficulty"), "difficulty"),
                question_type=coerce_required_text(value_for("question_type"), "question_type"),
                question=coerce_required_text(value_for("question"), "question"),
                scenario=coerce_required_text(value_for("scenario"), "scenario"),
                gold_issue=coerce_required_text(value_for("gold_issue"), "gold_issue"),
                gold_citation_primary=coerce_optional_text(value_for("gold_citation_primary")),
                gold_citation_secondary=coerce_optional_text(value_for("gold_citation_secondary")),
                gold_answer_short=coerce_required_text(value_for("gold_answer_short"), "gold_answer_short"),
                gold_answer_full=coerce_required_text(value_for("gold_answer_full"), "gold_answer_full"),
                abstain_required=parse_yes_no_flag(value_for("abstain_required")),
                missing_information=coerce_optional_text(value_for("missing_information")),
                source_document=coerce_optional_text(value_for("source_document")),
                source_url=coerce_optional_text(value_for("source_url")),
                annotator=coerce_optional_text(value_for("annotator")),
                review_status=coerce_optional_text(value_for("review_status")),
                notes=coerce_optional_text(value_for("notes")),
            )
        )

    return cases


def load_benchmark_workbook(
    workbook_path: Path,
    *,
    sheet_name: str = WORKBOOK_SHEET_NAME,
) -> list[BenchmarkCase]:
    load_workbook = require_openpyxl()
    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        worksheet = workbook[sheet_name]
        rows = [tuple(row) for row in worksheet.iter_rows(values_only=True)]
    finally:
        workbook.close()
    return parse_benchmark_rows(rows)


def load_benchmark_jsonl(input_path: Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        cases.append(BenchmarkCase(**payload))
    return cases


def write_benchmark_jsonl(cases: Sequence[BenchmarkCase], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")


def write_results_csv(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULTS_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_results_jsonl(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def expected_citations(case: BenchmarkCase) -> tuple[str, ...]:
    citations = [case.gold_citation_primary, case.gold_citation_secondary]
    return tuple(citation for citation in citations if citation)


def citation_document_family(text: str) -> str | None:
    normalized = normalize_for_matching(text)
    if "bo luat" in normalized or "45/2019" in normalized or "qh 14" in normalized:
        return "bo_luat_2019"
    if "nghi dinh" in normalized or "145/2020" in normalized or "nd cp" in normalized:
        return "nghi_dinh_145"
    return None


def citation_matches_expected(expected: str, observed: str) -> bool:
    expected_normalized = normalize_for_matching(expected)
    observed_normalized = normalize_for_matching(observed)

    expected_family = citation_document_family(expected)
    observed_family = citation_document_family(observed)
    if expected_family and observed_family and expected_family != observed_family:
        return False

    expected_tokens = set(extract_legal_hint_tokens(expected))
    observed_tokens = set(extract_legal_hint_tokens(observed))
    if expected_tokens:
        return expected_tokens.issubset(observed_tokens)

    return expected_normalized in observed_normalized or observed_normalized in expected_normalized


def retrieval_hit_at_k(
    case: BenchmarkCase,
    observed_citations: Sequence[str],
    *,
    k: int = 5,
) -> bool:
    gold_citations = expected_citations(case)
    if not gold_citations:
        return False

    for expected in gold_citations:
        for observed in observed_citations[:k]:
            if citation_matches_expected(expected, observed):
                return True
    return False


def score_citation_correctness(
    case: BenchmarkCase,
    observed_citations: Sequence[str],
) -> str:
    gold_citations = expected_citations(case)
    if not observed_citations:
        return "no"
    if not gold_citations:
        return "no"

    matched = [
        expected
        for expected in gold_citations
        if any(citation_matches_expected(expected, observed) for observed in observed_citations)
    ]
    if len(matched) == len(gold_citations):
        return "exact"
    if matched:
        return "partial"
    return "no"


def summarize_benchmark_cases(cases: Sequence[BenchmarkCase]) -> dict[str, object]:
    difficulties: dict[str, int] = {}
    categories: dict[str, int] = {}
    for case in cases:
        difficulties[case.difficulty] = difficulties.get(case.difficulty, 0) + 1
        categories[case.category] = categories.get(case.category, 0) + 1
    return {
        "case_count": len(cases),
        "difficulty_distribution": difficulties,
        "category_distribution": categories,
    }


__all__ = [
    "BENCHMARK_JSONL_NAME",
    "BenchmarkCase",
    "RESULTS_COLUMNS",
    "WORKBOOK_RESULTS_SHEET_NAME",
    "WORKBOOK_SHEET_NAME",
    "citation_matches_expected",
    "expected_citations",
    "load_benchmark_jsonl",
    "load_benchmark_workbook",
    "parse_benchmark_rows",
    "parse_yes_no_flag",
    "require_openpyxl",
    "retrieval_hit_at_k",
    "score_citation_correctness",
    "summarize_benchmark_cases",
    "write_benchmark_jsonl",
    "write_results_csv",
    "write_results_jsonl",
]
