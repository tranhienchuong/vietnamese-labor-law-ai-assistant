from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    "expected_citations_in_scope",
    "expected_citations_out_of_scope",
    "case_scope",
    "retrieved_citations",
    "generated_answer",
    "generated_legal_basis",
    "insufficient_context",
)

LEGAL_ARTICLE_RE = re.compile(r"\bdieu\s+(?P<value>\d+[a-z]?)")
LEGAL_CLAUSE_RE = re.compile(r"\bkhoan\s+(?P<value>\d+)")
LEGAL_POINT_GROUP_RE = re.compile(
    r"\bdiem\s+(?P<value>[a-z](?:\s*(?:,|va)\s*[a-z])*)"
)
TOP_LEVEL_CITATION_SPLIT_RE = re.compile(r"\s*;\s*")

DOCUMENT_FAMILY_LABELS = {
    "bo_luat_2019": "Bộ luật Lao động 2019",
    "nghi_dinh_145": "Nghị định 145/2020/NĐ-CP",
    "nghi_dinh_12_2022": "Nghị định 12/2022/NĐ-CP",
    "luat_cong_doan_2024": "Luật Công đoàn 2024",
    "thuvienphapluat_article": "THƯ VIỆN PHÁP LUẬT",
}
DOCUMENT_FAMILY_SIGNATURES = {
    "bo_luat_2019": (
        "bo luat lao dong 2019",
        "45 2019 qh 14",
        "45 2019 qh14",
    ),
    "nghi_dinh_145": (
        "nghi dinh 145 2020 nd cp",
        "145 2020 nd cp",
    ),
    "nghi_dinh_12_2022": (
        "nghi dinh 12 2022 nd cp",
        "12 2022 nd cp",
    ),
    "luat_cong_doan_2024": ("luat cong doan 2024",),
    "thuvienphapluat_article": ("thu vien phap luat",),
}


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
    gold_citations: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        raw_citations: Iterable[str | None]
        if self.gold_citations:
            raw_citations = tuple(self.gold_citations)
        else:
            raw_citations = (
                self.gold_citation_primary,
                self.gold_citation_secondary,
            )
        object.__setattr__(
            self,
            "gold_citations",
            expand_expected_citations(raw_citations),
        )


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


def unique_preserve_order(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return tuple(ordered)


def normalize_signature_text(text: str) -> str:
    normalized = normalize_for_matching(text)
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def citation_document_families(text: str) -> tuple[str, ...]:
    normalized = normalize_signature_text(text)
    matches: list[str] = []
    for family, signatures in DOCUMENT_FAMILY_SIGNATURES.items():
        if any(signature in normalized for signature in signatures):
            matches.append(family)
    return tuple(matches)


def split_citation_field(text: str) -> tuple[str, ...]:
    return unique_preserve_order(
        part.strip(" ;")
        for part in TOP_LEVEL_CITATION_SPLIT_RE.split(text.strip())
        if part.strip(" ;")
    )


def extract_point_values(text: str) -> tuple[str, ...]:
    normalized = normalize_for_matching(text)
    values: list[str] = []
    for match in LEGAL_POINT_GROUP_RE.finditer(normalized):
        raw_values = match.group("value")
        for point in re.split(r"\s*(?:,|va)\s*", raw_values):
            cleaned = point.strip()
            if cleaned:
                values.append(cleaned)
    return unique_preserve_order(values)


def compose_atomic_citation(
    *,
    article: str | None = None,
    clause: str | None = None,
    point: str | None = None,
    document_family: str | None = None,
) -> str:
    parts: list[str] = []
    if article:
        parts.append(f"Điều {article}")
    if clause:
        parts.append(f"khoản {clause}")
    if point:
        parts.append(f"điểm {point}")
    if document_family:
        parts.append(DOCUMENT_FAMILY_LABELS[document_family])
    return " ".join(parts).strip()


def expand_compound_citation(fragment: str, *, default_family: str | None = None) -> tuple[str, ...]:
    cleaned_fragment = fragment.strip()
    if not cleaned_fragment:
        return ()

    explicit_families = citation_document_families(cleaned_fragment)
    document_family = explicit_families[0] if len(explicit_families) == 1 else default_family
    normalized = normalize_signature_text(cleaned_fragment)
    article_values = unique_preserve_order(match.group("value") for match in LEGAL_ARTICLE_RE.finditer(normalized))
    clause_values = unique_preserve_order(match.group("value") for match in LEGAL_CLAUSE_RE.finditer(normalized))
    point_values = extract_point_values(cleaned_fragment)

    if len(article_values) > 1 and (clause_values or point_values):
        return (cleaned_fragment,)
    if len(clause_values) > 1 and point_values:
        return (cleaned_fragment,)

    if len(article_values) > 1:
        return unique_preserve_order(
            compose_atomic_citation(article=article, document_family=document_family)
            for article in article_values
        )

    if article_values:
        article = article_values[0]
        if point_values:
            clause = clause_values[0] if len(clause_values) == 1 else None
            if clause is None and clause_values:
                return (cleaned_fragment,)
            return unique_preserve_order(
                compose_atomic_citation(
                    article=article,
                    clause=clause,
                    point=point,
                    document_family=document_family,
                )
                for point in point_values
            )

        if len(clause_values) > 1:
            return unique_preserve_order(
                compose_atomic_citation(
                    article=article,
                    clause=clause,
                    document_family=document_family,
                )
                for clause in clause_values
            )

        return (
            compose_atomic_citation(
                article=article,
                clause=clause_values[0] if clause_values else None,
                document_family=document_family,
            ),
        )

    return (cleaned_fragment,)


def expand_expected_citations(values: Iterable[str | None]) -> tuple[str, ...]:
    expanded: list[str] = []
    for value in values:
        if not value:
            continue
        fragments = split_citation_field(value)
        families = citation_document_families(value)
        default_family = families[0] if len(families) == 1 else None
        for fragment in fragments:
            expanded.extend(expand_compound_citation(fragment, default_family=default_family))
    return unique_preserve_order(expanded)


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
                gold_citations=expand_expected_citations(
                    (
                        coerce_optional_text(value_for("gold_citation_primary")),
                        coerce_optional_text(value_for("gold_citation_secondary")),
                    )
                ),
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
    return tuple(case.gold_citations)


def partition_citations_by_scope(
    citations: Sequence[str],
    *,
    allowed_document_families: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not allowed_document_families:
        return tuple(citations), ()

    allowed = set(allowed_document_families)
    in_scope: list[str] = []
    out_of_scope: list[str] = []

    for citation in citations:
        family = citation_document_family(citation)
        if family is None or family in allowed:
            in_scope.append(citation)
        else:
            out_of_scope.append(citation)

    return tuple(in_scope), tuple(out_of_scope)


def expected_citations_in_scope(
    case: BenchmarkCase,
    *,
    allowed_document_families: Sequence[str] | None,
) -> tuple[str, ...]:
    in_scope, _ = partition_citations_by_scope(
        expected_citations(case),
        allowed_document_families=allowed_document_families,
    )
    return in_scope


def expected_citations_out_of_scope(
    case: BenchmarkCase,
    *,
    allowed_document_families: Sequence[str] | None,
) -> tuple[str, ...]:
    _, out_of_scope = partition_citations_by_scope(
        expected_citations(case),
        allowed_document_families=allowed_document_families,
    )
    return out_of_scope


def expected_citation_scope(
    case: BenchmarkCase,
    *,
    allowed_document_families: Sequence[str] | None,
) -> str:
    citations = expected_citations(case)
    if not citations:
        return "no_gold_citation"

    in_scope, out_of_scope = partition_citations_by_scope(
        citations,
        allowed_document_families=allowed_document_families,
    )
    if in_scope and out_of_scope:
        return "mixed_scope"
    if in_scope:
        return "in_scope"
    return "out_of_scope"


def document_families_from_chunk_paths(chunk_paths: Sequence[str | Path]) -> tuple[str, ...]:
    families: list[str] = []
    for chunk_path in chunk_paths:
        for family in citation_document_families(str(chunk_path)):
            families.append(family)
    return unique_preserve_order(families)


def citation_document_family(text: str) -> str | None:
    families = citation_document_families(text)
    if len(families) == 1:
        return families[0]
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
    allowed_document_families: Sequence[str] | None = None,
) -> bool | None:
    gold_citations = expected_citations_in_scope(
        case,
        allowed_document_families=allowed_document_families,
    )
    if not gold_citations:
        return None if allowed_document_families else False

    for expected in gold_citations:
        for observed in observed_citations[:k]:
            if citation_matches_expected(expected, observed):
                return True
    return False


def score_citation_correctness(
    case: BenchmarkCase,
    observed_citations: Sequence[str],
) -> str:
    return score_citation_correctness_for_scope(case, observed_citations)


def score_citation_correctness_for_scope(
    case: BenchmarkCase,
    observed_citations: Sequence[str],
    *,
    allowed_document_families: Sequence[str] | None = None,
) -> str:
    gold_citations = expected_citations_in_scope(
        case,
        allowed_document_families=allowed_document_families,
    )
    if not gold_citations:
        return "na" if allowed_document_families else "no"
    if not observed_citations:
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
    "document_families_from_chunk_paths",
    "expected_citations",
    "expected_citations_in_scope",
    "expected_citations_out_of_scope",
    "expected_citation_scope",
    "load_benchmark_jsonl",
    "load_benchmark_workbook",
    "parse_benchmark_rows",
    "parse_yes_no_flag",
    "partition_citations_by_scope",
    "require_openpyxl",
    "retrieval_hit_at_k",
    "score_citation_correctness",
    "score_citation_correctness_for_scope",
    "summarize_benchmark_cases",
    "write_benchmark_jsonl",
    "write_results_csv",
    "write_results_jsonl",
]
