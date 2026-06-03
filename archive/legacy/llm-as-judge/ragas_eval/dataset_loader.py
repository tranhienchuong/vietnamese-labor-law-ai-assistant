from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE_REQUIRED_FIELDS = ("id", "user_input", "reference")

FIELD_ALIASES = {
    "id": ("id", "sample_id", "case_id"),
    "user_input": ("user_input", "question", "query", "input"),
    "response": ("response", "generated_answer", "model_answer", "prediction", "output"),
    "reference": (
        "reference",
        "gold_answer",
        "gold_answer_full",
        "expected_answer",
        "ground_truth",
        "answer",
        "gold_answer_short",
    ),
    "retrieved_contexts": (
        "retrieved_contexts",
        "contexts",
        "retrieved_context",
        "retrieval_contexts",
        "source_contexts",
    ),
    "reference_contexts": (
        "reference_contexts",
        "gold_contexts",
        "gold_context",
        "expected_contexts",
        "ground_truth_contexts",
    ),
    "gold_citation": ("gold_citation", "expected_citations", "gold_citations"),
}


@dataclass(frozen=True)
class BenchmarkSample:
    id: str
    user_input: str
    reference: str
    response: str = ""
    retrieved_contexts: tuple[str, ...] = field(default_factory=tuple)
    reference_contexts: tuple[str, ...] = field(default_factory=tuple)
    gold_citation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_ragas_record(self) -> dict[str, Any]:
        return {
            "user_input": self.user_input,
            "response": self.response,
            "retrieved_contexts": list(self.retrieved_contexts),
            "reference": self.reference,
        }

    def to_output_base(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_input": self.user_input,
        }

    def to_legal_judge_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_input": self.user_input,
            "response": self.response,
            "retrieved_contexts": list(self.retrieved_contexts),
            "reference": self.reference,
            "reference_contexts": list(self.reference_contexts),
            "gold_citation": self.gold_citation,
        }


@dataclass(frozen=True)
class ValidationIssue:
    sample_id: str
    field: str
    message: str
    row_number: int

    def format(self) -> str:
        return f"{self.sample_id} (row {self.row_number}) [{self.field}]: {self.message}"


@dataclass(frozen=True)
class ValidationReport:
    total_samples: int
    valid_samples: tuple[BenchmarkSample, ...]
    errors: tuple[ValidationIssue, ...]

    @property
    def valid_count(self) -> int:
        return len(self.valid_samples)

    @property
    def error_count(self) -> int:
        return len(self.errors)


class BenchmarkValidationError(ValueError):
    def __init__(self, errors: Sequence[ValidationIssue]) -> None:
        message = "Benchmark schema validation failed:\n" + "\n".join(
            f"- {error.format()}" for error in errors
        )
        super().__init__(message)
        self.errors = tuple(errors)


def read_benchmark_records(input_path: Path) -> list[dict[str, Any]]:
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(input_path)
    if suffix == ".json":
        return _read_json(input_path)
    if suffix == ".csv":
        return _read_csv(input_path)
    raise ValueError("Unsupported benchmark format. Expected .jsonl, .json, or .csv.")


def _read_jsonl(input_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        input_path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL line {line_number} must be an object.")
        records.append(payload)
    return records


def _read_json(input_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        payload = payload["samples"]
    if not isinstance(payload, list):
        raise ValueError("JSON benchmark must be a list or an object with a samples list.")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"JSON sample {index} must be an object.")
        records.append(item)
    return records


def _read_csv(input_path: Path) -> list[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def validate_benchmark_schema(
    records: Sequence[Mapping[str, Any]],
    *,
    require_response: bool = False,
    require_retrieved_contexts: bool = False,
) -> ValidationReport:
    samples: list[BenchmarkSample] = []
    errors: list[ValidationIssue] = []

    for row_number, record in enumerate(records, start=1):
        normalized = normalize_benchmark_record(record)
        sample_id = normalized.get("id") or f"<missing id at row {row_number}>"
        row_errors = _validate_normalized_record(
            normalized,
            sample_id=sample_id,
            row_number=row_number,
            require_response=require_response,
            require_retrieved_contexts=require_retrieved_contexts,
        )
        if row_errors:
            errors.extend(row_errors)
            continue
        samples.append(_sample_from_normalized(normalized, record))

    return ValidationReport(
        total_samples=len(records),
        valid_samples=tuple(samples),
        errors=tuple(errors),
    )


def load_benchmark_samples(
    input_path: Path,
    *,
    require_response: bool = False,
    require_retrieved_contexts: bool = False,
) -> list[BenchmarkSample]:
    records = read_benchmark_records(input_path)
    report = validate_benchmark_schema(
        records,
        require_response=require_response,
        require_retrieved_contexts=require_retrieved_contexts,
    )
    if report.errors:
        raise BenchmarkValidationError(report.errors)
    return list(report.valid_samples)


def normalize_benchmark_record(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for canonical_name, aliases in FIELD_ALIASES.items():
        value = _first_present(record, aliases)
        if value is not None:
            normalized[canonical_name] = value

    gold_citation = _gold_citation_from_record(record)
    if gold_citation:
        normalized["gold_citation"] = gold_citation

    if "retrieved_contexts" in normalized:
        normalized["retrieved_contexts"] = _coerce_contexts(normalized["retrieved_contexts"])
    else:
        normalized["retrieved_contexts"] = ()

    if "reference_contexts" in normalized:
        normalized["reference_contexts"] = _coerce_contexts(normalized["reference_contexts"])
    else:
        normalized["reference_contexts"] = ()

    for text_field in ("id", "user_input", "response", "reference", "gold_citation"):
        if text_field in normalized:
            normalized[text_field] = _coerce_text(normalized[text_field])

    return normalized


def _first_present(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return None


def _gold_citation_from_record(record: Mapping[str, Any]) -> str:
    values: list[str] = []
    for field_name in (
        "gold_citation",
        "expected_citations",
        "gold_citations",
        "gold_citation_primary",
        "gold_citation_secondary",
    ):
        if field_name not in record or record[field_name] in (None, ""):
            continue
        value = record[field_name]
        if isinstance(value, list):
            values.extend(str(item).strip() for item in value if str(item).strip())
        else:
            values.append(str(value).strip())
    return " | ".join(value for value in values if value)


def _coerce_contexts(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, tuple):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return tuple(str(item).strip() for item in parsed if str(item).strip())
        separator = "\n" if "\n" in stripped else " | "
        return tuple(part.strip() for part in stripped.split(separator) if part.strip())
    return (str(value).strip(),)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " | ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _validate_normalized_record(
    normalized: Mapping[str, Any],
    *,
    sample_id: str,
    row_number: int,
    require_response: bool,
    require_retrieved_contexts: bool,
) -> list[ValidationIssue]:
    errors: list[ValidationIssue] = []
    for field_name in BASE_REQUIRED_FIELDS:
        if not normalized.get(field_name):
            errors.append(
                ValidationIssue(
                    sample_id=sample_id,
                    field=field_name,
                    message="missing required field",
                    row_number=row_number,
                )
            )

    if require_response and not normalized.get("response"):
        errors.append(
            ValidationIssue(
                sample_id=sample_id,
                field="response",
                message="response is required for evaluation runs",
                row_number=row_number,
            )
        )

    contexts = normalized.get("retrieved_contexts", ())
    if not isinstance(contexts, tuple) or not all(isinstance(item, str) for item in contexts):
        errors.append(
            ValidationIssue(
                sample_id=sample_id,
                field="retrieved_contexts",
                message="must be a list of strings",
                row_number=row_number,
            )
        )
    elif require_retrieved_contexts and not contexts:
        errors.append(
            ValidationIssue(
                sample_id=sample_id,
                field="retrieved_contexts",
                message="retrieved_contexts is required for RAGAS metrics",
                row_number=row_number,
            )
        )

    for field_name in ("reference_contexts",):
        value = normalized.get(field_name, ())
        if not isinstance(value, tuple) or not all(isinstance(item, str) for item in value):
            errors.append(
                ValidationIssue(
                    sample_id=sample_id,
                    field=field_name,
                    message="must be a list of strings",
                    row_number=row_number,
                )
            )

    return errors


def _sample_from_normalized(
    normalized: Mapping[str, Any],
    original_record: Mapping[str, Any],
) -> BenchmarkSample:
    known_fields = set(FIELD_ALIASES) | {
        alias for aliases in FIELD_ALIASES.values() for alias in aliases
    }
    metadata = {
        key: value
        for key, value in original_record.items()
        if key not in known_fields and key not in {"gold_citation_primary", "gold_citation_secondary"}
    }
    return BenchmarkSample(
        id=str(normalized["id"]),
        user_input=str(normalized["user_input"]),
        reference=str(normalized["reference"]),
        response=str(normalized.get("response", "")),
        retrieved_contexts=tuple(normalized.get("retrieved_contexts", ())),
        reference_contexts=tuple(normalized.get("reference_contexts", ())),
        gold_citation=str(normalized.get("gold_citation", "")),
        metadata=metadata,
    )


def _print_report(report: ValidationReport) -> None:
    print(f"Total samples: {report.total_samples}")
    print(f"Valid samples: {report.valid_count}")
    print(f"Errored samples: {report.error_count}")
    if report.errors:
        print("Errors:")
        for error in report.errors:
            print(f"- {error.format()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate or load a RAGAS benchmark dataset.")
    parser.add_argument("--input", required=True, type=Path, help="Path to .jsonl, .json, or .csv.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate schema without requiring generated responses.",
    )
    parser.add_argument(
        "--require-response",
        action="store_true",
        help="Require response for evaluation-ready datasets.",
    )
    parser.add_argument(
        "--require-contexts",
        action="store_true",
        help="Require retrieved_contexts for evaluation-ready datasets.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = read_benchmark_records(args.input)
    report = validate_benchmark_schema(
        records,
        require_response=args.require_response,
        require_retrieved_contexts=args.require_contexts,
    )
    _print_report(report)
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
