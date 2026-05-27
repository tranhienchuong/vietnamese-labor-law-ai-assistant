from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from .curated_text_validation import (
    infer_curated_document_title,
    infer_document_id,
    infer_document_type,
    normalize_spaces,
    repair_article_title,
    strip_marker_prefix,
)


CURATED_LEGAL_UNIT_FILENAMES = (
    "45_2019_QH14.txt",
    "nghi_dinh_145_2020_nd_cp_clean.txt",
    "nghi_dinh_135_2020_nd_cp_clean.txt",
    "thong_tu_09_2020_tt_bldtbxh_clean.txt",
    "thong_tu_10_2020_tt_bldtbxh_clean.txt",
    "92_2015_QH13_labor_only.txt",
)
FULL_BLTTDS_FILENAME = "bo_luat_92_2015_qh13_to_tung_dan_su_2015_clean.txt"
BLTTDS_LABOR_ONLY_TITLE = (
    "Bộ luật Tố tụng dân sự số 92/2015/QH13 - labor-related filtered subset"
)

ARTICLE_RE = re.compile(r"^Điều\s+(?P<number>\d+[A-Za-z]?)\.\s*(?P<title>.*)$")
PART_RE = re.compile(r"^PHẦN\s+(?P<number>.+)$", re.IGNORECASE)
CHAPTER_RE = re.compile(
    r"^CHƯƠNG\s+(?P<number>[IVXLCDM0-9]+[A-ZĐ]?)\.?\s*(?P<title>.*)$",
    re.IGNORECASE,
)
SECTION_RE = re.compile(
    r"^MỤC\s+(?P<number>[IVXLCDM0-9]+|\d+)\.?\s*(?P<title>.*)$",
    re.IGNORECASE,
)
CLAUSE_RE = re.compile(r"^(?P<label>\d+)\.(?!\d)\s+(?P<text>.*)$")
SUBPOINT_RE = re.compile(r"^(?P<label>[a-zđ](?:\.\d+|\d+))\)\s+(?P<text>.*)$", re.IGNORECASE)
POINT_RE = re.compile(r"^(?P<label>[a-zđ])\)\s+(?P<text>.*)$", re.IGNORECASE)

VERY_SHORT_TEXT_THRESHOLD = 20


@dataclass(frozen=True)
class HeadingContext:
    level: str
    number: str
    heading: str
    start_index: int
    start_line: int
    lines: tuple[str, ...]


@dataclass(frozen=True)
class ArticleSpan:
    number: str
    title: str
    start_index: int
    end_index: int
    start_line: int
    end_line: int
    lines: tuple[str, ...]
    part: HeadingContext | None
    chapter: HeadingContext | None
    section: HeadingContext | None

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


@dataclass(frozen=True)
class DocumentMetadata:
    document_id: str
    document_title: str
    document_type: str
    source_file: str


@dataclass(frozen=True)
class ParsedLegalDocument:
    metadata: DocumentMetadata
    articles: tuple[ArticleSpan, ...]
    units: tuple[dict[str, Any], ...]


def normalize_legal_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def is_article_start(line: str) -> bool:
    return bool(ARTICLE_RE.match(line.strip()))


def classify_heading(line: str, index: int) -> HeadingContext | None:
    stripped = line.strip()
    if not stripped:
        return None

    if match := PART_RE.match(stripped):
        return HeadingContext(
            level="part",
            number=normalize_spaces(match.group("number")),
            heading=stripped,
            start_index=index,
            start_line=index + 1,
            lines=(stripped,),
        )
    if match := CHAPTER_RE.match(stripped):
        return HeadingContext(
            level="chapter",
            number=match.group("number"),
            heading=stripped,
            start_index=index,
            start_line=index + 1,
            lines=(stripped,),
        )
    if match := SECTION_RE.match(stripped):
        return HeadingContext(
            level="section",
            number=match.group("number"),
            heading=stripped,
            start_index=index,
            start_line=index + 1,
            lines=(stripped,),
        )
    return None


def heading_level(line: str) -> str | None:
    heading = classify_heading(line, 0)
    return heading.level if heading else None


def parse_heading_contexts(lines: Sequence[str]) -> list[HeadingContext]:
    headings: list[HeadingContext] = []
    index = 0
    while index < len(lines):
        base_heading = classify_heading(lines[index], index)
        if not base_heading:
            index += 1
            continue

        heading_lines = [lines[index].strip()]
        lookahead = index + 1
        while lookahead < len(lines):
            candidate = lines[lookahead].strip()
            if is_article_start(candidate) or heading_level(candidate):
                break
            if candidate:
                heading_lines.append(candidate)
            lookahead += 1

        heading_text = normalize_spaces(" ".join(heading_lines))
        headings.append(
            HeadingContext(
                level=base_heading.level,
                number=base_heading.number,
                heading=heading_text,
                start_index=base_heading.start_index,
                start_line=base_heading.start_line,
                lines=tuple(heading_lines),
            )
        )
        index = lookahead
    return headings


def trim_trailing_blank_lines(lines: Sequence[str]) -> tuple[str, ...]:
    end = len(lines)
    while end > 0 and not lines[end - 1].strip():
        end -= 1
    return tuple(lines[:end])


def parse_article_spans(text: str) -> list[ArticleSpan]:
    lines = normalize_legal_text(text).splitlines()
    headings = parse_heading_contexts(lines)
    heading_by_index = {heading.start_index: heading for heading in headings}
    heading_start_indexes = sorted(heading_by_index)

    article_starts: list[tuple[int, re.Match[str]]] = []
    for index, line in enumerate(lines):
        if match := ARTICLE_RE.match(line.strip()):
            article_starts.append((index, match))

    article_start_indexes = {index for index, _match in article_starts}
    context_by_article_index: dict[
        int, tuple[HeadingContext | None, HeadingContext | None, HeadingContext | None]
    ] = {}
    current_part: HeadingContext | None = None
    current_chapter: HeadingContext | None = None
    current_section: HeadingContext | None = None

    for index in range(len(lines)):
        if heading := heading_by_index.get(index):
            if heading.level == "part":
                current_part = heading
                current_chapter = None
                current_section = None
            elif heading.level == "chapter":
                current_chapter = heading
                current_section = None
            elif heading.level == "section":
                current_section = heading
        if index in article_start_indexes:
            context_by_article_index[index] = (current_part, current_chapter, current_section)

    spans: list[ArticleSpan] = []
    for position, (start_index, match) in enumerate(article_starts):
        next_article_index = (
            article_starts[position + 1][0] if position + 1 < len(article_starts) else len(lines)
        )
        next_heading_index = next(
            (
                heading_index
                for heading_index in heading_start_indexes
                if start_index < heading_index < next_article_index
            ),
            None,
        )
        end_index = next_heading_index if next_heading_index is not None else next_article_index
        part, chapter, section = context_by_article_index.get(start_index, (None, None, None))
        spans.append(
            ArticleSpan(
                number=match.group("number"),
                title=repair_article_title(lines, start_index, match.group("title")),
                start_index=start_index,
                end_index=end_index,
                start_line=start_index + 1,
                end_line=end_index,
                lines=trim_trailing_blank_lines(lines[start_index:end_index]),
                part=part,
                chapter=chapter,
                section=section,
            )
        )
    return spans


def detect_body_marker(line: str) -> tuple[str | None, str | None]:
    stripped = strip_marker_prefix(line)
    if not stripped:
        return None, None
    if match := CLAUSE_RE.match(stripped):
        return "clause", match.group("label")
    if match := SUBPOINT_RE.match(stripped):
        return "point", match.group("label")
    if match := POINT_RE.match(stripped):
        return "point", match.group("label")
    return None, None


def unit_base_record(metadata: DocumentMetadata, article: ArticleSpan) -> dict[str, Any]:
    return {
        "document_id": metadata.document_id,
        "document_title": metadata.document_title,
        "document_type": metadata.document_type,
        "source_file": metadata.source_file,
        "part_number": article.part.number if article.part else None,
        "part_heading": article.part.heading if article.part else None,
        "chapter_number": article.chapter.number if article.chapter else None,
        "chapter_heading": article.chapter.heading if article.chapter else None,
        "section_number": article.section.number if article.section else None,
        "section_heading": article.section.heading if article.section else None,
        "article_number": article.number,
        "article_title": article.title,
    }


def text_from_line_span(lines: Sequence[str], start_index: int, end_index: int) -> str:
    return "\n".join(trim_trailing_blank_lines(lines[start_index:end_index])).strip()


def build_article_unit(metadata: DocumentMetadata, article: ArticleSpan) -> dict[str, Any]:
    record = unit_base_record(metadata, article)
    record.update(
        {
            "clause_ref": None,
            "point_ref": None,
            "unit_level": "article",
            "text": article.text,
        }
    )
    return record


def build_clause_and_point_units(
    metadata: DocumentMetadata,
    article: ArticleSpan,
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    current_clause_start: int | None = None
    current_clause_ref: str | None = None
    current_point_start: int | None = None
    current_point_ref: str | None = None

    def close_point(end_index: int) -> None:
        nonlocal current_point_start, current_point_ref
        if current_point_start is None or current_point_ref is None:
            return
        text = text_from_line_span(article.lines, current_point_start, end_index)
        if text:
            record = unit_base_record(metadata, article)
            record.update(
                {
                    "clause_ref": current_clause_ref,
                    "point_ref": current_point_ref,
                    "unit_level": "point",
                    "text": text,
                }
            )
            units.append(record)
        current_point_start = None
        current_point_ref = None

    def close_clause(end_index: int) -> None:
        nonlocal current_clause_start, current_clause_ref
        if current_clause_start is None or current_clause_ref is None:
            return
        text = text_from_line_span(article.lines, current_clause_start, end_index)
        if text:
            record = unit_base_record(metadata, article)
            record.update(
                {
                    "clause_ref": current_clause_ref,
                    "point_ref": None,
                    "unit_level": "clause",
                    "text": text,
                }
            )
            units.append(record)
        current_clause_start = None
        current_clause_ref = None

    for index, line in enumerate(article.lines):
        marker_kind, marker_label = detect_body_marker(line)
        if marker_kind == "clause":
            close_point(index)
            close_clause(index)
            current_clause_start = index
            current_clause_ref = marker_label
        elif marker_kind == "point":
            close_point(index)
            current_point_start = index
            current_point_ref = marker_label

    close_point(len(article.lines))
    close_clause(len(article.lines))
    return units


def infer_legal_unit_document_metadata(path: Path, text: str) -> DocumentMetadata:
    if path.name == "92_2015_QH13_labor_only.txt":
        title = BLTTDS_LABOR_ONLY_TITLE
    else:
        title = infer_curated_document_title(text, path)
    document_type = infer_document_type(title, path, text)
    return DocumentMetadata(
        document_id=infer_document_id(path, text),
        document_title=title,
        document_type=document_type,
        source_file=path.name,
    )


def parse_legal_document(path: Path) -> ParsedLegalDocument:
    text = normalize_legal_text(path.read_text(encoding="utf-8"))
    metadata = infer_legal_unit_document_metadata(path, text)
    articles = tuple(parse_article_spans(text))
    units: list[dict[str, Any]] = []

    for article in articles:
        units.append(build_article_unit(metadata, article))
        units.extend(build_clause_and_point_units(metadata, article))

    return ParsedLegalDocument(metadata=metadata, articles=articles, units=tuple(units))


def resolve_curated_legal_unit_paths(
    input_dir: Path,
    filenames: Sequence[str] = CURATED_LEGAL_UNIT_FILENAMES,
) -> list[Path]:
    paths: list[Path] = []
    missing: list[str] = []
    for filename in filenames:
        path = input_dir / filename
        if path.exists():
            paths.append(path)
        else:
            missing.append(filename)
    if missing:
        raise FileNotFoundError(
            "Missing required curated legal text file(s): " + ", ".join(missing)
        )
    return paths


def build_legal_units_from_directory(
    input_dir: Path,
    *,
    filenames: Sequence[str] = CURATED_LEGAL_UNIT_FILENAMES,
) -> tuple[list[ParsedLegalDocument], list[dict[str, Any]]]:
    paths = resolve_curated_legal_unit_paths(input_dir, filenames)
    documents = [parse_legal_document(path) for path in paths]
    units = [unit for document in documents for unit in document.units]
    return documents, units


def count_units_by_level(units: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = {"article": 0, "clause": 0, "point": 0}
    for unit in units:
        level = str(unit["unit_level"])
        counts[level] = counts.get(level, 0) + 1
    return counts


def build_legal_units_summary(
    documents: Sequence[ParsedLegalDocument],
    units: Sequence[dict[str, Any]],
    *,
    input_dir: Path,
) -> dict[str, Any]:
    units_by_document: dict[str, list[dict[str, Any]]] = {}
    for unit in units:
        units_by_document.setdefault(str(unit["document_id"]), []).append(unit)

    legal_unit_counts_by_document: dict[str, dict[str, Any]] = {}
    article_counts_by_document: dict[str, int] = {}
    clause_counts_by_document: dict[str, int] = {}
    point_counts_by_document: dict[str, int] = {}
    suspicious_missing_titles: list[dict[str, Any]] = []
    duplicate_articles: list[dict[str, Any]] = []

    for document in documents:
        document_id = document.metadata.document_id
        document_units = units_by_document.get(document_id, [])
        level_counts = count_units_by_level(document_units)
        legal_unit_counts_by_document[document_id] = {
            "document_title": document.metadata.document_title,
            "source_file": document.metadata.source_file,
            "total": len(document_units),
            **level_counts,
        }
        article_counts_by_document[document_id] = level_counts.get("article", 0)
        clause_counts_by_document[document_id] = level_counts.get("clause", 0)
        point_counts_by_document[document_id] = level_counts.get("point", 0)

        seen_articles: dict[str, list[ArticleSpan]] = {}
        for article in document.articles:
            if not article.title:
                suspicious_missing_titles.append(
                    {
                        "document_id": document_id,
                        "source_file": document.metadata.source_file,
                        "article_number": article.number,
                        "line": article.start_line,
                    }
                )
            seen_articles.setdefault(article.number, []).append(article)

        for article_number, occurrences in seen_articles.items():
            if len(occurrences) > 1:
                duplicate_articles.append(
                    {
                        "document_id": document_id,
                        "source_file": document.metadata.source_file,
                        "article_number": article_number,
                        "lines": [article.start_line for article in occurrences],
                    }
                )

    short_units = []
    for unit in units:
        text = normalize_spaces(str(unit.get("text", "")))
        if 0 < len(text) < VERY_SHORT_TEXT_THRESHOLD:
            short_units.append(
                {
                    "document_id": unit["document_id"],
                    "source_file": unit["source_file"],
                    "article_number": unit["article_number"],
                    "clause_ref": unit["clause_ref"],
                    "point_ref": unit["point_ref"],
                    "unit_level": unit["unit_level"],
                    "text_length": len(text),
                    "text": text,
                }
            )

    source_files = [document.metadata.source_file for document in documents]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "source_files": source_files,
        "excluded_files": [FULL_BLTTDS_FILENAME],
        "document_count": len(documents),
        "unit_count": len(units),
        "legal_unit_counts_by_document": legal_unit_counts_by_document,
        "article_counts_by_document": article_counts_by_document,
        "clause_counts_by_document": clause_counts_by_document,
        "point_counts_by_document": point_counts_by_document,
        "suspicious_articles_with_missing_title": suspicious_missing_titles,
        "duplicate_article_numbers_in_same_document": duplicate_articles,
        "units_with_very_short_text": short_units[:200],
    }


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Legal Units Summary",
        "",
        f"- Input directory: `{summary['input_dir']}`",
        f"- Documents parsed: {summary['document_count']}",
        f"- Legal units: {summary['unit_count']}",
        f"- Excluded files: {', '.join(summary['excluded_files'])}",
        "",
        "## Counts By Document",
        "",
        "| Document ID | Source file | Articles | Clauses | Points | Total units |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for document_id, counts in summary["legal_unit_counts_by_document"].items():
        lines.append(
            "| "
            f"{document_id} | "
            f"{counts['source_file']} | "
            f"{counts['article']} | "
            f"{counts['clause']} | "
            f"{counts['point']} | "
            f"{counts['total']} |"
        )

    lines.extend(["", "## Suspicious Articles With Missing Title", ""])
    missing_titles = summary["suspicious_articles_with_missing_title"]
    if missing_titles:
        for item in missing_titles:
            lines.append(
                f"- {item['source_file']} Điều {item['article_number']} "
                f"(line {item['line']})"
            )
    else:
        lines.append("None.")

    lines.extend(["", "## Duplicate Article Numbers", ""])
    duplicates = summary["duplicate_article_numbers_in_same_document"]
    if duplicates:
        for item in duplicates:
            lines.append(
                f"- {item['source_file']} Điều {item['article_number']}: "
                f"lines {', '.join(str(line) for line in item['lines'])}"
            )
    else:
        lines.append("None.")

    lines.extend(["", "## Very Short Units", ""])
    short_units = summary["units_with_very_short_text"]
    if short_units:
        for item in short_units[:50]:
            ref_parts = [f"Điều {item['article_number']}"]
            if item["clause_ref"]:
                ref_parts.append(f"khoản {item['clause_ref']}")
            if item["point_ref"]:
                ref_parts.append(f"điểm {item['point_ref']}")
            lines.append(
                f"- {item['source_file']} {' '.join(ref_parts)} "
                f"({item['unit_level']}, {item['text_length']} chars): {item['text']}"
            )
    else:
        lines.append("None.")

    return "\n".join(lines).strip() + "\n"


def write_legal_unit_artifacts(
    documents: Sequence[ParsedLegalDocument],
    units: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    input_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "legal_units.jsonl"
    summary_json_path = output_dir / "legal_units_summary.json"
    summary_md_path = output_dir / "legal_units_summary.md"

    summary = build_legal_units_summary(documents, units, input_dir=input_dir)
    write_jsonl(jsonl_path, units)
    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_md_path.write_text(render_summary_markdown(summary), encoding="utf-8")
    return jsonl_path, summary_json_path, summary_md_path
