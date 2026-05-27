from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from .corpus_pipeline import normalize_for_matching


FILTER_HEADER = (
    "This is a labor-related filtered subset of Bộ luật Tố tụng dân sự 2015 "
    "for retrieval and legal graph construction."
)

STRICT_ARTICLE_RE = re.compile(r"^Điều\s+(?P<number>\d+[A-Za-z]?)\.\s+(?P<title>.*)$")
PART_RE = re.compile(r"^PHẦN\b", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^CHƯƠNG\s+", re.IGNORECASE)
SECTION_RE = re.compile(r"^MỤC\s+(?:\d+|[IVXLCDM]+)\b", re.IGNORECASE)

RAW_ONLY_KEYWORDS = {"đình công"}
LABOR_KEYWORDS = (
    "tranh chấp lao động",
    "tranh chấp về lao động",
    "người lao động",
    "người sử dụng lao động",
    "hợp đồng lao động",
    "chấm dứt hợp đồng lao động",
    "thỏa ước lao động",
    "thỏa ước lao động tập thể",
    "yêu cầu về lao động",
    "vụ án lao động",
    "vụ việc lao động",
    "bản án, quyết định lao động",
    "phán quyết lao động",
    "Tòa lao động",
    "đình công",
    "tổ chức đại diện tập thể lao động",
    "tập thể lao động",
    "pháp luật lao động",
    "Bộ luật lao động",
    "quan hệ lao động",
    "hòa giải viên lao động",
    "Hội đồng trọng tài lao động",
    "kỷ luật lao động",
    "sa thải",
    "tiền lương",
    "trả công lao động",
    "trợ cấp thôi việc",
    "trợ cấp mất việc",
    "trợ cấp mất sức lao động",
    "bảo hiểm xã hội",
    "bảo hiểm y tế",
    "bảo hiểm thất nghiệp",
    "tai nạn lao động",
    "bệnh nghề nghiệp",
)
PROCEDURAL_CONTEXT_KEYWORDS = (
    "thẩm quyền",
    "Tòa án",
    "khởi kiện",
    "hòa giải",
    "chứng minh",
    "chứng cứ",
    "biện pháp khẩn cấp",
    "tố tụng",
    "xét xử",
    "đại diện",
    "bảo vệ quyền",
    "thi hành",
    "án phí",
    "phúc thẩm",
    "giám đốc thẩm",
    "tái thẩm",
    "rút gọn",
    "tạm đình chỉ",
    "đình chỉ",
    "khiếu nại",
    "yêu cầu",
    "thụ lý",
    "xét đơn",
    "quyết định",
    "bản án",
    "thời hiệu",
)


@dataclass(frozen=True)
class HeadingBlock:
    level: str
    start_index: int
    start_line: int
    lines: tuple[str, ...]

    @property
    def key(self) -> tuple[str, int]:
        return self.level, self.start_line

    @property
    def text(self) -> str:
        return " ".join(line.strip() for line in self.lines if line.strip())


@dataclass(frozen=True)
class Article:
    number: str
    title: str
    start_index: int
    end_index: int
    start_line: int
    end_line: int
    lines: tuple[str, ...]
    part: HeadingBlock | None = None
    chapter: HeadingBlock | None = None
    section: HeadingBlock | None = None

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass(frozen=True)
class ArticleSelection:
    article: Article
    matched_keywords: tuple[str, ...]
    procedural_context_keywords: tuple[str, ...]
    reason: str
    confidence: str
    uncertain: bool
    suggested_by_validation: bool


def classify_heading_start(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if PART_RE.match(stripped):
        return "part"
    if CHAPTER_RE.match(stripped):
        return "chapter"
    if SECTION_RE.match(stripped):
        return "section"
    return None


def is_article_start(line: str) -> bool:
    return bool(STRICT_ARTICLE_RE.match(line.strip()))


def parse_heading_blocks(lines: Sequence[str]) -> list[HeadingBlock]:
    headings: list[HeadingBlock] = []
    index = 0
    while index < len(lines):
        level = classify_heading_start(lines[index])
        if not level:
            index += 1
            continue

        block_lines = [lines[index].strip()]
        lookahead = index + 1
        while lookahead < len(lines):
            candidate = lines[lookahead].strip()
            if is_article_start(candidate) or classify_heading_start(candidate):
                break
            if candidate:
                block_lines.append(candidate)
            lookahead += 1

        headings.append(
            HeadingBlock(
                level=level,
                start_index=index,
                start_line=index + 1,
                lines=tuple(block_lines),
            )
        )
        index = lookahead
    return headings


def trim_trailing_blank_lines(lines: Sequence[str]) -> tuple[str, ...]:
    end = len(lines)
    while end > 0 and not lines[end - 1].strip():
        end -= 1
    return tuple(lines[:end])


def parse_articles(text: str) -> list[Article]:
    lines = text.splitlines()
    headings = parse_heading_blocks(lines)
    heading_by_index = {heading.start_index: heading for heading in headings}
    article_starts: list[tuple[int, re.Match[str]]] = []

    for index, line in enumerate(lines):
        if match := STRICT_ARTICLE_RE.match(line.strip()):
            article_starts.append((index, match))

    current_part: HeadingBlock | None = None
    current_chapter: HeadingBlock | None = None
    current_section: HeadingBlock | None = None
    context_by_article_index: dict[
        int, tuple[HeadingBlock | None, HeadingBlock | None, HeadingBlock | None]
    ] = {}

    article_start_indexes = {start_index for start_index, _match in article_starts}
    for index in range(len(lines)):
        heading = heading_by_index.get(index)
        if heading:
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

    articles: list[Article] = []
    heading_indexes = sorted(heading.start_index for heading in headings)
    for position, (start_index, match) in enumerate(article_starts):
        next_article_index = (
            article_starts[position + 1][0] if position + 1 < len(article_starts) else len(lines)
        )
        next_heading_index = next(
            (
                heading_index
                for heading_index in heading_indexes
                if start_index < heading_index < next_article_index
            ),
            None,
        )
        end_index = next_heading_index if next_heading_index is not None else next_article_index
        part, chapter, section = context_by_article_index.get(start_index, (None, None, None))
        article_lines = trim_trailing_blank_lines(lines[start_index:end_index])
        articles.append(
            Article(
                number=match.group("number"),
                title=match.group("title").strip(),
                start_index=start_index,
                end_index=end_index,
                start_line=start_index + 1,
                end_line=end_index,
                lines=article_lines,
                part=part,
                chapter=chapter,
                section=section,
            )
        )
    return articles


def keyword_matches(keyword: str, text: str, normalized_text: str) -> bool:
    if keyword in RAW_ONLY_KEYWORDS:
        return keyword.casefold() in text.casefold()
    return normalize_for_matching(keyword) in normalized_text


def matched_keywords(article: Article) -> tuple[str, ...]:
    normalized = normalize_for_matching(article.text)
    return tuple(
        keyword
        for keyword in LABOR_KEYWORDS
        if keyword_matches(keyword, article.text, normalized)
    )


def matched_procedural_keywords(article: Article) -> tuple[str, ...]:
    normalized = normalize_for_matching(article.text)
    return tuple(
        keyword
        for keyword in PROCEDURAL_CONTEXT_KEYWORDS
        if normalize_for_matching(keyword) in normalized
    )


def select_labor_related_articles(
    articles: Sequence[Article],
    *,
    validation_suggested_articles: Iterable[str] | None = None,
) -> list[ArticleSelection]:
    suggestions = set(validation_suggested_articles or ())
    selected: list[ArticleSelection] = []

    for article in articles:
        labor_hits = matched_keywords(article)
        procedure_hits = matched_procedural_keywords(article)
        has_broad_labor_reference = "lao dong" in normalize_for_matching(article.text)
        if not labor_hits and not (has_broad_labor_reference and procedure_hits):
            continue

        if labor_hits:
            reason = "Matched explicit labor terms."
            confidence = "high"
            uncertain = False
        else:
            reason = "Mentions labor in a procedural context, but without a labor-specific marker."
            confidence = "medium"
            uncertain = True

        selected.append(
            ArticleSelection(
                article=article,
                matched_keywords=labor_hits,
                procedural_context_keywords=procedure_hits,
                reason=reason,
                confidence=confidence,
                uncertain=uncertain,
                suggested_by_validation=article.number in suggestions,
            )
        )

    return selected


def render_labor_subset(
    selections: Sequence[ArticleSelection],
    *,
    source_name: str | None = None,
) -> str:
    output_lines = [FILTER_HEADER]
    if source_name:
        output_lines.extend(["", f"Source: {source_name}"])
    output_lines.append("")

    emitted_contexts: set[tuple[str, int]] = set()
    for selection in selections:
        article = selection.article
        for heading in (article.part, article.chapter, article.section):
            if not heading or heading.key in emitted_contexts:
                continue
            if output_lines and output_lines[-1] != "":
                output_lines.append("")
            output_lines.extend(heading.lines)
            output_lines.append("")
            emitted_contexts.add(heading.key)

        if output_lines and output_lines[-1] != "":
            output_lines.append("")
        output_lines.extend(article.lines)
        output_lines.append("")

    return "\n".join(output_lines).strip() + "\n"


def load_validation_suggested_articles(validation_report_path: Path | None) -> set[str]:
    if not validation_report_path or not validation_report_path.exists():
        return set()

    payload = json.loads(validation_report_path.read_text(encoding="utf-8"))
    for document in payload.get("documents", []):
        if document.get("file_name") == "bo_luat_92_2015_qh13_to_tung_dan_su_2015_clean.txt":
            return {
                str(article.get("article_number"))
                for article in document.get("labor_related_units", {}).get("articles", [])
                if article.get("article_number")
            }
    return set()


def article_to_report_entry(selection: ArticleSelection) -> dict[str, Any]:
    article = selection.article
    return {
        "article_number": article.number,
        "article_title": article.title,
        "start_line": article.start_line,
        "end_line": article.end_line,
        "part": article.part.text if article.part else None,
        "chapter": article.chapter.text if article.chapter else None,
        "section": article.section.text if article.section else None,
        "matched_keywords": list(selection.matched_keywords),
        "procedural_context_keywords": list(selection.procedural_context_keywords),
        "reason": selection.reason,
        "confidence": selection.confidence,
        "manual_review_required": selection.uncertain,
        "suggested_by_validation": selection.suggested_by_validation,
    }


def build_filter_report(
    *,
    input_path: Path,
    output_path: Path,
    articles: Sequence[Article],
    selections: Sequence[ArticleSelection],
    validation_suggested_articles: Iterable[str],
) -> dict[str, Any]:
    kept_entries = [article_to_report_entry(selection) for selection in selections]
    kept_numbers = {entry["article_number"] for entry in kept_entries}
    strict_article_numbers = {article.number for article in articles}
    suggested_articles = set(validation_suggested_articles)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_article_count": len(articles),
        "kept_article_count": len(selections),
        "removed_article_count": len(articles) - len(selections),
        "kept_articles": kept_entries,
        "filter_keywords": {
            "labor_keywords": list(LABOR_KEYWORDS),
            "broad_keyword": "lao động",
            "procedural_context_keywords": list(PROCEDURAL_CONTEXT_KEYWORDS),
        },
        "validation_suggested_articles": sorted(
            suggested_articles & strict_article_numbers, key=lambda value: int(value)
        ),
        "validation_suggested_articles_not_kept": sorted(
            (suggested_articles & strict_article_numbers) - kept_numbers,
            key=lambda value: int(value),
        ),
        "validation_suggestions_not_strict_article_starts": sorted(
            suggested_articles - strict_article_numbers,
            key=lambda value: int(value),
        ),
        "uncertain_articles_for_manual_review": [
            entry for entry in kept_entries if entry["manual_review_required"]
        ],
    }


def write_filter_artifacts(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "blttds_labor_filter_report.json"
    markdown_path = output_dir / "blttds_labor_filter_report.md"

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# BLTTDS Labor Filter Report",
        "",
        f"- Input: `{report['input_path']}`",
        f"- Output: `{report['output_path']}`",
        f"- Total strict articles: {report['total_article_count']}",
        f"- Kept articles: {report['kept_article_count']}",
        f"- Removed articles: {report['removed_article_count']}",
        "",
        "## Kept Articles",
        "",
        "| Article | Line | Confidence | Manual review | Matched keywords | Title |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]

    for entry in report["kept_articles"]:
        keywords = ", ".join(entry["matched_keywords"]) or "labor + procedural context"
        manual = "yes" if entry["manual_review_required"] else "no"
        lines.append(
            "| "
            f"Điều {entry['article_number']} | "
            f"{entry['start_line']} | "
            f"{entry['confidence']} | "
            f"{manual} | "
            f"{keywords} | "
            f"{entry['article_title']} |"
        )

    lines.extend(
        [
            "",
            "## Manual Review",
            "",
        ]
    )
    uncertain = report["uncertain_articles_for_manual_review"]
    if not uncertain:
        lines.append("No uncertain kept articles.")
    else:
        for entry in uncertain:
            lines.append(
                f"- Điều {entry['article_number']} (line {entry['start_line']}): "
                f"{entry['article_title']}"
            )

    if report["validation_suggested_articles_not_kept"]:
        lines.extend(
            [
                "",
                "## Validation Suggestions Not Kept",
                "",
                ", ".join(f"Điều {number}" for number in report["validation_suggested_articles_not_kept"]),
            ]
        )

    if report["validation_suggestions_not_strict_article_starts"]:
        lines.extend(
            [
                "",
                "## Non-Strict Validation Suggestions",
                "",
                "These came from loose validation matches and were not strict `Điều N.` article starts:",
                "",
                ", ".join(
                    f"Điều {number}"
                    for number in report["validation_suggestions_not_strict_article_starts"]
                ),
            ]
        )

    return "\n".join(lines).strip() + "\n"


def filter_blttds_labor_subset(
    input_path: Path,
    output_path: Path,
    report_dir: Path,
    *,
    validation_report_path: Path | None = None,
) -> dict[str, Any]:
    text = input_path.read_text(encoding="utf-8")
    articles = parse_articles(text)
    validation_suggestions = load_validation_suggested_articles(validation_report_path)
    selections = select_labor_related_articles(
        articles,
        validation_suggested_articles=validation_suggestions,
    )
    subset_text = render_labor_subset(selections, source_name=input_path.name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(subset_text, encoding="utf-8")

    report = build_filter_report(
        input_path=input_path,
        output_path=output_path,
        articles=articles,
        selections=selections,
        validation_suggested_articles=validation_suggestions,
    )
    json_path, markdown_path = write_filter_artifacts(report, report_dir)
    report["json_report_path"] = str(json_path)
    report["markdown_report_path"] = str(markdown_path)
    return report
