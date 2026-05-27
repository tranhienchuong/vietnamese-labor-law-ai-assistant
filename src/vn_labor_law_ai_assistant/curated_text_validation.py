from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence
import unicodedata

from .corpus_pipeline import (
    CHAPTER_RE,
    SECTION_RE,
    infer_document_title,
    normalize_for_matching,
    slugify_text,
)


EXPECTED_CURATED_FILENAMES = (
    "45_2019_QH14.txt",
    "nghi_dinh_145_2020_nd_cp_clean.txt",
    "nghi_dinh_135_2020_nd_cp_clean.txt",
    "thong_tu_09_2020_tt_bldtbxh_clean.txt",
    "thong_tu_10_2020_tt_bldtbxh_clean.txt",
    "bo_luat_92_2015_qh13_to_tung_dan_su_2015_clean.txt",
)

ARTICLE_LINE_RE = re.compile(
    r"^Điều\s+(?P<number>\d+[A-Za-z]?)\.?\s*(?P<title>.*)$",
    re.IGNORECASE,
)
ANY_CHAPTER_LINE_RE = re.compile(
    r"^Chương\s+(?P<label>[^\s.]+)\.?\s*(?P<title>.*)$",
    re.IGNORECASE,
)
VALID_CHAPTER_LABEL_RE = re.compile(r"^(?:[IVXLCDM]+|\d+)$", re.IGNORECASE)
CLAUSE_MARKER_RE = re.compile(r"^(?P<label>\d+)\.(?!\d)\s+(?P<text>.+)$")
POINT_MARKER_RE = re.compile(r"^(?P<label>[a-zđ])\)\s+(?P<text>.+)$", re.IGNORECASE)
SUBPOINT_MARKER_RE = re.compile(
    r"^(?P<label>[a-zđ](?:\.\d+|\d+))\)\s+(?P<text>.+)$",
    re.IGNORECASE,
)
NUMERIC_SUBPOINT_RE = re.compile(r"^\d+\.\d+\.\s+")
LETTER_DIGIT_JOIN_RE = re.compile(
    r"(?P<prefix>[A-Za-zÀ-ỹ]{2,})(?P<prefix_digits>\d+)|"
    r"(?P<suffix_digits>\d+)(?P<suffix>[A-Za-zÀ-ỹ]{2,})"
)

QH_ID_RE = re.compile(
    r"(?P<number>\d+)[_\-/\s]+(?P<year>\d{4})[_\-/\s]+qh(?P<session>\d+)",
    re.IGNORECASE,
)
ND_ID_RE = re.compile(
    r"(?P<number>\d+)[_\-/\s]+(?P<year>\d{4})[_\-/\s]+n[đd][_\-/\s]*cp",
    re.IGNORECASE,
)
TT_ID_RE = re.compile(
    r"(?P<number>\d+)[_\-/\s]+(?P<year>\d{4})[_\-/\s]+tt[_\-/\s]+bl[đd]tbxh",
    re.IGNORECASE,
)
LEGAL_SIGNATURE_RE = re.compile(
    r"(?P<number>\d+)\s*/\s*(?P<year>\d{4})\s*/\s*"
    r"(?P<kind>QH\s*\d+|N[ĐD]\s*-\s*CP|TT\s*-\s*BL[ĐD]TBXH)",
    re.IGNORECASE,
)
CIVIL_PROCEDURE_TITLE_RE = re.compile(
    r"Bộ\s+Luật\s+Tố\s+tụng\s+dân\s+sự\s+số\s+"
    r"(?P<number>\d+)\s*/\s*(?P<year>\d{4})\s*/\s*QH\s*(?P<session>\d+)",
    re.IGNORECASE,
)

SKIPPED_TITLE_LINES = {
    "quoc hoi",
    "chinh phu",
    "cong hoa xa hoi chu nghia viet nam",
    "doc lap - tu do - hanh phuc",
}
TITLE_STOP_PREFIXES = (
    "can cu",
    "theo de nghi",
    "chinh phu ban hanh",
    "bo truong",
    "chuong ",
    "dieu ",
    "so:",
    "luat so:",
    "ha noi",
)
HEADING_CONTINUATION_WORDS = {
    "của",
    "về",
    "theo",
    "và",
    "hoặc",
    "đối",
    "với",
    "trong",
    "tại",
    "đủ",
    "chưa",
    "nông",
    "chức",
    "năng",
    "phát",
    "sự",
}
JOINED_WORD_PATTERNS = (
    "sửdụng",
    "laođộng",
    "hợpđồng",
    "ngườilao",
    "ngườisử",
    "trợcấp",
    "thờigian",
    "quyđịnh",
    "nghịđịnh",
    "bộluật",
)
DISTORTED_TERM_PATTERNS = (
    ("HỢP ĐÔNG", "HỢP ĐỒNG"),
    ("CHÁM DỨT", "CHẤM DỨT"),
    ("sản phâm", "sản phẩm"),
)
LABOR_KEYWORDS = (
    "lao động",
    "hợp đồng lao động",
    "người lao động",
    "người sử dụng lao động",
    "tranh chấp lao động",
    "việc lao động",
    "thỏa ước lao động",
    "đình công",
    "sa thải",
    "tiền lương",
    "bảo hiểm thất nghiệp",
    "tai nạn lao động",
    "bệnh nghề nghiệp",
)
MEASUREMENT_SUFFIXES = {
    "cm",
    "mm",
    "kg",
    "kva",
    "kv",
    "kw",
    "hz",
    "mhz",
}
SUGGESTED_CIVIL_PROCEDURE_FILTERED_FILE = "data/curated/92_2015_QH13_labor_only.txt"
MAX_REPORTED_ISSUES = 200


@dataclass(frozen=True)
class ArticleSpan:
    number: str
    title: str
    line: int
    start_index: int
    end_index: int


def strip_marker_prefix(line: str) -> str:
    return re.sub(r"^[\s\"'`“”‘’\-–—•]+", "", line).strip()


def normalize_curated_text_for_validation(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00ad", "")
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def preview(value: str, limit: int = 180) -> str:
    value = normalize_spaces(value)
    return value if len(value) <= limit else f"{value[: limit - 3]}..."


def normalized_line_key(line: str) -> str:
    return normalize_for_matching(normalize_spaces(strip_marker_prefix(line)))


def starts_with_lowercase_or_digit(line: str) -> bool:
    stripped = strip_marker_prefix(line)
    if not stripped:
        return False
    first = stripped[0]
    return first.isdigit() or first == first.lower() and first != first.upper()


def is_marker_line(line: str) -> bool:
    stripped = strip_marker_prefix(line)
    return bool(
        ARTICLE_LINE_RE.match(stripped)
        or CHAPTER_RE.match(stripped)
        or SECTION_RE.match(stripped)
        or CLAUSE_MARKER_RE.match(stripped)
        or POINT_MARKER_RE.match(stripped)
        or SUBPOINT_MARKER_RE.match(stripped)
        or NUMERIC_SUBPOINT_RE.match(stripped)
    )


def find_next_non_empty_line(lines: Sequence[str], start_index: int) -> tuple[int, str] | None:
    for index in range(start_index + 1, len(lines)):
        if lines[index].strip():
            return index, lines[index].strip()
    return None


def looks_like_heading_continuation(title: str, next_line: str) -> bool:
    if not title.strip() or not next_line.strip() or is_marker_line(next_line):
        return False
    if title.rstrip().endswith((".", ":", ";")):
        return False

    normalized_title = normalize_for_matching(title)
    last_word = normalized_title.rsplit(" ", 1)[-1] if normalized_title else ""
    return last_word in HEADING_CONTINUATION_WORDS or starts_with_lowercase_or_digit(next_line)


def repair_article_title(lines: Sequence[str], article_index: int, title: str) -> str:
    repaired = normalize_spaces(title)
    index = article_index
    for _ in range(3):
        next_line = find_next_non_empty_line(lines, index)
        if not next_line:
            break
        next_index, value = next_line
        if not looks_like_heading_continuation(repaired, value):
            break
        repaired = normalize_spaces(f"{repaired} {strip_marker_prefix(value)}")
        index = next_index
    return repaired


def normalize_signature(number: str, year: str, kind: str) -> str:
    compact_kind = re.sub(r"\s+", "", kind.upper()).replace("D", "Đ")
    if compact_kind.startswith("NĐ"):
        compact_kind = "NĐ-CP"
    elif compact_kind.startswith("TT"):
        compact_kind = "TT-BLĐTBXH"
    elif compact_kind.startswith("QH"):
        session_match = re.search(r"\d+", compact_kind)
        compact_kind = f"QH{session_match.group(0)}" if session_match else compact_kind
    return f"{number}/{year}/{compact_kind}"


def find_legal_signature(text: str, fallback_name: str) -> str | None:
    for source in (text, fallback_name):
        if match := LEGAL_SIGNATURE_RE.search(source):
            return normalize_signature(
                match.group("number"),
                match.group("year"),
                match.group("kind"),
            )

    if match := QH_ID_RE.search(fallback_name):
        return f"{match.group('number')}/{match.group('year')}/QH{match.group('session')}"
    if match := ND_ID_RE.search(fallback_name):
        return f"{match.group('number')}/{match.group('year')}/NĐ-CP"
    if match := TT_ID_RE.search(fallback_name):
        return f"{match.group('number')}/{match.group('year')}/TT-BLĐTBXH"
    return None


def infer_document_id(path: Path, text: str = "") -> str:
    fallback_name = path.stem.removesuffix("_clean")
    searchable = f"{fallback_name}\n{text[:4000]}"

    if match := TT_ID_RE.search(searchable):
        return f"thong-tu-{match.group('number')}-{match.group('year')}-tt-bldtbxh"
    if match := ND_ID_RE.search(searchable):
        return f"nghi-dinh-{match.group('number')}-{match.group('year')}-nd-cp"
    if match := QH_ID_RE.search(searchable):
        return f"{match.group('number')}-{match.group('year')}-qh{match.group('session')}"

    if signature := find_legal_signature(text, fallback_name):
        normalized = signature.lower().replace("/", "-").replace("đ", "d")
        if normalized.endswith("nd-cp"):
            return f"nghi-dinh-{normalized}"
        if normalized.endswith("tt-bldtbxh"):
            return f"thong-tu-{normalized}"
        return normalized

    return slugify_text(fallback_name)


def infer_document_type(document_title: str, path: Path, text: str = "") -> str:
    normalized = normalize_for_matching(f"{document_title} {path.stem} {text[:1000]}")
    if "thong tu" in normalized or TT_ID_RE.search(path.stem):
        return "Thông tư"
    if "nghi dinh" in normalized or ND_ID_RE.search(path.stem):
        return "Nghị định"
    if "bo luat" in normalized or "luat so" in normalized or QH_ID_RE.search(path.stem):
        return "Bộ luật"
    return "Văn bản pháp luật"


def extract_document_subtitle(lines: Sequence[str], kind: str) -> str | None:
    normalized_kind = normalize_for_matching(kind)
    start_index: int | None = None
    for index, line in enumerate(lines):
        stripped = strip_marker_prefix(line)
        if normalize_for_matching(stripped) == normalized_kind:
            start_index = index
            break
    if start_index is None:
        return None

    title_lines: list[str] = []
    for line in lines[start_index + 1 :]:
        stripped = normalize_spaces(strip_marker_prefix(line))
        if not stripped:
            if title_lines:
                break
            continue
        normalized = normalize_for_matching(stripped)
        if normalized in SKIPPED_TITLE_LINES or stripped.startswith("----"):
            continue
        if normalized.startswith(TITLE_STOP_PREFIXES):
            break
        title_lines.append(stripped)

    return normalize_spaces(" ".join(title_lines)) or None


def infer_curated_document_title(text: str, path: Path) -> str:
    cleaned_text = normalize_curated_text_for_validation(text)
    lines = cleaned_text.splitlines()
    header_text = "\n".join(lines[:40])
    fallback_name = path.stem.removesuffix("_clean")

    if match := CIVIL_PROCEDURE_TITLE_RE.search(header_text):
        signature = (
            f"{match.group('number')}/{match.group('year')}/"
            f"QH{match.group('session')}"
        )
        return f"Bộ luật Tố tụng dân sự số {signature}"

    signature = find_legal_signature(cleaned_text, fallback_name)
    if signature:
        normalized_signature = signature.lower()
        if "tt-b" in normalized_signature:
            subtitle = extract_document_subtitle(lines, "THÔNG TƯ")
            return f"Thông tư {signature}" + (f" - {subtitle}" if subtitle else "")
        if "nđ-cp" in normalized_signature:
            subtitle = extract_document_subtitle(lines, "NGHỊ ĐỊNH")
            return f"Nghị định {signature}" + (f" - {subtitle}" if subtitle else "")
        if "/qh" in normalized_signature:
            inferred = infer_document_title(cleaned_text, fallback_name)
            return inferred

    return infer_document_title(cleaned_text, fallback_name)


def append_issue(
    issues: list[dict[str, Any]],
    *,
    issue_type: str,
    line: int | None,
    sample: str,
    message: str,
    expected: str | None = None,
) -> None:
    if len(issues) >= MAX_REPORTED_ISSUES:
        return
    issue = {
        "type": issue_type,
        "line": line,
        "sample": preview(sample),
        "message": message,
    }
    if expected:
        issue["expected"] = expected
    if issue not in issues:
        issues.append(issue)


def detect_markers(lines: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    detected_chapters: list[dict[str, Any]] = []
    detected_sections: list[dict[str, Any]] = []
    detected_articles: list[dict[str, Any]] = []
    detected_clauses: list[dict[str, Any]] = []
    detected_points: list[dict[str, Any]] = []
    current_article_number: str | None = None

    for index, line in enumerate(lines):
        stripped = strip_marker_prefix(line)
        line_number = index + 1
        if not stripped:
            continue

        if chapter_match := CHAPTER_RE.match(stripped):
            detected_chapters.append(
                {
                    "line": line_number,
                    "label": chapter_match.group("number"),
                    "title": chapter_match.group("title").strip(),
                    "text": stripped,
                }
            )
            continue

        if section_match := SECTION_RE.match(stripped):
            detected_sections.append(
                {
                    "line": line_number,
                    "label": section_match.group("number"),
                    "title": section_match.group("title").strip(),
                    "text": stripped,
                }
            )
            continue

        if article_match := ARTICLE_LINE_RE.match(stripped):
            article_title = repair_article_title(lines, index, article_match.group("title"))
            current_article_number = article_match.group("number")
            detected_articles.append(
                {
                    "line": line_number,
                    "number": current_article_number,
                    "title": article_title,
                    "text": stripped,
                }
            )
            continue

        if NUMERIC_SUBPOINT_RE.match(stripped):
            continue

        if clause_match := CLAUSE_MARKER_RE.match(stripped):
            detected_clauses.append(
                {
                    "line": line_number,
                    "label": clause_match.group("label"),
                    "article_number": current_article_number,
                    "text": stripped,
                }
            )
            continue

        if subpoint_match := SUBPOINT_MARKER_RE.match(stripped):
            detected_points.append(
                {
                    "line": line_number,
                    "label": subpoint_match.group("label"),
                    "kind": "subpoint",
                    "article_number": current_article_number,
                    "text": stripped,
                }
            )
            continue

        if point_match := POINT_MARKER_RE.match(stripped):
            detected_points.append(
                {
                    "line": line_number,
                    "label": point_match.group("label"),
                    "kind": "point",
                    "article_number": current_article_number,
                    "text": stripped,
                }
            )

    return {
        "detected_chapters": detected_chapters,
        "detected_sections": detected_sections,
        "detected_articles": detected_articles,
        "detected_clauses": detected_clauses,
        "detected_points": detected_points,
    }


def detect_repeated_headers_or_footers(lines: Sequence[str]) -> list[dict[str, Any]]:
    occurrences: dict[str, dict[str, Any]] = {}

    for index, line in enumerate(lines):
        stripped = normalize_spaces(strip_marker_prefix(line))
        if (
            len(stripped) < 8
            or is_marker_line(stripped)
            or "|" in stripped
            or not looks_like_repeated_header_footer_candidate(stripped)
        ):
            continue
        normalized = normalized_line_key(stripped)
        if not normalized or normalized.isdigit():
            continue
        current = occurrences.setdefault(
            normalized,
            {"sample": stripped, "count": 0, "lines": []},
        )
        current["count"] += 1
        if len(current["lines"]) < 10:
            current["lines"].append(index + 1)

    repeated: list[dict[str, Any]] = []
    for value in occurrences.values():
        if value["count"] >= 3:
            repeated.append(
                {
                    "type": "repeated_header_footer",
                    "line": value["lines"][0],
                    "sample": preview(str(value["sample"])),
                    "message": "Line repeats at least three times and may be a retained header or footer.",
                    "count": value["count"],
                    "lines": value["lines"],
                }
            )

    return repeated


def looks_like_repeated_header_footer_candidate(line: str) -> bool:
    normalized = normalize_for_matching(line)
    if any(
        hint in normalized
        for hint in (
            "cong hoa xa hoi chu nghia viet nam",
            "doc lap - tu do - hanh phuc",
            "quoc hoi",
            "chinh phu",
            "bo lao dong",
            "cong bao",
            "trang ",
            "page ",
        )
    ):
        return True
    letters = [character for character in line if character.isalpha()]
    return bool(letters) and sum(character.isupper() for character in letters) / len(letters) > 0.85


def detect_possible_ocr_errors(lines: Sequence[str]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    for index, line in enumerate(lines):
        stripped = strip_marker_prefix(line)
        line_number = index + 1
        if not stripped:
            continue

        if "\ufffd" in stripped:
            append_issue(
                issues,
                issue_type="replacement_character",
                line=line_number,
                sample=stripped,
                message="Unicode replacement character found.",
            )

        if re.search(r"Ä|á»|Æ", stripped):
            append_issue(
                issues,
                issue_type="possible_mojibake",
                line=line_number,
                sample=stripped,
                message="Line contains common UTF-8 mojibake fragments.",
            )

        normalized_no_case = stripped.lower()
        for joined_word in JOINED_WORD_PATTERNS:
            if joined_word in normalized_no_case:
                append_issue(
                    issues,
                    issue_type="joined_word",
                    line=line_number,
                    sample=stripped,
                    message=f"Possible missing whitespace around '{joined_word}'.",
                )

        if (
            has_suspicious_letter_digit_join(stripped)
            and not LEGAL_SIGNATURE_RE.search(stripped)
            and not SUBPOINT_MARKER_RE.match(stripped)
            and "QH" not in stripped
        ):
            append_issue(
                issues,
                issue_type="letter_digit_join",
                line=line_number,
                sample=stripped,
                message="A letter and digit are joined without whitespace.",
            )

        if re.search(r"\bkhoản\s+l(?=\b|\d)", stripped, re.IGNORECASE):
            append_issue(
                issues,
                issue_type="digit_letter_confusion",
                line=line_number,
                sample=stripped,
                message="Possible OCR confusion between digit '1' and letter 'l' after 'khoản'.",
                expected="khoản 1",
            )

        if re.search(r"\bkhoán\s+\d+\s+Điều\b", stripped, re.IGNORECASE):
            append_issue(
                issues,
                issue_type="distorted_word",
                line=line_number,
                sample=stripped,
                message="Possible OCR substitution 'khoán' where a legal reference likely means 'khoản'.",
                expected="khoản",
            )

        for bad, expected in DISTORTED_TERM_PATTERNS:
            if bad.lower() in normalized_no_case:
                append_issue(
                    issues,
                    issue_type="distorted_word",
                    line=line_number,
                    sample=stripped,
                    message=f"Possible distorted legal term '{bad}'.",
                    expected=expected,
                )

        if chapter_match := ANY_CHAPTER_LINE_RE.match(stripped):
            label = chapter_match.group("label")
            if (label[:1].isdigit() or label[:1].isupper()) and not VALID_CHAPTER_LABEL_RE.match(label):
                append_issue(
                    issues,
                    issue_type="wrong_chapter_heading",
                    line=line_number,
                    sample=stripped,
                    message="Chapter marker has a non-Roman/non-numeric label.",
                )

    issues.extend(detect_repeated_headers_or_footers(lines))
    return issues[:MAX_REPORTED_ISSUES]


def has_suspicious_letter_digit_join(line: str) -> bool:
    for match in LETTER_DIGIT_JOIN_RE.finditer(line):
        prefix = match.group("prefix")
        if prefix:
            if prefix.lower() in MEASUREMENT_SUFFIXES:
                continue
            if any(character.isupper() for character in prefix) and len(prefix) <= 5:
                continue
            return True

        suffix = match.group("suffix")
        if not suffix:
            continue
        normalized_suffix = suffix.lower()
        if normalized_suffix in MEASUREMENT_SUFFIXES:
            continue
        if suffix.isupper() or (any(character.isupper() for character in suffix) and len(suffix) <= 4):
            continue
        return True
    return False


def detect_possible_broken_headings(lines: Sequence[str]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    for index, line in enumerate(lines):
        stripped = strip_marker_prefix(line)
        if not stripped:
            continue
        line_number = index + 1

        if chapter_match := ANY_CHAPTER_LINE_RE.match(stripped):
            label = chapter_match.group("label")
            if (label[:1].isdigit() or label[:1].isupper()) and not VALID_CHAPTER_LABEL_RE.match(label):
                append_issue(
                    issues,
                    issue_type="wrong_chapter_heading",
                    line=line_number,
                    sample=stripped,
                    message="Chapter heading may contain OCR noise.",
                )

        if not re.match(r"^Điều\s+\d", stripped):
            continue

        article_match = ARTICLE_LINE_RE.match(stripped)
        if not article_match:
            append_issue(
                issues,
                issue_type="malformed_article_heading",
                line=line_number,
                sample=stripped,
                message="Line starts with 'Điều' but does not match the expected article heading format.",
            )
            continue

        title = article_match.group("title").strip()
        if not title:
            append_issue(
                issues,
                issue_type="missing_article_title",
                line=line_number,
                sample=stripped,
                message="Article marker has no title on the same line.",
            )
            continue

        next_line = find_next_non_empty_line(lines, index)
        if next_line and looks_like_heading_continuation(title, next_line[1]):
            append_issue(
                issues,
                issue_type="article_heading_continuation",
                line=line_number,
                sample=f"{stripped} / {strip_marker_prefix(next_line[1])}",
                message="Article heading appears to continue on the next line.",
            )

    return issues


def iter_article_spans(lines: Sequence[str]) -> Iterable[ArticleSpan]:
    starts: list[tuple[int, re.Match[str]]] = []
    for index, line in enumerate(lines):
        stripped = strip_marker_prefix(line)
        if match := ARTICLE_LINE_RE.match(stripped):
            starts.append((index, match))

    for position, (start_index, match) in enumerate(starts):
        end_index = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        yield ArticleSpan(
            number=match.group("number"),
            title=repair_article_title(lines, start_index, match.group("title")),
            line=start_index + 1,
            start_index=start_index,
            end_index=end_index,
        )


def extract_labor_related_units(lines: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    normalized_keywords = {
        keyword: normalize_for_matching(keyword)
        for keyword in LABOR_KEYWORDS
    }
    articles: list[dict[str, Any]] = []

    for article in iter_article_spans(lines):
        article_text = "\n".join(lines[article.start_index : article.end_index])
        normalized_article = normalize_for_matching(article_text)
        matched_keywords = [
            keyword
            for keyword, normalized_keyword in normalized_keywords.items()
            if normalized_keyword in normalized_article
        ]
        if not matched_keywords:
            continue

        excerpt_line = next(
            (
                line
                for line in lines[article.start_index : article.end_index]
                if any(
                    normalized_keyword in normalize_for_matching(line)
                    for normalized_keyword in normalized_keywords.values()
                )
            ),
            lines[article.start_index],
        )
        articles.append(
            {
                "article_number": article.number,
                "article_title": article.title,
                "line": article.line,
                "matched_keywords": matched_keywords,
                "excerpt": preview(excerpt_line),
            }
        )

    sections: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        stripped = normalize_spaces(strip_marker_prefix(line))
        if not stripped:
            continue
        normalized = normalize_for_matching(stripped)
        if "lao dong" not in normalized:
            continue
        is_heading = stripped.isupper() or bool(CHAPTER_RE.match(stripped) or SECTION_RE.match(stripped))
        if is_heading:
            sections.append({"line": index + 1, "text": stripped})

    return {"articles": articles, "sections": sections}


def is_civil_procedure_code(document_id: str, document_title: str) -> bool:
    normalized = normalize_for_matching(f"{document_id} {document_title}")
    return "92 2015 qh13" in normalized.replace("-", " ") or "to tung dan su" in normalized


def build_document_warnings(
    *,
    document_id: str,
    document_title: str,
    markers: dict[str, list[dict[str, Any]]],
    possible_ocr_errors: Sequence[dict[str, Any]],
    possible_broken_headings: Sequence[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []

    if not markers["detected_articles"]:
        warnings.append("No article headings were detected.")
    if not markers["detected_chapters"]:
        warnings.append("No chapter headings were detected.")
    if possible_ocr_errors:
        warnings.append(f"{len(possible_ocr_errors)} possible OCR/text issue(s) detected.")
    if possible_broken_headings:
        warnings.append(f"{len(possible_broken_headings)} possible broken heading(s) detected.")

    if is_civil_procedure_code(document_id, document_title):
        warnings.append(
            "Bộ luật Tố tụng dân sự 2015 should not be fully included in the labor law graph; "
            "use only labor-related procedural provisions."
        )
        warnings.append(
            f"Suggested filtered curated file: {SUGGESTED_CIVIL_PROCEDURE_FILTERED_FILE}."
        )

    return warnings


def validate_curated_text(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    normalized_text = normalize_curated_text_for_validation(text)
    lines = normalized_text.splitlines()
    document_title = infer_curated_document_title(text, path)
    document_id = infer_document_id(path, text)
    document_type = infer_document_type(document_title, path, text)
    markers = detect_markers(lines)
    possible_ocr_errors = detect_possible_ocr_errors(lines)
    possible_broken_headings = detect_possible_broken_headings(lines)
    warnings = build_document_warnings(
        document_id=document_id,
        document_title=document_title,
        markers=markers,
        possible_ocr_errors=possible_ocr_errors,
        possible_broken_headings=possible_broken_headings,
    )

    document: dict[str, Any] = {
        "file_name": path.name,
        "path": str(path),
        "document_id": document_id,
        "document_title": document_title,
        "document_type": document_type,
        "character_count": len(text),
        **markers,
        "possible_ocr_errors": possible_ocr_errors,
        "possible_broken_headings": possible_broken_headings,
        "warnings": warnings,
    }

    if is_civil_procedure_code(document_id, document_title):
        document["labor_related_units"] = extract_labor_related_units(lines)
        document["suggested_filtered_file"] = SUGGESTED_CIVIL_PROCEDURE_FILTERED_FILE

    return document


def marker_count(document: dict[str, Any], key: str) -> int:
    value = document.get(key)
    return len(value) if isinstance(value, list) else 0


def validation_paths(
    input_dir: Path,
    expected_filenames: Sequence[str] = EXPECTED_CURATED_FILENAMES,
) -> tuple[list[Path], list[str]]:
    warnings: list[str] = []
    paths: list[Path] = []
    seen: set[Path] = set()

    for file_name in expected_filenames:
        path = input_dir / file_name
        if path.exists():
            paths.append(path)
            seen.add(path.resolve())
        else:
            warnings.append(f"Missing expected curated text file: {file_name}")

    for path in sorted(input_dir.glob("*.txt")):
        resolved = path.resolve()
        if resolved not in seen:
            paths.append(path)
            warnings.append(f"Found extra curated text file not in expected set: {path.name}")

    return paths, warnings


def validate_curated_directory(
    input_dir: Path,
    expected_filenames: Sequence[str] = EXPECTED_CURATED_FILENAMES,
) -> dict[str, Any]:
    paths, warnings = validation_paths(input_dir, expected_filenames)
    documents = [validate_curated_text(path) for path in paths]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "document_count": len(documents),
        "documents": documents,
        "warnings": warnings,
    }


def format_issue(issue: dict[str, Any]) -> str:
    line = issue.get("line")
    line_label = f"line {line}: " if line else ""
    expected = f" Expected: {issue['expected']}." if issue.get("expected") else ""
    return f"- {line_label}{issue.get('type')}: {issue.get('sample')} ({issue.get('message')}){expected}"


def render_markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Curated Legal Text Validation",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Input directory: `{report['input_dir']}`",
        f"Document count: `{report['document_count']}`",
        "",
    ]

    if report.get("warnings"):
        lines.extend(["## Run Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
        lines.append("")

    lines.extend(
        [
            "## Summary",
            "",
            "| document_id | title | type | chars | chapters | sections | articles | clauses | points | OCR/text issues | broken headings | warnings |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for document in report["documents"]:
        lines.append(
            "| {document_id} | {title} | {type} | {chars} | {chapters} | {sections} | "
            "{articles} | {clauses} | {points} | {ocr} | {broken} | {warnings} |".format(
                document_id=document["document_id"],
                title=document["document_title"].replace("|", "\\|"),
                type=document["document_type"],
                chars=document["character_count"],
                chapters=marker_count(document, "detected_chapters"),
                sections=marker_count(document, "detected_sections"),
                articles=marker_count(document, "detected_articles"),
                clauses=marker_count(document, "detected_clauses"),
                points=marker_count(document, "detected_points"),
                ocr=marker_count(document, "possible_ocr_errors"),
                broken=marker_count(document, "possible_broken_headings"),
                warnings=len(document.get("warnings", [])),
            )
        )

    for document in report["documents"]:
        lines.extend(
            [
                "",
                f"## {document['document_title']}",
                "",
                f"- File: `{document['file_name']}`",
                f"- Document ID: `{document['document_id']}`",
                f"- Document type: `{document['document_type']}`",
                f"- Character count: `{document['character_count']}`",
                (
                    "- Marker counts: "
                    f"chapters `{marker_count(document, 'detected_chapters')}`, "
                    f"sections `{marker_count(document, 'detected_sections')}`, "
                    f"articles `{marker_count(document, 'detected_articles')}`, "
                    f"clauses `{marker_count(document, 'detected_clauses')}`, "
                    f"points/subpoints `{marker_count(document, 'detected_points')}`"
                ),
            ]
        )

        if document.get("warnings"):
            lines.extend(["", "### Warnings", ""])
            lines.extend(f"- {warning}" for warning in document["warnings"])

        if document.get("possible_ocr_errors"):
            lines.extend(["", "### Possible OCR/Text Issues", ""])
            for issue in document["possible_ocr_errors"][:20]:
                lines.append(format_issue(issue))
            if len(document["possible_ocr_errors"]) > 20:
                lines.append(f"- ... {len(document['possible_ocr_errors']) - 20} more issue(s)")

        if document.get("possible_broken_headings"):
            lines.extend(["", "### Possible Broken Headings", ""])
            for issue in document["possible_broken_headings"][:20]:
                lines.append(format_issue(issue))
            if len(document["possible_broken_headings"]) > 20:
                lines.append(
                    f"- ... {len(document['possible_broken_headings']) - 20} more issue(s)"
                )

        labor_related = document.get("labor_related_units")
        if isinstance(labor_related, dict):
            articles = labor_related.get("articles", [])
            sections = labor_related.get("sections", [])
            lines.extend(["", "### Labor-Related Units", ""])
            lines.append(
                f"Suggested filtered file: `{document.get('suggested_filtered_file')}`"
            )
            lines.append(f"Detected labor-related articles: `{len(articles)}`")
            for article in articles[:30]:
                lines.append(
                    "- Điều {number} (line {line}): {title}".format(
                        number=article["article_number"],
                        line=article["line"],
                        title=article["article_title"],
                    )
                )
            if len(articles) > 30:
                lines.append(f"- ... {len(articles) - 30} more article(s)")
            if sections:
                lines.append(f"Detected labor-related section/headline candidates: `{len(sections)}`")
                for section in sections[:20]:
                    lines.append(f"- line {section['line']}: {section['text']}")
                if len(sections) > 20:
                    lines.append(f"- ... {len(sections) - 20} more section/headline(s)")

    lines.append("")
    return "\n".join(lines)


def write_validation_artifacts(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "curated_text_validation.json"
    markdown_path = output_dir / "curated_text_validation.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


__all__ = [
    "EXPECTED_CURATED_FILENAMES",
    "SUGGESTED_CIVIL_PROCEDURE_FILTERED_FILE",
    "detect_markers",
    "detect_possible_broken_headings",
    "detect_possible_ocr_errors",
    "extract_labor_related_units",
    "infer_curated_document_title",
    "infer_document_id",
    "render_markdown_report",
    "validate_curated_directory",
    "validate_curated_text",
    "write_validation_artifacts",
]
