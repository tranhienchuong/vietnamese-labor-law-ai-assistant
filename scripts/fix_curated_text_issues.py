from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CURATED_DIR = REPO_ROOT / "corpus" / "data" / "curated"
VALIDATION_DIR = REPO_ROOT / "artifacts" / "validation"
VALIDATION_JSON_PATH = VALIDATION_DIR / "curated_text_validation.json"
REPORT_JSON_PATH = VALIDATION_DIR / "curated_text_correction_report.json"
REPORT_MD_PATH = VALIDATION_DIR / "curated_text_correction_report.md"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class TextReplacement:
    file_name: str
    old: str
    new: str
    correction_type: str
    confidence: str = "high"


@dataclass(frozen=True)
class RegexReplacement:
    file_name: str
    pattern: str
    replacement: str
    correction_type: str
    confidence: str = "high"


@dataclass(frozen=True)
class HeadingMerge:
    file_name: str
    article_number: str
    continuation_count: int
    correction_type: str = "article_heading_merge"
    confidence: str = "high"
    body_split_marker: str | None = None
    normalize_article_punctuation: bool = False


TEXT_REPLACEMENTS: tuple[TextReplacement, ...] = (
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "khoán 2 Điều",
        "khoản 2 Điều",
        "ocr_legal_reference",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "và khoán 2",
        "và khoản 2",
        "ocr_legal_reference",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "khoản l",
        "khoản 1",
        "ocr_digit_letter_confusion",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "HỢP ĐÔNG",
        "HỢP ĐỒNG",
        "ocr_distorted_word",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "hợp đông",
        "hợp đồng",
        "ocr_distorted_word",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "HỢP ĐỎNG",
        "HỢP ĐỒNG",
        "ocr_distorted_word",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "HỢP ĐÒNG",
        "HỢP ĐỒNG",
        "ocr_distorted_word",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "CHÁM DỨT",
        "CHẤM DỨT",
        "ocr_distorted_word",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "4rong",
        "trong",
        "ocr_digit_letter_confusion",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "sản phâm",
        "sản phẩm",
        "ocr_distorted_word",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "Chương IH",
        "Chương III",
        "chapter_heading_correction",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "Chương VIH",
        "Chương VIII",
        "chapter_heading_correction",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "ChươngVII",
        "Chương VII",
        "chapter_heading_correction",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "ChươngXI:",
        "Chương XI",
        "chapter_heading_correction",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "Chương XI:",
        "Chương XI",
        "chapter_heading_correction",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "ChươngV",
        "Chương V",
        "chapter_heading_correction",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "ĐÓI VỚI",
        "ĐỐI VỚI",
        "ocr_distorted_heading_word",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "GIAM ĐỌC",
        "GIÁM ĐỐC",
        "ocr_distorted_heading_word",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "CÓ VÔN",
        "CÓ VỐN",
        "ocr_distorted_heading_word",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "TRÁCH NHIỆM VẬT CHÁT",
        "TRÁCH NHIỆM VẬT CHẤT",
        "ocr_distorted_heading_word",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "THƠI GIỚ NGHĨ NGƠI",
        "THỜI GIỜ NGHỈ NGƠI",
        "ocr_distorted_heading_word",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "ĐIÊỀU KHOẢN THỊ HÀNH",
        "ĐIỀU KHOẢN THI HÀNH",
        "ocr_distorted_heading_word",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "năm giữ",
        "nắm giữ",
        "ocr_distorted_word",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "tông số",
        "tổng số",
        "ocr_distorted_word",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "cỗ phần",
        "cổ phần",
        "ocr_distorted_word",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "cô phần",
        "cổ phần",
        "ocr_distorted_word",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "trợ câp",
        "trợ cấp",
        "ocr_distorted_word",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "trợ cập",
        "trợ cấp",
        "ocr_distorted_word",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "mât việc",
        "mất việc",
        "ocr_distorted_word",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "mật việc",
        "mất việc",
        "ocr_distorted_word",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "Tổ tụng",
        "Tố tụng",
        "ocr_distorted_word",
    ),
    TextReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        "thực biện",
        "thực hiện",
        "ocr_distorted_word",
    ),
    TextReplacement(
        "thong_tu_09_2020_tt_bldtbxh_clean.txt",
        "sửdụng",
        "sử dụng",
        "joined_word",
    ),
    TextReplacement(
        "thong_tu_09_2020_tt_bldtbxh_clean.txt",
        "chưa đủ13",
        "chưa đủ 13",
        "letter_digit_spacing",
    ),
)

REGEX_REPLACEMENTS: tuple[RegexReplacement, ...] = (
    RegexReplacement(
        "nghi_dinh_145_2020_nd_cp_clean.txt",
        r"^([`\"'\s\-–—_‹›]+)?Chương\s+([IVXLCDM]+)\s*[-–—]*$",
        r"Chương \2",
        "chapter_heading_cleanup",
    ),
)

HEADING_MERGES: tuple[HeadingMerge, ...] = (
    HeadingMerge("45_2019_QH14.txt", "43", 1),
    HeadingMerge("45_2019_QH14.txt", "80", 1),
    HeadingMerge("45_2019_QH14.txt", "175", 1),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "5", 3),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "6", 2),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "7", 1),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "10", 2),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "11", 2),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "19", 1),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "29", 1, normalize_article_punctuation=True),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "42", 1),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "48", 1),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "53", 1),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "61", 1),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "62", 1),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "65", 1),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "66", 1),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "67", 1),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "68", 1),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "78", 1),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "82", 1),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "85", 1),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "86", 1),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "87", 1),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "89", 1),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "104", 1),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "106", 1),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "107", 1),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "108", 1),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "112", 1),
    HeadingMerge("nghi_dinh_145_2020_nd_cp_clean.txt", "113", 1),
    HeadingMerge("thong_tu_09_2020_tt_bldtbxh_clean.txt", "4", 1),
    HeadingMerge("thong_tu_09_2020_tt_bldtbxh_clean.txt", "7", 1),
    HeadingMerge(
        "thong_tu_09_2020_tt_bldtbxh_clean.txt",
        "8",
        1,
        body_split_marker="Ban hành kèm theo",
    ),
    HeadingMerge("thong_tu_09_2020_tt_bldtbxh_clean.txt", "9", 1),
    HeadingMerge(
        "thong_tu_09_2020_tt_bldtbxh_clean.txt",
        "10",
        1,
        body_split_marker="Ban hành kèm theo",
    ),
    HeadingMerge("thong_tu_10_2020_tt_bldtbxh_clean.txt", "5", 1),
    HeadingMerge(
        "thong_tu_10_2020_tt_bldtbxh_clean.txt",
        "10",
        1,
        body_split_marker=(
            "Danh mục nghề, công việc có ảnh hưởng xấu tới chức năng sinh sản "
            "và nuôi con được"
        ),
    ),
    HeadingMerge("thong_tu_10_2020_tt_bldtbxh_clean.txt", "11", 1),
    HeadingMerge("bo_luat_92_2015_qh13_to_tung_dan_su_2015_clean.txt", "516", 1),
)


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def add_report_entry(
    corrections: list[dict[str, Any]],
    *,
    file_name: str,
    line_number: int | None,
    original_text: str,
    corrected_text: str,
    correction_type: str,
    confidence: str,
    manual_review_required: bool = False,
) -> None:
    entry = {
        "file_name": file_name,
        "line_number_before_correction": line_number,
        "original_text": original_text,
        "corrected_text": corrected_text,
        "correction_type": correction_type,
        "confidence": confidence,
        "manual_review_required": manual_review_required,
    }
    if entry not in corrections:
        corrections.append(entry)


def apply_text_replacements(lines: list[str], file_name: str, corrections: list[dict[str, Any]]) -> None:
    replacements = [replacement for replacement in TEXT_REPLACEMENTS if replacement.file_name == file_name]
    for replacement in replacements:
        for index, line in enumerate(lines):
            if replacement.old not in line:
                continue
            original = line
            corrected = line.replace(replacement.old, replacement.new)
            if corrected == original:
                continue
            lines[index] = corrected
            add_report_entry(
                corrections,
                file_name=file_name,
                line_number=index + 1,
                original_text=original,
                corrected_text=corrected,
                correction_type=replacement.correction_type,
                confidence=replacement.confidence,
            )


def apply_regex_replacements(lines: list[str], file_name: str, corrections: list[dict[str, Any]]) -> None:
    replacements = [replacement for replacement in REGEX_REPLACEMENTS if replacement.file_name == file_name]
    for replacement in replacements:
        pattern = re.compile(replacement.pattern)
        for index, line in enumerate(lines):
            corrected = pattern.sub(replacement.replacement, line)
            if corrected == line:
                continue
            original = line
            lines[index] = corrected
            add_report_entry(
                corrections,
                file_name=file_name,
                line_number=index + 1,
                original_text=original,
                corrected_text=corrected,
                correction_type=replacement.correction_type,
                confidence=replacement.confidence,
            )


def find_article_heading_index(
    lines: list[str],
    article_number: str,
    *,
    allow_comma: bool = False,
) -> int | None:
    punctuation = r"[.,]" if allow_comma else r"\."
    pattern = re.compile(rf"^Điều\s+{re.escape(article_number)}{punctuation}\s")
    for index, line in enumerate(lines):
        if pattern.match(line.strip()):
            return index
    return None


def collect_heading_continuations(
    lines: list[str],
    start_index: int,
    continuation_count: int,
) -> tuple[list[int], list[str]]:
    indexes: list[int] = []
    values: list[str] = []
    index = start_index + 1
    while index < len(lines) and len(values) < continuation_count:
        if not lines[index].strip():
            index += 1
            continue
        indexes.append(index)
        values.append(lines[index].strip())
        index += 1
    return indexes, values


def split_heading_body(heading_text: str, marker: str | None) -> tuple[str, str | None]:
    if not marker or marker not in heading_text:
        return heading_text, None
    before, after = heading_text.split(marker, 1)
    return before.strip(), f"{marker}{after}".strip()


def apply_heading_merges(lines: list[str], file_name: str, corrections: list[dict[str, Any]]) -> None:
    merges = [merge for merge in HEADING_MERGES if merge.file_name == file_name]
    for merge in merges:
        heading_index = find_article_heading_index(
            lines,
            merge.article_number,
            allow_comma=merge.normalize_article_punctuation,
        )
        if heading_index is None:
            add_report_entry(
                corrections,
                file_name=file_name,
                line_number=None,
                original_text=f"Điều {merge.article_number}",
                corrected_text="Article heading not found; no correction applied.",
                correction_type="article_heading_merge_missing",
                confidence="low",
                manual_review_required=True,
            )
            continue

        continuation_indexes, continuation_values = collect_heading_continuations(
            lines,
            heading_index,
            merge.continuation_count,
        )
        if len(continuation_values) != merge.continuation_count:
            add_report_entry(
                corrections,
                file_name=file_name,
                line_number=heading_index + 1,
                original_text=lines[heading_index],
                corrected_text="Expected continuation line not found; no correction applied.",
                correction_type="article_heading_merge_incomplete",
                confidence="low",
                manual_review_required=True,
            )
            continue

        original_lines = [lines[heading_index], *continuation_values]
        heading_line = lines[heading_index].strip()
        if merge.normalize_article_punctuation:
            heading_line = re.sub(
                rf"^(Điều\s+{re.escape(merge.article_number)}),",
                r"\1.",
                heading_line,
            )
        joined = normalize_spaces(" ".join([heading_line, *continuation_values]))
        joined_heading, body_text = split_heading_body(joined, merge.body_split_marker)
        lines[heading_index] = joined_heading

        for index in sorted(continuation_indexes, reverse=True):
            del lines[index]

        if body_text:
            lines.insert(heading_index + 1, "")
            lines.insert(heading_index + 2, body_text)
            corrected_text = f"{joined_heading}\n\n{body_text}"
        else:
            corrected_text = joined_heading

        add_report_entry(
            corrections,
            file_name=file_name,
            line_number=heading_index + 1,
            original_text="\n".join(original_lines),
            corrected_text=corrected_text,
            correction_type=merge.correction_type,
            confidence=merge.confidence,
        )


def add_manual_review_entries(corrections: list[dict[str, Any]]) -> None:
    if not VALIDATION_JSON_PATH.exists():
        return
    validation_report = json.loads(VALIDATION_JSON_PATH.read_text(encoding="utf-8"))
    for document in validation_report.get("documents", []):
        file_name = document.get("file_name")
        if not file_name:
            continue
        path = CURATED_DIR / file_name
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for issue in document.get("possible_ocr_errors", []):
            if issue.get("type") != "repeated_header_footer":
                continue
            issue_lines = issue.get("lines") or [issue.get("line")]
            for line_number in issue_lines:
                if not isinstance(line_number, int) or line_number < 1 or line_number > len(lines):
                    continue
                original = lines[line_number - 1]
                add_report_entry(
                    corrections,
                    file_name=file_name,
                    line_number=line_number,
                    original_text=original,
                    corrected_text=original,
                    correction_type="manual_review_repeated_header_footer_not_removed",
                    confidence="low",
                    manual_review_required=True,
                )
        corrected_heading_numbers = {
            merge.article_number
            for merge in HEADING_MERGES
            if merge.file_name == file_name
        }
        for issue in document.get("possible_broken_headings", []):
            line_number = issue.get("line")
            if not isinstance(line_number, int) or line_number < 1 or line_number > len(lines):
                continue
            original = lines[line_number - 1]
            article_match = re.match(r"^Điều\s+(\d+[A-Za-z]?)[.,]\s", original.strip())
            if article_match and article_match.group(1) in corrected_heading_numbers:
                continue
            if any(
                replacement.file_name == file_name and replacement.old in original
                for replacement in TEXT_REPLACEMENTS
                if replacement.correction_type == "chapter_heading_correction"
            ):
                continue
            add_report_entry(
                corrections,
                file_name=file_name,
                line_number=line_number,
                original_text=original,
                corrected_text=original,
                correction_type="manual_review_possible_broken_heading_not_changed",
                confidence="low",
                manual_review_required=True,
            )


def write_file(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def summarize_by_file(corrections: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for correction in corrections:
        file_name = correction["file_name"]
        item = summary.setdefault(file_name, {"applied": 0, "manual_review": 0})
        if correction["manual_review_required"]:
            item["manual_review"] += 1
        else:
            item["applied"] += 1
    return summary


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Curated Text Correction Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Curated directory: `{report['curated_dir']}`",
        "",
        "## Summary",
        "",
        "| file | applied corrections | manual review items |",
        "| --- | ---: | ---: |",
    ]
    for file_name, item in sorted(report["summary_by_file"].items()):
        lines.append(f"| {file_name} | {item['applied']} | {item['manual_review']} |")

    lines.extend(["", "## Corrections", ""])
    for correction in report["corrections"]:
        review = "yes" if correction["manual_review_required"] else "no"
        lines.extend(
            [
                f"### {correction['file_name']}:{correction['line_number_before_correction']}",
                "",
                f"- Type: `{correction['correction_type']}`",
                f"- Confidence: `{correction['confidence']}`",
                f"- Manual review required: `{review}`",
                "- Original:",
                "```text",
                correction["original_text"],
                "```",
                "- Corrected:",
                "```text",
                correction["corrected_text"],
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    corrections: list[dict[str, Any]] = []
    file_names = sorted(
        {
            replacement.file_name
            for replacement in TEXT_REPLACEMENTS
        }
        | {replacement.file_name for replacement in REGEX_REPLACEMENTS}
        | {merge.file_name for merge in HEADING_MERGES}
    )

    for file_name in file_names:
        path = CURATED_DIR / file_name
        lines = path.read_text(encoding="utf-8").splitlines()
        apply_text_replacements(lines, file_name, corrections)
        apply_regex_replacements(lines, file_name, corrections)
        apply_heading_merges(lines, file_name, corrections)
        write_file(path, lines)

    add_manual_review_entries(corrections)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "curated_dir": str(CURATED_DIR),
        "correction_count": sum(1 for item in corrections if not item["manual_review_required"]),
        "manual_review_count": sum(1 for item in corrections if item["manual_review_required"]),
        "summary_by_file": summarize_by_file(corrections),
        "corrections": corrections,
    }
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD_PATH.write_text(render_markdown(report), encoding="utf-8")

    print(
        f"Applied {report['correction_count']} correction(s); "
        f"recorded {report['manual_review_count']} manual review item(s)."
    )
    print(f"JSON report: {REPORT_JSON_PATH}")
    print(f"Markdown report: {REPORT_MD_PATH}")


if __name__ == "__main__":
    main()
