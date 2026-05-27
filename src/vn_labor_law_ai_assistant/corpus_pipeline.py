from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Sequence
import unicodedata

import fitz
from pypdf import PdfReader


ARTICLE_RE = re.compile(r"^Điều\s+(?P<number>\d+[A-Za-z]?)\.\s*(?P<title>.+)$")
PART_RE = re.compile(r"^Phần\s+(?P<number>.+)$", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^Chương\s+(?P<number>[IVXLCDM0-9]+)\.?\s*(?P<title>.*)$", re.IGNORECASE)
SECTION_RE = re.compile(r"^Mục\s+(?P<number>[IVXLCDM0-9]+)\.?\s*(?P<title>.*)$", re.IGNORECASE)
GROUP_RE = re.compile(r"^[IVXLCDM]+\.\s+.+$")
PAGE_ONLY_RE = re.compile(r"^\d+$")
SEPARATOR_RE = re.compile(r"^[=\-]{3,}$")
SOURCE_HINT_RE = re.compile(r"^Nguồn:\s*(?P<title>.+)$", re.IGNORECASE)
CLAUSE_SPLIT_RE = re.compile(r"(?m)^(?=\d+\.\s)")
SUBPOINT_SPLIT_RE = re.compile(r"(?m)^(?=[a-zđ]\.\d+\)\s)")
POINT_SPLIT_RE = re.compile(r"(?m)^(?=[a-zđ]\)\s)")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;])\s+")
CLAUSE_START_RE = re.compile(r"^(?P<label>\d+)\.\s")
SUBPOINT_START_RE = re.compile(r"^(?P<label>[a-zđ]\.\d+)\)\s", re.IGNORECASE)
POINT_START_RE = re.compile(r"^(?P<label>[a-zđ])\)\s", re.IGNORECASE)
LEGAL_MARKER_BOUNDARY_RE = re.compile(r"(?m)^(?:[a-zđ]\.\d+\)\s|[a-zđ]\)\s|\d+\.\s)", re.IGNORECASE)
SEQUENTIAL_CHUNK_ARTICLES = {"219"}
NORMAL_LEGAL_CHUNK_TYPES = {
    "article_full",
    "article_intro",
    "article_sequential",
    "clause",
    "clause_part",
}
CURATED_LEGAL_CHUNK_FILENAMES = (
    "45_2019_QH14.txt",
    "nghi_dinh_145_2020_nd_cp_clean.txt",
    "nghi_dinh_135_2020_nd_cp_clean.txt",
    "thong_tu_09_2020_tt_bldtbxh_clean.txt",
    "thong_tu_10_2020_tt_bldtbxh_clean.txt",
    "92_2015_QH13_labor_only.txt",
)
FULL_BLTTDS_FILENAME = "bo_luat_92_2015_qh13_to_tung_dan_su_2015_clean.txt"
VERY_SHORT_CHUNK_THRESHOLD = 20
VERY_LONG_CHUNK_THRESHOLD = 3000
CANONICAL_DOCUMENT_METADATA_BY_FILENAME = {
    "45_2019_QH14.txt": {
        "document_id": "45-2019-qh14",
        "document_title": "Bộ luật Lao động 2019",
        "document_type": "bo_luat",
    },
    "nghi_dinh_145_2020_nd_cp_clean.txt": {
        "document_id": "nghi-dinh-145-2020-nd-cp",
        "document_title": "Nghị định 145/2020/NĐ-CP",
        "document_type": "nghi_dinh",
    },
    "nghi_dinh_135_2020_nd_cp_clean.txt": {
        "document_id": "nghi-dinh-135-2020-nd-cp",
        "document_title": "Nghị định 135/2020/NĐ-CP",
        "document_type": "nghi_dinh",
    },
    "thong_tu_09_2020_tt_bldtbxh_clean.txt": {
        "document_id": "thong-tu-09-2020-tt-bldtbxh",
        "document_title": "Thông tư 09/2020/TT-BLĐTBXH",
        "document_type": "thong_tu",
    },
    "thong_tu_10_2020_tt_bldtbxh_clean.txt": {
        "document_id": "thong-tu-10-2020-tt-bldtbxh",
        "document_title": "Thông tư 10/2020/TT-BLĐTBXH",
        "document_type": "thong_tu",
    },
    "92_2015_QH13_labor_only.txt": {
        "document_id": "92-2015-qh13-labor-only",
        "document_title": "Bộ luật Tố tụng dân sự 2015 - phần liên quan lao động",
        "document_type": "bo_luat",
    },
}
APPENDIX_HEADING_RE = re.compile(r"^PHỤ\s+LỤC(?:\s+(?P<label>[IVXLCDM0-9A-Z]+))?\b", re.IGNORECASE)
FORM_HEADING_RE = re.compile(r"^Mẫu\s+số\s+(?P<number>\d+)\s*/?\s*(?P<code>PL[IVXLCDM]+)", re.IGNORECASE)
TABLE_HEADING_RE = re.compile(r"^BẢNG\b", re.IGNORECASE)
CATALOG_HEADING_RE = re.compile(r"^DANH\s+MỤC\b", re.IGNORECASE)
ADMIN_FOOTER_RE = re.compile(
    r"^(?:Nơi nhận\b|TM\.\s|KT\.\s)",
    re.IGNORECASE,
)

TOPIC_RULES = {
    "cham_dut_hop_dong_lao_dong": ["cham dut hop dong lao dong"],
    "don_phuong_cham_dut": ["don phuong cham dut"],
    "bao_truoc": ["bao truoc", "thoi han bao truoc"],
    "tro_cap": ["tro cap thoi viec", "tro cap mat viec"],
    "ky_luat_sa_thai": ["sa thai", "ky luat"],
    "thay_doi_co_cau_kinh_te": ["thay doi co cau", "ly do kinh te", "phuong an su dung lao dong"],
    "tam_hoan_hop_dong": ["tam hoan thuc hien hop dong lao dong"],
    "hop_dong_lao_dong": ["hop dong lao dong", "giao ket hop dong lao dong", "loai hop dong lao dong"],
    "dao_tao_nghe": ["dao tao nghe", "chi phi dao tao"],
    "bao_ve_thai_san": ["thai san", "mang thai", "nuoi con duoi 12 thang"],
}

ACTOR_RULES = {
    "nguoi_lao_dong": ["nguoi lao dong"],
    "nguoi_su_dung_lao_dong": ["nguoi su dung lao dong", "doanh nghiep"],
    "lao_dong_nu": ["lao dong nu", "mang thai", "nuoi con duoi 12 thang"],
    "nguoi_lao_dong_nuoc_ngoai": ["nguoi nuoc ngoai", "la nguoi nuoc ngoai"],
    "to_chuc_dai_dien_nguoi_lao_dong": ["to chuc dai dien nguoi lao dong"],
}

ISSUE_TYPE_RULES = {
    "can_cu_cham_dut": ["cac truong hop cham dut hop dong lao dong", "cham dut hop dong lao dong"],
    "thoi_han_bao_truoc": ["bao truoc", "thoi han bao truoc"],
    "tro_cap_thoi_viec": ["tro cap thoi viec"],
    "tro_cap_mat_viec": ["tro cap mat viec"],
    "boi_thuong": ["boi thuong"],
    "trai_phap_luat": ["trai phap luat"],
    "sa_thai": ["sa thai"],
    "tam_hoan_hop_dong": ["tam hoan"],
    "thong_bao_cham_dut": ["thong bao cham dut"],
    "dao_tao": ["dao tao nghe", "chi phi dao tao"],
    "thay_doi_co_cau_kinh_te": ["thay doi co cau", "ly do kinh te", "phuong an su dung lao dong"],
    "bao_ve_thai_san": ["thai san", "mang thai", "nuoi con duoi 12 thang"],
    "quyen_don_phuong_cham_dut": ["quyen don phuong cham dut"],
    "giao_ket_hop_dong": [
        "giao ket hop dong lao dong",
        "hop dong lao dong la su thoa thuan",
        "noi dung chu yeu cua hop dong lao dong",
    ],
    "sua_doi_bo_sung_hop_dong": [
        "sua doi, bo sung hop dong lao dong",
        "sua doi bo sung hop dong lao dong",
    ],
    "thong_tin_giao_ket": ["nghia vu cung cap thong tin khi giao ket hop dong lao dong"],
    "loai_hop_dong": ["loai hop dong lao dong"],
    "dieu_chuyen_cong_viec": [
        "chuyen nguoi lao dong lam cong viec khac",
        "chuyen nguoi lao dong lam cong viec khac so voi hop dong lao dong",
    ],
    "doi_thoai_tai_noi_lam_viec": [
        "doi thoai tai noi lam viec",
        "to chuc doi thoai tai noi lam viec",
        "noi dung doi thoai tai noi lam viec",
    ],
    "noi_quy_lao_dong": [
        "noi quy lao dong",
        "dang ky noi quy lao dong",
        "ho so dang ky noi quy lao dong",
        "hieu luc cua noi quy lao dong",
        "noi dung noi quy lao dong",
    ],
    "xu_ly_ky_luat_lao_dong": [
        "ky luat lao dong",
        "hinh thuc xu ly ky luat lao dong",
        "tham quyen xu ly ky luat lao dong",
        "nguyen tac, trinh tu, thu tuc xu ly ky luat lao dong",
        "thoi hieu xu ly ky luat lao dong",
        "tam dinh chi cong viec",
        "xoa ky luat, giam thoi han chap hanh ky luat lao dong",
    ],
    "nghia_vu_khi_cham_dut": [
        "nghia vu cua nguoi lao dong khi don phuong cham dut",
        "nghia vu cua nguoi su dung lao dong khi don phuong cham dut",
        "trach nhiem khi cham dut hop dong lao dong",
    ],
}

KNOWN_JOIN_FIXES = {
    "Nghịđịnh": "Nghị định",
    "Bộluật": "Bộ luật",
    "Trợcấp": "Trợ cấp",
    "sửdụng": "sử dụng",
    "kỹthuật": "kỹ thuật",
    "tổlái": "tổ lái",
    "ởnước": "ở nước",
    "từđủ": "từ đủ",
    "bịmất": "bị mất",
    "nghềnghiệp": "nghề nghiệp",
    "sốngành": "số ngành",
}
FALLBACK_QH_TITLE_RE = re.compile(
    r"(?P<number>\d+)[_\-/ ]+(?P<year>\d{4})[_\-/ ]+qh(?P<session>\d+)",
    re.IGNORECASE,
)
FALLBACK_ND_CP_TITLE_RE = re.compile(
    r"(?P<number>\d+)[_\-/ ]+(?P<year>\d{4})[_\-/ ]+nd[_\-/ ]+cp",
    re.IGNORECASE,
)
TITLE_QH_SIGNATURE_RE = re.compile(
    r"(?P<number>\d+)\s*/?\s*(?P<year>\d{4})\s*/?\s*qh\s*(?P<session>\d+)",
    re.IGNORECASE,
)
TITLE_ND_CP_SIGNATURE_RE = re.compile(
    r"(?P<number>\d+)\s*/?\s*(?P<year>\d{4})\s*/?\s*nd\s*-?\s*cp",
    re.IGNORECASE,
)
PARTIAL_EXTRACT_HINTS = (
    "bo du lieu trich xuat",
    "pham vi trich xuat",
    "danh sach dieu da trich",
)


@dataclass
class PageRecord:
    page_number: int
    text: str


@dataclass
class SectionRecord:
    section_id: str
    heading: str
    article_number: str | None
    article_title: str | None
    chapter_heading: str | None
    section_heading: str | None
    source_pages: list[int]
    text: str
    part_number: str | None = None
    part_heading: str | None = None


@dataclass
class LegalPointGroup:
    point_ref: str
    parts: list[str]


@dataclass
class LegalClauseGroup:
    clause_ref: str | None
    intro_parts: list[str]
    points: list[LegalPointGroup]


@dataclass(frozen=True)
class CuratedTextCandidate:
    path: Path
    document_title: str
    canonical_title_key: str
    cleaned_text_length: int
    is_partial_extract: bool


@dataclass(frozen=True)
class CanonicalDocumentMetadata:
    document_id: str
    document_title: str
    document_type: str


@dataclass(frozen=True)
class AppendixBlock:
    chunk_id: str
    appendix_id: str
    appendix_heading: str
    chunk_type: str
    text: str


def slugify_text(value: str) -> str:
    value = value.replace("đ", "d").replace("Đ", "D")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return value or "document"


def normalize_for_matching(value: str) -> str:
    value = value.replace("đ", "d").replace("Đ", "D")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_extracted_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00ad", "")
    text = text.replace("\xa0", " ")
    text = text.replace(";", ";")
    text = text.replace("\r", "\n")

    for bad, good in KNOWN_JOIN_FIXES.items():
        text = text.replace(bad, good)

    text = re.sub(r"(?<=[A-Za-zÀ-ỹ])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[A-Za-zÀ-ỹ])", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        if PAGE_ONLY_RE.match(line) or SEPARATOR_RE.match(line):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" +([,.;:?!])", r"\1", cleaned)
    return cleaned


def split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def build_page_records(pdf_path: Path) -> tuple[list[PageRecord], dict[str, int]]:
    doc = fitz.open(pdf_path)
    reader = PdfReader(str(pdf_path))

    pages: list[PageRecord] = []
    stats = {"page_count": doc.page_count, "text_page_count": 0}

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        blocks = page.get_text("blocks", sort=True)
        block_texts = [" ".join(block[4].split()) for block in blocks if block[4].strip()]
        page_text = "\n\n".join(block_texts).strip()

        if not page_text:
            fallback_text = (reader.pages[page_index].extract_text() or "").strip()
            page_text = normalize_extracted_text(fallback_text)

        if page_text:
            page_text = normalize_extracted_text(page_text)
            stats["text_page_count"] += 1
        pages.append(PageRecord(page_number=page_index + 1, text=page_text))

    return pages, stats


def build_cleaned_text(page_records: list[PageRecord]) -> str:
    paragraphs: list[str] = []
    for record in page_records:
        if not record.text:
            continue
        paragraphs.extend(split_paragraphs(record.text))
    return "\n\n".join(paragraphs).strip()


def build_page_records_from_text(text: str) -> list[PageRecord]:
    cleaned = normalize_extracted_text(text)
    return [PageRecord(page_number=1, text=cleaned)] if cleaned else []


def infer_document_title(text: str, fallback_title: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        source_match = SOURCE_HINT_RE.match(stripped)
        if source_match:
            return source_match.group("title").strip()
    if inferred_title := infer_legal_title_from_fallback(fallback_title):
        return inferred_title
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if ARTICLE_RE.match(stripped) or SECTION_RE.match(stripped) or CHAPTER_RE.match(stripped) or GROUP_RE.match(stripped):
            continue
        if stripped.isupper() and "BỘ DỮ LIỆU" in stripped:
            continue
        return stripped
    return fallback_title


def infer_legal_title_from_fallback(fallback_title: str) -> str | None:
    if qh_match := FALLBACK_QH_TITLE_RE.search(fallback_title):
        return (
            f"Bộ luật số {qh_match.group('number')}/{qh_match.group('year')}/"
            f"QH{qh_match.group('session')}"
        )
    if nd_match := FALLBACK_ND_CP_TITLE_RE.search(fallback_title):
        return f"Nghị định {nd_match.group('number')}/{nd_match.group('year')}/NĐ-CP"
    return None


def is_partial_curated_extract(text: str) -> bool:
    normalized_text = normalize_for_matching(text)
    return any(hint in normalized_text for hint in PARTIAL_EXTRACT_HINTS)


def canonical_document_identity_key(document_title: str) -> str:
    normalized_title = normalize_for_matching(document_title)
    if qh_match := TITLE_QH_SIGNATURE_RE.search(normalized_title):
        return (
            f"qh:{qh_match.group('number')}/{qh_match.group('year')}/"
            f"qh{qh_match.group('session')}"
        )
    if nd_match := TITLE_ND_CP_SIGNATURE_RE.search(normalized_title):
        return f"nd:{nd_match.group('number')}/{nd_match.group('year')}/nd-cp"
    return normalized_title


def inspect_curated_text_candidate(text_path: Path) -> CuratedTextCandidate:
    source_text = text_path.read_text(encoding="utf-8")
    page_records = build_page_records_from_text(source_text)
    cleaned_text = build_cleaned_text(page_records)
    document_title = infer_document_title(cleaned_text, fallback_title=text_path.stem)
    return CuratedTextCandidate(
        path=text_path,
        document_title=document_title,
        canonical_title_key=canonical_document_identity_key(document_title),
        cleaned_text_length=len(cleaned_text),
        is_partial_extract=is_partial_curated_extract(cleaned_text),
    )


def select_curated_text_sources(curated_text_paths: Sequence[Path]) -> tuple[list[Path], list[str]]:
    skipped_full_blttds = [
        path for path in curated_text_paths if path.name == FULL_BLTTDS_FILENAME
    ]
    curated_text_paths = [
        path for path in curated_text_paths if path.name != FULL_BLTTDS_FILENAME
    ]
    grouped_candidates: dict[str, list[CuratedTextCandidate]] = {}
    for path in curated_text_paths:
        candidate = inspect_curated_text_candidate(path)
        grouped_candidates.setdefault(candidate.canonical_title_key, []).append(candidate)

    selected_paths: list[Path] = []
    warnings: list[str] = []
    if skipped_full_blttds:
        warnings.append(
            "Skipped full Bộ luật Tố tụng dân sự source; use "
            "92_2015_QH13_labor_only.txt for labor graph/index chunking."
        )

    for candidates in grouped_candidates.values():
        preferred = sorted(
            candidates,
            key=lambda candidate: (
                candidate.is_partial_extract,
                -candidate.cleaned_text_length,
                candidate.path.as_posix(),
            ),
        )[0]
        selected_paths.append(preferred.path)

        skipped_candidates = [candidate for candidate in candidates if candidate.path != preferred.path]
        if skipped_candidates:
            skipped_paths = ", ".join(candidate.path.name for candidate in skipped_candidates)
            warnings.append(
                f"Skipped duplicate curated sources for '{preferred.document_title}': "
                f"kept {preferred.path.name}, skipped {skipped_paths}."
            )

    return sorted(selected_paths), warnings


def resolve_canonical_document_metadata(text_path: Path, cleaned_text: str) -> CanonicalDocumentMetadata:
    if metadata := CANONICAL_DOCUMENT_METADATA_BY_FILENAME.get(text_path.name):
        return CanonicalDocumentMetadata(
            document_id=metadata["document_id"],
            document_title=metadata["document_title"],
            document_type=metadata["document_type"],
        )
    return CanonicalDocumentMetadata(
        document_id=slugify_text(text_path.stem),
        document_title=infer_document_title(cleaned_text, fallback_title=text_path.stem),
        document_type="unknown",
    )


def is_appendix_boundary(line: str) -> bool:
    stripped = line.strip()
    return bool(
        APPENDIX_HEADING_RE.match(stripped)
        or FORM_HEADING_RE.match(stripped)
        or TABLE_HEADING_RE.match(stripped)
        or CATALOG_HEADING_RE.match(stripped)
    )


def find_last_article_line(lines: Sequence[str]) -> int | None:
    last_article_index: int | None = None
    for index, line in enumerate(lines):
        if ARTICLE_RE.match(line.strip()):
            last_article_index = index
    return last_article_index


def find_admin_footer_start(lines: Sequence[str], start_index: int) -> int | None:
    for index in range(start_index, len(lines)):
        if is_admin_footer_line(lines[index]):
            return index
    return None


def is_admin_footer_line(line: str) -> bool:
    stripped = line.strip()
    if ADMIN_FOOTER_RE.match(stripped):
        return True
    return stripped.isupper() and stripped in {"BỘ TRƯỞNG", "THỦ TƯỚNG", "CHÍNH PHỦ"}


def find_first_admin_footer_after_article(lines: Sequence[str]) -> int | None:
    seen_article = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if ARTICLE_RE.match(stripped):
            seen_article = True
            continue
        if seen_article and is_admin_footer_line(stripped):
            return index
    return None


def find_appendix_start(lines: Sequence[str], start_index: int) -> int | None:
    for index in range(start_index, len(lines)):
        stripped = lines[index].strip()
        if APPENDIX_HEADING_RE.match(stripped) or FORM_HEADING_RE.match(stripped):
            return index
    return None


def split_main_text_and_appendix_text(text: str) -> tuple[str, str]:
    cleaned = normalize_extracted_text(text)
    lines = cleaned.splitlines()
    if find_last_article_line(lines) is None:
        return cleaned, ""

    footer_start = find_first_admin_footer_after_article(lines)
    if footer_start is not None:
        appendix_start = find_appendix_start(lines, footer_start + 1)
        return "\n".join(lines[:footer_start]).strip(), (
            "\n".join(lines[appendix_start:]).strip() if appendix_start is not None else ""
        )

    last_article_index = find_last_article_line(lines)
    appendix_start = find_appendix_start(lines, (last_article_index or 0) + 1)
    footer_start = find_admin_footer_start(lines, (last_article_index or 0) + 1)

    main_end_candidates = [index for index in [footer_start, appendix_start] if index is not None]
    main_end = min(main_end_candidates) if main_end_candidates else len(lines)

    appendix_search_start = footer_start + 1 if footer_start is not None else main_end
    appendix_start = appendix_start
    if appendix_start is None or appendix_start < appendix_search_start:
        appendix_start = find_appendix_start(lines, appendix_search_start)

    main_text = "\n".join(lines[:main_end]).strip()
    appendix_text = "\n".join(lines[appendix_start:]).strip() if appendix_start is not None else ""
    return main_text, appendix_text


def normalize_appendix_label(label: str | None) -> str:
    if not label:
        return "Khong_Ro"
    return normalize_legal_id_token(label.strip().upper())


def infer_appendix_label_from_form_code(code: str) -> str:
    normalized_code = normalize_legal_id_token(code).upper()
    if not normalized_code.startswith("PL"):
        return "Khong_Ro"
    label = normalized_code[2:] or "I"
    return label


def appendix_chunk_base(document_id: str, *parts: object) -> str:
    return "_".join(
        part
        for part in [infer_chunk_id_prefix(document_id), *[normalize_legal_id_token(value) for value in parts]]
        if part
    )


def pack_appendix_lines(heading: str, body_lines: Sequence[str], max_chars: int) -> list[str]:
    groups: list[list[str]] = []
    current: list[str] = []

    for line in body_lines:
        candidate = [*current, line]
        candidate_text = "\n".join([heading, *candidate]).strip()
        if current and len(candidate_text) > max_chars:
            groups.append(current)
            current = [line]
            continue
        current = candidate

    if current:
        groups.append(current)

    if not groups:
        return [heading]
    return ["\n".join([heading, *group]).strip() for group in groups]


def make_appendix_blocks_from_lines(
    *,
    document_id: str,
    base_id: str,
    appendix_heading: str,
    chunk_type: str,
    body_lines: Sequence[str],
    max_chars: int,
    split_suffix: str = "Nhom",
) -> list[AppendixBlock]:
    texts = pack_appendix_lines(appendix_heading, body_lines, max_chars)
    blocks: list[AppendixBlock] = []
    for index, text in enumerate(texts, start=1):
        chunk_id = base_id if len(texts) == 1 else f"{base_id}_{split_suffix}_{index:02d}"
        blocks.append(
            AppendixBlock(
                chunk_id=chunk_id,
                appendix_id=base_id,
                appendix_heading=appendix_heading,
                chunk_type=chunk_type,
                text=text,
            )
        )
    return blocks


def split_lines_at_headings(lines: Sequence[str], pattern: re.Pattern[str]) -> list[list[str]]:
    starts = [index for index, line in enumerate(lines) if pattern.match(line.strip())]
    if not starts:
        return [list(lines)] if lines else []
    sections: list[list[str]] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        section_lines = [line for line in lines[start:end] if line.strip()]
        if section_lines:
            sections.append(section_lines)
    return sections


def build_nd135_appendix_blocks(
    appendix_text: str,
    document_id: str,
    max_chars: int,
) -> list[AppendixBlock]:
    lines = [line.strip() for line in appendix_text.splitlines()]
    appendix_sections = split_lines_at_headings(lines, APPENDIX_HEADING_RE)
    blocks: list[AppendixBlock] = []

    for section_lines in appendix_sections:
        heading_line = section_lines[0]
        appendix_match = APPENDIX_HEADING_RE.match(heading_line)
        label = normalize_appendix_label(appendix_match.group("label") if appendix_match else None)
        title_line = section_lines[1] if len(section_lines) > 1 else ""
        appendix_heading = f"{heading_line}. {title_line}".strip(". ")
        content_lines = section_lines[2:]

        male_index = next(
            (index for index, line in enumerate(content_lines) if normalize_for_matching(line) == "lao dong nam"),
            None,
        )
        female_index = next(
            (index for index, line in enumerate(content_lines) if normalize_for_matching(line) == "lao dong nu"),
            None,
        )

        if male_index is not None and female_index is not None and male_index < female_index:
            table_specs = [
                ("Bang_Nam", "Lao động nam", content_lines[male_index + 1 : female_index]),
                ("Bang_Nu", "Lao động nữ", content_lines[female_index + 1 :]),
            ]
            for table_token, table_heading, table_lines in table_specs:
                base_id = appendix_chunk_base(document_id, "Phu_Luc", label, table_token)
                blocks.extend(
                    make_appendix_blocks_from_lines(
                        document_id=document_id,
                        base_id=base_id,
                        appendix_heading=f"{appendix_heading}. {table_heading}",
                        chunk_type="appendix_table",
                        body_lines=table_lines,
                        max_chars=max_chars,
                    )
                )
            continue

        base_id = appendix_chunk_base(document_id, "Phu_Luc", label)
        blocks.extend(
            make_appendix_blocks_from_lines(
                document_id=document_id,
                base_id=base_id,
                appendix_heading=appendix_heading,
                chunk_type="appendix_list",
                body_lines=content_lines,
                max_chars=max_chars,
            )
        )

    return blocks


def build_generic_appendix_blocks(
    appendix_text: str,
    document_id: str,
    max_chars: int,
) -> list[AppendixBlock]:
    lines = [line.strip() for line in appendix_text.splitlines() if line.strip()]
    starts = [
        index
        for index, line in enumerate(lines)
        if FORM_HEADING_RE.match(line) or APPENDIX_HEADING_RE.match(line)
    ]
    if not starts:
        return []

    blocks: list[AppendixBlock] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        segment_lines = lines[start:end]
        heading_line = segment_lines[0]

        if form_match := FORM_HEADING_RE.match(heading_line):
            form_number = normalize_legal_id_token(form_match.group("number"))
            form_code = normalize_legal_id_token(form_match.group("code")).upper()
            appendix_label = infer_appendix_label_from_form_code(form_code)
            base_id = appendix_chunk_base(
                document_id,
                "Phu_Luc",
                appendix_label,
                "Mau",
                form_number,
                form_code,
            )
            appendix_heading = heading_line
            chunk_type = "appendix_form"
            split_suffix = "Phan"
        else:
            appendix_match = APPENDIX_HEADING_RE.match(heading_line)
            appendix_label = normalize_appendix_label(
                appendix_match.group("label") if appendix_match else None
            )
            base_id = appendix_chunk_base(document_id, "Phu_Luc", appendix_label)
            appendix_heading = heading_line
            chunk_type = "appendix"
            split_suffix = "Nhom"

        blocks.extend(
            make_appendix_blocks_from_lines(
                document_id=document_id,
                base_id=base_id,
                appendix_heading=appendix_heading,
                chunk_type=chunk_type,
                body_lines=segment_lines[1:],
                max_chars=max_chars,
                split_suffix=split_suffix,
            )
        )

    return blocks


def build_appendix_blocks(
    appendix_text: str,
    document_id: str,
    max_chars: int,
) -> list[AppendixBlock]:
    if not appendix_text.strip():
        return []
    if document_id == "nghi-dinh-135-2020-nd-cp":
        return build_nd135_appendix_blocks(appendix_text, document_id, max_chars)
    return build_generic_appendix_blocks(appendix_text, document_id, max_chars)


def list_pages_for_citation(source_pages: list[int]) -> str | None:
    if not source_pages:
        return None
    if len(source_pages) == 1:
        return f"tr. {source_pages[0]}"
    return f"tr. {source_pages[0]}-{source_pages[-1]}"


def normalize_point_refs(point_ref: str | None = None, point_refs: Sequence[object] | None = None) -> list[str]:
    refs: list[str] = []
    for value in [point_ref, *(point_refs or [])]:
        if value is None:
            continue
        ref = str(value).strip()
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def infer_part_number(part_line: str) -> str:
    match = PART_RE.match(part_line.strip())
    if not match:
        return ""
    value = match.group("number").strip()
    return value.split(".", 1)[0].strip()


def normalize_legal_id_token(value: object) -> str:
    text = str(value).strip()
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text


def normalize_point_id_token(value: str) -> str:
    text = str(value).strip().replace("đ", "dd").replace("Đ", "DD")
    return normalize_legal_id_token(text.replace(".", ""))


def infer_article_occurrence_from_section_id(section_id: str) -> str | None:
    occurrence_match = re.search(r"-occurrence-(?P<number>\d+)$", section_id)
    if not occurrence_match:
        return None
    return occurrence_match.group("number")


def infer_chunk_id_prefix(section_id: str) -> str:
    normalized = normalize_for_matching(section_id).replace("_", "-")
    if "45-2019-qh14" in normalized:
        return "45_2019_QH14"
    if "nghi-dinh-145-2020" in normalized or "nghi-dinh-145" in normalized:
        return "ND145_2020"
    if "nghi-dinh-135-2020" in normalized or "nghi-dinh-135" in normalized:
        return "ND135_2020"
    if "thong-tu-09-2020" in normalized or "thong-tu-09" in normalized:
        return "TT09_2020"
    if "thong-tu-10-2020" in normalized or "thong-tu-10" in normalized:
        return "TT10_2020"
    if "92-2015-qh13" in normalized or "blttds" in normalized:
        return "BLTTDS_2015"
    return normalize_legal_id_token(section_id).upper()


def build_stable_chunk_id(
    *,
    section: SectionRecord,
    index: int,
    clause_ref: str | None,
    point_refs: Sequence[str],
    chunk_type: str,
) -> str:
    if not section.article_number:
        return f"{section.section_id}-chunk-{index:02d}"

    parts = [
        infer_chunk_id_prefix(section.section_id),
        "Dieu",
        normalize_legal_id_token(section.article_number),
    ]
    if article_occurrence := infer_article_occurrence_from_section_id(section.section_id):
        parts.extend(["Lan", normalize_legal_id_token(article_occurrence)])
    if chunk_type == "article_intro":
        parts.append("Intro")
    elif chunk_type == "article_sequential":
        parts.extend(["Chunk", f"{index:02d}"])

    if clause_ref:
        parts.extend(["Khoan", normalize_legal_id_token(clause_ref)])
    point_ref_values = normalize_point_refs(point_refs=point_refs)
    if point_ref_values:
        parts.append("Diem")
        parts.extend(normalize_point_id_token(point_ref) for point_ref in point_ref_values)

    return "_".join(part for part in parts if part)


def dedupe_chunk_ids(section_chunks: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: dict[str, int] = {}
    for chunk in section_chunks:
        chunk_id = str(chunk["chunk_id"])
        count = seen.get(chunk_id, 0) + 1
        seen[chunk_id] = count
        if count > 1:
            chunk["chunk_id"] = f"{chunk_id}_Chunk_{int(chunk['chunk_index']):02d}"
    return section_chunks


def format_point_refs_for_citation(point_refs: Sequence[str]) -> str | None:
    refs = normalize_point_refs(point_refs=point_refs)
    if not refs:
        return None
    if len(refs) == 1:
        return f"điểm {refs[0]}"
    return "các điểm " + ", ".join(refs)


def build_citation_text(
    document_title: str,
    article_number: str | None,
    article_title: str | None,
    clause_ref: str | None,
    point_ref: str | None,
    source_pages: list[int],
    point_refs: Sequence[str] | None = None,
) -> str:
    parts = [document_title]
    reference_parts: list[str] = []
    if article_number:
        article_label = f"Điều {article_number}"
        if article_title:
            article_label = f"{article_label} ({article_title})"
        reference_parts.append(article_label)
    if clause_ref:
        reference_parts.append(f"khoản {clause_ref}")
    if point_label := format_point_refs_for_citation(normalize_point_refs(point_ref, point_refs)):
        reference_parts.append(point_label)
    if reference_parts:
        parts.append(", ".join(reference_parts))
    page_label = list_pages_for_citation(source_pages)
    if page_label:
        parts.append(page_label)
    return ", ".join(parts)


def extract_chunk_body(text: str, heading: str) -> str:
    prefix = f"{heading}\n\n"
    if text.startswith(prefix):
        return text[len(prefix) :].strip()
    if text.startswith(heading):
        return text[len(heading) :].strip()
    return text.strip()


def infer_chunk_level(clause_ref: str | None, point_ref: str | None) -> str:
    if point_ref:
        return "point"
    if clause_ref:
        return "clause"
    return "article"


def match_legal_marker(text: str) -> tuple[str | None, str | None]:
    if subpoint_match := SUBPOINT_START_RE.match(text):
        return "point", subpoint_match.group("label")
    if point_match := POINT_START_RE.match(text):
        return "point", point_match.group("label")
    if clause_match := CLAUSE_START_RE.match(text):
        return "clause", clause_match.group("label")
    return None, None


def split_legal_marker_segments(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    matches = list(LEGAL_MARKER_BOUNDARY_RE.finditer(stripped))
    if not matches:
        return [stripped]

    starts = [match.start() for match in matches]
    if starts[0] != 0:
        starts = [0, *starts]

    segments: list[str] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(stripped)
        segment = stripped[start:end].strip()
        if segment:
            segments.append(segment)
    return segments


def build_legal_units(body_paragraphs: list[str], body_limit: int) -> list[dict[str, object]]:
    units: list[dict[str, object]] = []
    current_clause: str | None = None
    current_point: str | None = None

    for paragraph in body_paragraphs:
        for segment in split_legal_marker_segments(paragraph):
            marker_kind, marker_label = match_legal_marker(segment)

            clause_ref = current_clause
            point_ref = current_point

            if marker_kind == "clause":
                clause_ref = marker_label
                point_ref = None
                current_clause = clause_ref
                current_point = None
            elif marker_kind == "point":
                point_ref = marker_label
                current_point = point_ref

            level = infer_chunk_level(clause_ref=clause_ref, point_ref=point_ref)

            for piece in split_text_for_chunking(segment, body_limit):
                units.append(
                    {
                        "text": piece,
                        "clause_ref": clause_ref,
                        "point_ref": point_ref,
                        "level": level,
                    }
                )

    return units


def collect_labels(rule_map: dict[str, list[str]], normalized_text: str) -> list[str]:
    labels = [label for label, keywords in rule_map.items() if any(keyword in normalized_text for keyword in keywords)]
    return sorted(labels)


def infer_chunk_taxonomy(
    document_title: str,
    section_heading: str | None,
    article_title: str | None,
    body_text: str,
) -> tuple[list[str], list[str], list[str]]:
    context = " ".join(part for part in [document_title, section_heading or "", article_title or "", body_text] if part).strip()
    normalized = normalize_for_matching(context)
    topics = collect_labels(TOPIC_RULES, normalized)
    actors = collect_labels(ACTOR_RULES, normalized)
    issue_types = collect_labels(ISSUE_TYPE_RULES, normalized)
    return topics, actors, issue_types


def build_retrieval_text(
    document_title: str,
    heading: str,
    chapter_heading: str | None,
    section_heading: str | None,
    citation_text: str,
    topic: list[str],
    actor: list[str],
    issue_type: list[str],
    text: str,
) -> str:
    body_text = extract_chunk_body(text, heading)
    del document_title, topic, actor, issue_type
    context_parts = [part for part in [chapter_heading, section_heading] if part]
    context_prefix = f"Trong {'; '.join(context_parts)}, " if context_parts else ""
    return f"{context_prefix}{citation_text} quy định: {body_text}".strip()


def derive_parent_point_ref(point_ref: str | None) -> str | None:
    if not point_ref or "." not in point_ref:
        return None
    return point_ref.rsplit(".", 1)[0]


def assign_parent_chunk_ids(section_chunks: list[dict[str, object]]) -> list[dict[str, object]]:
    article_chunk_id: str | None = None
    clause_parent_ids: dict[str, str] = {}
    point_parent_ids: dict[tuple[str | None, str], str] = {}

    for chunk in section_chunks:
        clause_ref = str(chunk["clause_ref"]) if chunk.get("clause_ref") else None
        point_ref = str(chunk["point_ref"]) if chunk.get("point_ref") else None
        level = infer_chunk_level(clause_ref=clause_ref, point_ref=point_ref)

        parent_chunk_id: str | None = None

        if level == "article":
            article_chunk_id = str(chunk["chunk_id"])
        elif level == "clause":
            parent_chunk_id = article_chunk_id
            if clause_ref and clause_ref not in clause_parent_ids:
                clause_parent_ids[clause_ref] = str(chunk["chunk_id"])
        else:
            parent_point_ref = derive_parent_point_ref(point_ref)
            if parent_point_ref:
                parent_chunk_id = point_parent_ids.get((clause_ref, parent_point_ref))
            if parent_chunk_id is None and clause_ref:
                parent_chunk_id = clause_parent_ids.get(clause_ref)
            if parent_chunk_id is None:
                parent_chunk_id = article_chunk_id
            if point_ref and (clause_ref, point_ref) not in point_parent_ids:
                point_parent_ids[(clause_ref, point_ref)] = str(chunk["chunk_id"])

        chunk["parent_chunk_id"] = parent_chunk_id

    return section_chunks


def enrich_chunk(
    chunk: dict[str, object],
    document_title: str,
    source_kind: str,
) -> dict[str, object]:
    heading = str(chunk["heading"])
    text = str(chunk["text"])
    article_number = chunk["article_number"]
    article_title = chunk["article_title"]
    section_heading = chunk["section_heading"]
    chapter_heading = chunk["chapter_heading"]
    appendix_heading = str(chunk.get("appendix_heading") or "")
    clause_ref = str(chunk["clause_ref"]) if chunk.get("clause_ref") else None
    point_ref = str(chunk["point_ref"]) if chunk.get("point_ref") else None
    point_refs = normalize_point_refs(point_ref, chunk.get("point_refs") if isinstance(chunk.get("point_refs"), list) else None)

    source_pages = list(chunk["source_pages"]) if source_kind == "raw_pdf" else []
    body_text = extract_chunk_body(text, heading)
    level = str(chunk.get("level") or infer_chunk_level(clause_ref=clause_ref, point_ref=point_ref))
    chunk_type = str(chunk.get("chunk_type") or level)
    topic, actor, issue_type = infer_chunk_taxonomy(
        document_title=document_title,
        section_heading=str(section_heading) if section_heading else None,
        article_title=str(article_title) if article_title else None,
        body_text=body_text,
    )
    citation_text = build_citation_text(
        document_title=document_title,
        article_number=str(article_number) if article_number else None,
        article_title=str(article_title) if article_title else None,
        clause_ref=clause_ref,
        point_ref=point_ref,
        source_pages=source_pages,
        point_refs=point_refs,
    )
    if appendix_heading and not article_number:
        citation_text = f"{document_title}, {appendix_heading}"
    retrieval_text = build_retrieval_text(
        document_title=document_title,
        heading=heading,
        chapter_heading=appendix_heading or (str(chapter_heading) if chapter_heading else None),
        section_heading=str(section_heading) if section_heading else None,
        citation_text=citation_text,
        topic=topic,
        actor=actor,
        issue_type=issue_type,
        text=text,
    )

    parent_chunk_id = chunk.get("parent_chunk_id")

    enriched = {
        **chunk,
        "level": level,
        "chunk_type": chunk_type,
        "point_refs": point_refs,
        "parent_chunk_id": parent_chunk_id,
        "citation_text": citation_text,
        "retrieval_text": retrieval_text,
        "page_content": retrieval_text,
        "topic": topic,
        "actor": actor,
        "issue_type": issue_type,
    }

    if source_pages:
        enriched["source_pages"] = source_pages
    else:
        enriched.pop("source_pages", None)

    return enriched


def resolve_curated_legal_chunk_paths(
    input_dir: Path,
    filenames: Sequence[str] = CURATED_LEGAL_CHUNK_FILENAMES,
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


def make_appendix_chunks(blocks: Sequence[AppendixBlock]) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for block in blocks:
        if len(block.text.strip()) < VERY_SHORT_CHUNK_THRESHOLD:
            continue
        chunks.append(
            {
                "chunk_id": block.chunk_id,
                "section_id": block.appendix_id,
                "article_number": None,
                "article_title": None,
                "heading": block.appendix_heading,
                "part_number": None,
                "part_heading": None,
                "chapter_heading": None,
                "section_heading": None,
                "appendix_id": block.appendix_id,
                "appendix_heading": block.appendix_heading,
                "source_pages": [],
                "chunk_index": len(chunks) + 1,
                "char_count": len(block.text),
                "text": block.text,
                "clause_ref": None,
                "point_ref": None,
                "point_refs": [],
                "chunk_type": block.chunk_type,
                "level": "appendix",
                "parent_chunk_id": None,
            }
        )
    return chunks


def build_curated_chunk_records(
    text_paths: Sequence[Path],
    *,
    max_chars: int = 1200,
) -> tuple[list[dict[str, object]], list[str]]:
    selected_paths, warnings = select_curated_text_sources(text_paths)
    records: list[dict[str, object]] = []

    for text_path in selected_paths:
        source_text = text_path.read_text(encoding="utf-8")
        main_text, appendix_text = split_main_text_and_appendix_text(source_text)
        page_records = build_page_records_from_text(main_text)
        cleaned_text = build_cleaned_text(page_records)
        metadata = resolve_canonical_document_metadata(text_path, cleaned_text)
        document_id = metadata.document_id
        document_title = metadata.document_title
        sections = split_sections(
            page_records=page_records,
            document_id=document_id,
            document_title=document_title,
        )
        chunks = chunk_sections(sections, max_chars=max_chars)
        appendix_chunks = dedupe_chunk_ids(
            make_appendix_chunks(
                blocks=build_appendix_blocks(appendix_text, document_id, max_chars),
            )
        )

        for chunk in [*chunks, *appendix_chunks]:
            records.append(
                {
                    "document_id": document_id,
                    "document_title": document_title,
                    "document_type": metadata.document_type,
                    "source_kind": "curated_text",
                    "source_path": str(text_path.resolve().as_posix()),
                    **enrich_chunk(
                        chunk=chunk,
                        document_title=document_title,
                        source_kind="curated_text",
                    ),
                }
            )

    return records, warnings


def summarize_legal_chunks(chunks: Sequence[dict[str, object]]) -> dict[str, object]:
    chunk_count_by_document: dict[str, int] = {}
    chunk_count_by_level: dict[str, int] = {}
    chunk_count_by_chunk_type: dict[str, int] = {}
    chunk_ids: dict[str, int] = {}
    missing_citation: list[dict[str, object]] = []
    missing_article_number: list[dict[str, object]] = []
    very_short_chunks: list[dict[str, object]] = []
    very_long_chunks: list[dict[str, object]] = []
    very_long_normal_chunks: list[dict[str, object]] = []

    for chunk in chunks:
        document_id = str(chunk.get("document_id") or "")
        level = str(chunk.get("level") or "")
        chunk_type = str(chunk.get("chunk_type") or "")
        chunk_id = str(chunk.get("chunk_id") or "")
        text = str(chunk.get("text") or "")
        char_count = int(chunk.get("char_count") or len(text))

        chunk_count_by_document[document_id] = chunk_count_by_document.get(document_id, 0) + 1
        chunk_count_by_level[level] = chunk_count_by_level.get(level, 0) + 1
        chunk_count_by_chunk_type[chunk_type] = chunk_count_by_chunk_type.get(chunk_type, 0) + 1
        chunk_ids[chunk_id] = chunk_ids.get(chunk_id, 0) + 1

        preview_record = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "source_path": chunk.get("source_path"),
            "article_number": chunk.get("article_number"),
            "appendix_id": chunk.get("appendix_id"),
            "appendix_heading": chunk.get("appendix_heading"),
            "clause_ref": chunk.get("clause_ref"),
            "point_refs": chunk.get("point_refs") or [],
            "chunk_type": chunk_type,
            "char_count": char_count,
        }
        if not chunk.get("citation_text"):
            missing_citation.append(preview_record)
        if chunk_type in NORMAL_LEGAL_CHUNK_TYPES and not chunk.get("article_number"):
            missing_article_number.append(preview_record)
        if 0 < char_count < VERY_SHORT_CHUNK_THRESHOLD:
            very_short_chunks.append({**preview_record, "text": text})
        if char_count > VERY_LONG_CHUNK_THRESHOLD:
            very_long_chunks.append(preview_record)
            if chunk_type in NORMAL_LEGAL_CHUNK_TYPES:
                very_long_normal_chunks.append(preview_record)

    duplicate_ids = {
        chunk_id: count for chunk_id, count in chunk_ids.items() if chunk_id and count > 1
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chunk_count": len(chunks),
        "chunk_count_by_document": chunk_count_by_document,
        "chunk_count_by_level": chunk_count_by_level,
        "chunk_count_by_chunk_type": chunk_count_by_chunk_type,
        "duplicate_chunk_id_count": sum(count - 1 for count in duplicate_ids.values()),
        "duplicate_chunk_ids": duplicate_ids,
        "chunks_missing_citation_text": missing_citation,
        "chunks_missing_article_number": missing_article_number,
        "very_short_chunks": very_short_chunks,
        "very_long_chunks": very_long_chunks,
        "very_long_normal_chunks": very_long_normal_chunks,
    }


def render_legal_chunks_summary_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Legal Chunks Summary",
        "",
        f"- Chunk count: {summary['chunk_count']}",
        f"- Duplicate chunk IDs: {summary['duplicate_chunk_id_count']}",
        f"- Missing citation text: {len(summary['chunks_missing_citation_text'])}",
        f"- Missing article number: {len(summary['chunks_missing_article_number'])}",
        f"- Very short chunks: {len(summary['very_short_chunks'])}",
        f"- Very long chunks: {len(summary['very_long_chunks'])}",
        f"- Very long normal legal chunks: {len(summary['very_long_normal_chunks'])}",
        "",
        "## Chunks By Document",
        "",
        "| Document ID | Chunks |",
        "| --- | ---: |",
    ]
    for document_id, count in sorted(summary["chunk_count_by_document"].items()):
        lines.append(f"| {document_id} | {count} |")

    lines.extend(["", "## Chunks By Level", "", "| Level | Chunks |", "| --- | ---: |"])
    for level, count in sorted(summary["chunk_count_by_level"].items()):
        lines.append(f"| {level} | {count} |")

    lines.extend(["", "## Chunks By Type", "", "| Chunk type | Chunks |", "| --- | ---: |"])
    for chunk_type, count in sorted(summary["chunk_count_by_chunk_type"].items()):
        lines.append(f"| {chunk_type} | {count} |")

    if summary["duplicate_chunk_ids"]:
        lines.extend(["", "## Duplicate Chunk IDs", ""])
        for chunk_id, count in sorted(summary["duplicate_chunk_ids"].items()):
            lines.append(f"- `{chunk_id}`: {count}")

    if summary["very_short_chunks"]:
        lines.extend(["", "## Very Short Chunks", ""])
        for chunk in summary["very_short_chunks"][:50]:
            lines.append(
                f"- `{chunk['chunk_id']}` ({chunk['char_count']} chars): {chunk.get('text', '')}"
            )

    if summary["very_long_chunks"]:
        lines.extend(["", "## Very Long Chunks", ""])
        for chunk in summary["very_long_chunks"][:50]:
            lines.append(f"- `{chunk['chunk_id']}` ({chunk['char_count']} chars)")

    if summary["very_long_normal_chunks"]:
        lines.extend(["", "## Very Long Normal Legal Chunks", ""])
        for chunk in summary["very_long_normal_chunks"][:50]:
            lines.append(f"- `{chunk['chunk_id']}` ({chunk['char_count']} chars)")

    return "\n".join(lines).strip() + "\n"


def write_legal_chunk_artifacts(
    chunks: Sequence[dict[str, object]],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = output_dir / "legal_chunks.jsonl"
    summary_json_path = output_dir / "legal_chunks_summary.json"
    summary_md_path = output_dir / "legal_chunks_summary.md"

    with chunks_path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    summary = summarize_legal_chunks(chunks)
    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_md_path.write_text(
        render_legal_chunks_summary_markdown(summary),
        encoding="utf-8",
    )
    return chunks_path, summary_json_path, summary_md_path


def split_by_regex_boundaries(text: str, pattern: re.Pattern[str]) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    matches = list(pattern.finditer(stripped))
    if not matches:
        return [stripped]

    starts = [match.start() for match in matches]
    if starts[0] != 0:
        starts = [0, *starts]

    parts: list[str] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(stripped)
        piece = stripped[start:end].strip()
        if piece:
            parts.append(piece)

    return parts if len(parts) > 1 else [stripped]


def split_by_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(stripped) if part.strip()]
    return parts if len(parts) > 1 else [stripped]


def split_by_nearest_whitespace(text: str, max_chars: int) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= max_chars:
        return [stripped]

    parts: list[str] = []
    remaining = stripped

    while len(remaining) > max_chars:
        cut = -1
        search_limit = min(max_chars, len(remaining) - 1)

        for index in range(search_limit, -1, -1):
            if remaining[index].isspace():
                cut = index
                break

        if cut == -1:
            for index in range(search_limit + 1, len(remaining)):
                if remaining[index].isspace():
                    cut = index
                    break

        if cut == -1:
            parts.append(remaining)
            return parts

        piece = remaining[:cut].strip()
        if piece:
            parts.append(piece)
        remaining = remaining[cut:].strip()

    if remaining:
        parts.append(remaining)

    return parts


def pack_text_units(units: list[str], max_chars: int, separator: str = "\n") -> list[str]:
    packed: list[str] = []
    current = ""

    for unit in units:
        candidate = f"{current}{separator}{unit}".strip() if current else unit
        if current and len(candidate) > max_chars:
            packed.append(current.strip())
            current = unit
            continue
        current = candidate

    if current:
        packed.append(current.strip())

    return packed


def split_text_for_chunking(text: str, max_chars: int) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= max_chars:
        return [stripped]

    for splitter in (
        lambda value: split_by_regex_boundaries(value, CLAUSE_SPLIT_RE),
        lambda value: split_by_regex_boundaries(value, SUBPOINT_SPLIT_RE),
        lambda value: split_by_regex_boundaries(value, POINT_SPLIT_RE),
        split_by_sentences,
    ):
        units = splitter(stripped)
        if len(units) <= 1:
            continue

        flattened: list[str] = []
        for unit in units:
            flattened.extend(split_text_for_chunking(unit, max_chars))
        return pack_text_units(flattened, max_chars)

    return split_by_nearest_whitespace(stripped, max_chars)


def split_sections(page_records: list[PageRecord], document_id: str, document_title: str) -> list[SectionRecord]:
    paragraph_records: list[dict[str, object]] = []
    for record in page_records:
        if not record.text:
            continue
        for paragraph in split_paragraphs(record.text):
            paragraph_records.append({"page_number": record.page_number, "text": paragraph})

    if not paragraph_records:
        return []

    sections: list[SectionRecord] = []
    preamble: list[dict[str, object]] = []
    current_section: dict[str, object] | None = None
    current_part_number: str | None = None
    current_part_heading: str | None = None
    current_chapter: str | None = None
    current_heading_group: str | None = None
    pending_part_line: str | None = None
    pending_chapter_line: str | None = None
    article_occurrences: dict[str, int] = {}

    def flush_current() -> None:
        nonlocal current_section
        if not current_section:
            return
        paragraphs = current_section["paragraphs"]
        heading = current_section["heading"]
        text = "\n\n".join([heading, *paragraphs]).strip()
        sections.append(
            SectionRecord(
                section_id=str(current_section["section_id"]),
                heading=str(heading),
                article_number=current_section["article_number"],
                article_title=current_section["article_title"],
                chapter_heading=current_section["chapter_heading"],
                section_heading=current_section["section_heading"],
                source_pages=sorted(current_section["source_pages"]),
                text=text,
                part_number=current_section["part_number"],
                part_heading=current_section["part_heading"],
            )
        )
        current_section = None

    for entry in paragraph_records:
        page_number = int(entry["page_number"])
        lines = [line.strip() for line in str(entry["text"]).splitlines() if line.strip()]

        for line in lines:
            part_match = PART_RE.match(line)
            chapter_match = CHAPTER_RE.match(line)
            section_match = SECTION_RE.match(line)
            group_match = GROUP_RE.match(line)
            article_match = ARTICLE_RE.match(line)

            if part_match:
                flush_current()
                current_part_number = infer_part_number(line)
                current_part_heading = line
                current_chapter = None
                current_heading_group = None
                pending_part_line = line
                pending_chapter_line = None
                continue

            if pending_part_line and not (section_match or group_match or article_match or chapter_match or part_match):
                current_part_heading = f"{pending_part_line}. {line}"
                pending_part_line = None
                continue

            pending_part_line = None

            if chapter_match:
                flush_current()
                current_chapter = line
                current_heading_group = None
                pending_chapter_line = line
                continue

            if pending_chapter_line and not (section_match or group_match or article_match or chapter_match):
                current_chapter = f"{pending_chapter_line}. {line}"
                pending_chapter_line = None
                continue

            pending_chapter_line = None

            if section_match:
                flush_current()
                current_heading_group = line
                continue

            if group_match:
                flush_current()
                current_heading_group = line
                continue

            if article_match:
                flush_current()
                article_number = article_match.group("number")
                article_title = article_match.group("title").strip()
                article_occurrence = article_occurrences.get(article_number, 0) + 1
                article_occurrences[article_number] = article_occurrence
                section_id = f"{document_id}-dieu-{article_number}"
                if article_occurrence > 1:
                    section_id = f"{section_id}-occurrence-{article_occurrence}"
                current_section = {
                    "section_id": section_id,
                    "heading": line,
                    "article_number": article_number,
                    "article_title": article_title,
                    "part_number": current_part_number,
                    "part_heading": current_part_heading,
                    "chapter_heading": current_chapter,
                    "section_heading": current_heading_group,
                    "source_pages": {page_number},
                    "paragraphs": [],
                }
                continue

            if current_section is None:
                preamble.append({"page_number": page_number, "text": line})
                continue

            current_section["paragraphs"].append(line)
            current_section["source_pages"].add(page_number)

    flush_current()

    if preamble:
        preamble_text = "\n\n".join(str(item["text"]) for item in preamble).strip()
        sections.insert(
            0,
            SectionRecord(
                section_id=f"{document_id}-preamble",
                heading=document_title,
                article_number=None,
                article_title=None,
                chapter_heading=None,
                section_heading=None,
                source_pages=sorted({int(item["page_number"]) for item in preamble}),
                text=preamble_text,
                part_number=None,
                part_heading=None,
            ),
        )

    if sections:
        return sections

    cleaned_text = build_cleaned_text(page_records)
    return [
        SectionRecord(
            section_id=f"{document_id}-fulltext",
            heading=document_title,
            article_number=None,
            article_title=None,
            chapter_heading=None,
            section_heading=None,
            source_pages=[record.page_number for record in page_records if record.text],
            text=cleaned_text,
            part_number=None,
            part_heading=None,
        )
    ]


def compose_section_chunk_text(heading: str, parts: Sequence[str]) -> str:
    clean_parts = [part.strip() for part in parts if part and part.strip()]
    if not clean_parts:
        return heading
    return "\n\n".join([heading, *clean_parts]).strip()


def flatten_point_parts(points: Sequence[LegalPointGroup]) -> list[str]:
    parts: list[str] = []
    for point in points:
        parts.extend(point.parts)
    return parts


def parse_clause_groups(body_paragraphs: Sequence[str]) -> tuple[list[str], list[LegalClauseGroup]]:
    article_intro_parts: list[str] = []
    clauses: list[LegalClauseGroup] = []
    current_clause: LegalClauseGroup | None = None
    current_point: LegalPointGroup | None = None

    for paragraph in body_paragraphs:
        for segment in split_legal_marker_segments(paragraph):
            marker_kind, marker_label = match_legal_marker(segment)

            if marker_kind == "clause":
                current_clause = LegalClauseGroup(
                    clause_ref=marker_label,
                    intro_parts=[segment],
                    points=[],
                )
                clauses.append(current_clause)
                current_point = None
                continue

            if marker_kind == "point":
                if current_clause is None:
                    current_clause = LegalClauseGroup(
                        clause_ref=None,
                        intro_parts=[],
                        points=[],
                    )
                    clauses.append(current_clause)
                current_point = LegalPointGroup(point_ref=str(marker_label), parts=[segment])
                current_clause.points.append(current_point)
                continue

            if current_clause is None:
                article_intro_parts.append(segment)
            elif current_point is not None:
                current_point.parts.append(segment)
            else:
                current_clause.intro_parts.append(segment)

    return article_intro_parts, clauses


def split_parts_for_chunking(
    *,
    heading: str,
    prefix_parts: Sequence[str],
    body_text: str,
    max_chars: int,
) -> list[list[str]]:
    prefix_text = compose_section_chunk_text(heading, prefix_parts)
    body_limit = max(max_chars - len(prefix_text) - 2, 100)
    if not body_text.strip():
        return [list(prefix_parts)]
    return [[*prefix_parts, piece] for piece in split_text_for_chunking(body_text, body_limit)]


def make_section_chunk(
    *,
    section: SectionRecord,
    index: int,
    text: str,
    clause_ref: str | None,
    point_refs: Sequence[str] = (),
    chunk_type: str,
) -> dict[str, object]:
    point_ref_values = normalize_point_refs(point_refs=point_refs)
    return {
        "chunk_id": build_stable_chunk_id(
            section=section,
            index=index,
            clause_ref=clause_ref,
            point_refs=point_ref_values,
            chunk_type=chunk_type,
        ),
        "section_id": section.section_id,
        "article_number": section.article_number,
        "article_title": section.article_title,
        "heading": section.heading,
        "part_number": section.part_number,
        "part_heading": section.part_heading,
        "chapter_heading": section.chapter_heading,
        "section_heading": section.section_heading,
        "source_pages": section.source_pages,
        "chunk_index": index,
        "char_count": len(text),
        "text": text,
        "clause_ref": clause_ref,
        "point_ref": None,
        "point_refs": point_ref_values,
        "chunk_type": chunk_type,
    }


def build_article_only_chunk_records(
    *,
    heading: str,
    body_parts: Sequence[str],
    max_chars: int,
    chunk_type: str,
) -> list[dict[str, object]]:
    body_text = "\n\n".join(part.strip() for part in body_parts if part and part.strip()).strip()
    if not body_text:
        return [{"text": heading, "clause_ref": None, "point_refs": [], "chunk_type": chunk_type}]

    composed = compose_section_chunk_text(heading, [body_text])
    if len(composed) <= max_chars:
        return [{"text": composed, "clause_ref": None, "point_refs": [], "chunk_type": chunk_type}]

    body_limit = max(max_chars - len(heading) - 2, 100)
    return [
        {
            "text": compose_section_chunk_text(heading, [piece]),
            "clause_ref": None,
            "point_refs": [],
            "chunk_type": chunk_type,
        }
        for piece in split_text_for_chunking(body_text, body_limit)
    ]


def build_clause_chunk_records(
    *,
    heading: str,
    article_intro_parts: Sequence[str],
    clause: LegalClauseGroup,
    max_chars: int,
) -> list[dict[str, object]]:
    prefix_parts = [*article_intro_parts, *clause.intro_parts]

    if not clause.points:
        clause_text = "\n\n".join(part.strip() for part in clause.intro_parts if part and part.strip())
        prefix = list(article_intro_parts)
        part_groups = split_parts_for_chunking(
            heading=heading,
            prefix_parts=prefix,
            body_text=clause_text,
            max_chars=max_chars,
        )
        return [
            {
                "text": compose_section_chunk_text(heading, parts),
                "clause_ref": clause.clause_ref,
                "point_refs": [],
                "chunk_type": "clause" if len(part_groups) == 1 else "clause_part",
            }
            for parts in part_groups
        ]

    all_point_refs = [point.point_ref for point in clause.points]
    all_parts = [*prefix_parts, *flatten_point_parts(clause.points)]
    if len(compose_section_chunk_text(heading, all_parts)) <= max_chars:
        return [
            {
                "text": compose_section_chunk_text(heading, all_parts),
                "clause_ref": clause.clause_ref,
                "point_refs": all_point_refs,
                "chunk_type": "clause",
            }
        ]

    chunk_records: list[dict[str, object]] = []
    pending_points: list[LegalPointGroup] = []

    def append_pending() -> None:
        nonlocal pending_points
        if not pending_points:
            return
        point_refs = [point.point_ref for point in pending_points]
        parts = [*prefix_parts, *flatten_point_parts(pending_points)]
        chunk_records.append(
            {
                "text": compose_section_chunk_text(heading, parts),
                "clause_ref": clause.clause_ref,
                "point_refs": point_refs,
                "chunk_type": "clause_part",
            }
        )
        pending_points = []

    for point in clause.points:
        candidate_points = [*pending_points, point]
        candidate_parts = [*prefix_parts, *flatten_point_parts(candidate_points)]
        if pending_points and len(compose_section_chunk_text(heading, candidate_parts)) > max_chars:
            append_pending()

        single_point_parts = [*prefix_parts, *point.parts]
        if len(compose_section_chunk_text(heading, single_point_parts)) > max_chars:
            append_pending()
            point_text = "\n\n".join(part.strip() for part in point.parts if part and part.strip())
            for parts in split_parts_for_chunking(
                heading=heading,
                prefix_parts=prefix_parts,
                body_text=point_text,
                max_chars=max_chars,
            ):
                chunk_records.append(
                    {
                        "text": compose_section_chunk_text(heading, parts),
                        "clause_ref": clause.clause_ref,
                        "point_refs": [point.point_ref],
                        "chunk_type": "clause_part",
                    }
                )
            continue

        pending_points.append(point)

    append_pending()
    return chunk_records


def chunk_sections(sections: list[SectionRecord], max_chars: int = 1200) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []

    for section in sections:
        if section.section_id.endswith("-preamble"):
            continue

        raw_paragraphs = split_paragraphs(section.text)
        if not raw_paragraphs:
            continue

        heading = section.heading.strip()
        body_paragraphs = raw_paragraphs
        if raw_paragraphs and raw_paragraphs[0] == heading:
            body_paragraphs = raw_paragraphs[1:]

        article_intro_parts, clauses = parse_clause_groups(body_paragraphs)
        chunk_records: list[dict[str, object]] = []

        if section.article_number in SEQUENTIAL_CHUNK_ARTICLES:
            chunk_records.extend(
                build_article_only_chunk_records(
                    heading=heading,
                    body_parts=body_paragraphs,
                    max_chars=max_chars,
                    chunk_type="article_sequential",
                )
            )
        elif not clauses:
            chunk_records.extend(
                build_article_only_chunk_records(
                    heading=heading,
                    body_parts=body_paragraphs,
                    max_chars=max_chars,
                    chunk_type="article_full",
                )
            )
        else:
            if article_intro_parts:
                chunk_records.extend(
                    build_article_only_chunk_records(
                        heading=heading,
                        body_parts=article_intro_parts,
                        max_chars=max_chars,
                        chunk_type="article_intro",
                    )
                )

            for clause in clauses:
                chunk_records.extend(
                    build_clause_chunk_records(
                        heading=heading,
                        article_intro_parts=article_intro_parts,
                        clause=clause,
                        max_chars=max_chars,
                    )
                )

        section_chunks: list[dict[str, object]] = []
        for index, record in enumerate(chunk_records, start=1):
            section_chunks.append(
                make_section_chunk(
                    section=section,
                    index=index,
                    text=str(record["text"]),
                    clause_ref=str(record["clause_ref"]) if record.get("clause_ref") else None,
                    point_refs=record.get("point_refs") if isinstance(record.get("point_refs"), list) else [],
                    chunk_type=str(record["chunk_type"]),
                )
            )

        chunks.extend(assign_parent_chunk_ids(dedupe_chunk_ids(section_chunks)))

    return chunks


def process_document(
    pdf_path: Path,
    cleaned_dir: Path,
    chunks_dir: Path,
    metadata_dir: Path,
    text_ratio_threshold: float = 0.2,
) -> dict[str, object]:
    document_id = slugify_text(pdf_path.stem)
    document_title = pdf_path.stem
    page_records, stats = build_page_records(pdf_path)
    total_chars = sum(len(record.text) for record in page_records)
    text_ratio = round(stats["text_page_count"] / max(stats["page_count"], 1), 3)
    status = "ready" if stats["text_page_count"] and text_ratio >= text_ratio_threshold and total_chars > 400 else "needs_ocr"

    cleaned_path = cleaned_dir / f"{document_id}.txt"
    chunks_path = chunks_dir / f"{document_id}.jsonl"
    metadata_path = metadata_dir / f"{document_id}.json"

    warnings: list[str] = []
    sections: list[SectionRecord] = []
    chunks: list[dict[str, object]] = []

    if status == "ready":
        cleaned_text = build_cleaned_text(page_records)
        cleaned_path.write_text(cleaned_text, encoding="utf-8")
        sections = split_sections(page_records, document_id=document_id, document_title=document_title)
        chunks = chunk_sections(sections)
        with chunks_path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                payload = {
                    "document_id": document_id,
                    "document_title": document_title,
                    "source_kind": "raw_pdf",
                    "source_path": str(pdf_path.as_posix()),
                    **enrich_chunk(chunk=chunk, document_title=document_title, source_kind="raw_pdf"),
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    else:
        warnings.append("Document appears to be scan-based or lacks extractable text. OCR is required before RAG indexing.")

    metadata = {
        "document_id": document_id,
        "document_title": document_title,
        "source_path": str(pdf_path.as_posix()),
        "source_kind": "raw_pdf",
        "status": status,
        "page_count": stats["page_count"],
        "text_page_count": stats["text_page_count"],
        "text_ratio": text_ratio,
        "total_characters": total_chars,
        "cleaned_text_path": str(cleaned_path.as_posix()) if status == "ready" else None,
        "chunks_path": str(chunks_path.as_posix()) if status == "ready" else None,
        "section_count": len(sections),
        "chunk_count": len(chunks),
        "warnings": warnings,
        "pages_with_text": [record.page_number for record in page_records if record.text],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def process_curated_text(
    text_path: Path,
    chunks_dir: Path,
    metadata_dir: Path,
) -> dict[str, object]:
    document_id = slugify_text(text_path.stem)
    source_text = text_path.read_text(encoding="utf-8")
    page_records = build_page_records_from_text(source_text)
    cleaned_text = build_cleaned_text(page_records)
    document_title = infer_document_title(cleaned_text, fallback_title=text_path.stem)
    sections = split_sections(page_records, document_id=document_id, document_title=document_title)
    chunks = chunk_sections(sections)

    chunks_path = chunks_dir / f"{document_id}.jsonl"
    metadata_path = metadata_dir / f"{document_id}.json"

    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            payload = {
                "document_id": document_id,
                "document_title": document_title,
                "source_kind": "curated_text",
                "source_path": str(text_path.as_posix()),
                **enrich_chunk(chunk=chunk, document_title=document_title, source_kind="curated_text"),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    metadata = {
        "document_id": document_id,
        "document_title": document_title,
        "source_path": str(text_path.as_posix()),
        "source_kind": "curated_text",
        "status": "ready",
        "page_count": 1,
        "text_page_count": 1,
        "text_ratio": 1.0,
        "total_characters": len(cleaned_text),
        "cleaned_text_path": str(text_path.as_posix()),
        "chunks_path": str(chunks_path.as_posix()),
        "section_count": len(sections),
        "chunk_count": len(chunks),
        "warnings": [],
        "pages_with_text": [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def build_corpus(
    raw_dir: Path,
    cleaned_dir: Path,
    chunks_dir: Path,
    metadata_dir: Path,
    curated_text_paths: list[Path] | None = None,
) -> dict[str, object]:
    raw_dir = raw_dir.resolve()
    cleaned_dir = cleaned_dir.resolve()
    chunks_dir = chunks_dir.resolve()
    metadata_dir = metadata_dir.resolve()

    cleaned_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    curated_text_paths = [path.resolve() for path in (curated_text_paths or [])]
    curated_text_paths, curated_selection_warnings = select_curated_text_sources(curated_text_paths)
    curated_document_ids = {slugify_text(path.stem) for path in curated_text_paths}
    pdf_paths = sorted(raw_dir.glob("*.pdf"))
    documents = [
        process_document(
            pdf_path=pdf_path,
            cleaned_dir=cleaned_dir,
            chunks_dir=chunks_dir,
            metadata_dir=metadata_dir,
        )
        for pdf_path in pdf_paths
        if slugify_text(pdf_path.stem) not in curated_document_ids
    ]

    for curated_text_path in curated_text_paths:
        documents.append(
            process_curated_text(
                text_path=curated_text_path,
                chunks_dir=chunks_dir,
                metadata_dir=metadata_dir,
            )
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(documents),
        "ready_documents": sum(1 for doc in documents if doc["status"] == "ready"),
        "needs_ocr_documents": sum(1 for doc in documents if doc["status"] == "needs_ocr"),
        "documents": documents,
        "warnings": curated_selection_warnings,
    }

    manifest_path = metadata_dir / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


__all__ = [
    "ARTICLE_RE",
    "PART_RE",
    "SectionRecord",
    "build_curated_chunk_records",
    "build_corpus",
    "build_cleaned_text",
    "build_page_records",
    "chunk_sections",
    "infer_document_title",
    "infer_chunk_taxonomy",
    "normalize_extracted_text",
    "normalize_for_matching",
    "pack_text_units",
    "process_document",
    "process_curated_text",
    "resolve_curated_legal_chunk_paths",
    "split_by_nearest_whitespace",
    "split_by_sentences",
    "slugify_text",
    "split_sections",
    "split_text_for_chunking",
    "summarize_legal_chunks",
    "write_legal_chunk_artifacts",
]
