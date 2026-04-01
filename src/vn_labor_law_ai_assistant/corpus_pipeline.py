from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import unicodedata

import fitz
from pypdf import PdfReader


ARTICLE_RE = re.compile(r"^Điều\s+(?P<number>\d+[A-Za-z]?)\.\s*(?P<title>.+)$")
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

TOPIC_RULES = {
    "cham_dut_hop_dong_lao_dong": ["cham dut hop dong lao dong"],
    "don_phuong_cham_dut": ["don phuong cham dut"],
    "bao_truoc": ["bao truoc", "thoi han bao truoc"],
    "tro_cap": ["tro cap", "tro cap thoi viec", "tro cap mat viec"],
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


def list_pages_for_citation(source_pages: list[int]) -> str | None:
    if not source_pages:
        return None
    if len(source_pages) == 1:
        return f"tr. {source_pages[0]}"
    return f"tr. {source_pages[0]}-{source_pages[-1]}"


def build_citation_text(
    document_title: str,
    article_number: str | None,
    article_title: str | None,
    clause_ref: str | None,
    point_ref: str | None,
    source_pages: list[int],
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
    if point_ref:
        reference_parts.append(f"điểm {point_ref}")
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
    chunk_index = int(chunk["chunk_index"])
    heading = str(chunk["heading"])
    text = str(chunk["text"])
    article_number = chunk["article_number"]
    article_title = chunk["article_title"]
    section_heading = chunk["section_heading"]
    chapter_heading = chunk["chapter_heading"]
    clause_ref = str(chunk["clause_ref"]) if chunk.get("clause_ref") else None
    point_ref = str(chunk["point_ref"]) if chunk.get("point_ref") else None

    source_pages = list(chunk["source_pages"]) if source_kind == "raw_pdf" else []
    body_text = extract_chunk_body(text, heading)
    level = infer_chunk_level(clause_ref=clause_ref, point_ref=point_ref)
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
    )
    retrieval_text = build_retrieval_text(
        document_title=document_title,
        heading=heading,
        chapter_heading=str(chapter_heading) if chapter_heading else None,
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
        "parent_chunk_id": parent_chunk_id,
        "citation_text": citation_text,
        "retrieval_text": retrieval_text,
        "topic": topic,
        "actor": actor,
        "issue_type": issue_type,
    }

    if source_pages:
        enriched["source_pages"] = source_pages
    else:
        enriched.pop("source_pages", None)

    return enriched


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
    current_chapter: str | None = None
    current_heading_group: str | None = None
    pending_chapter_line: str | None = None

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
            )
        )
        current_section = None

    for entry in paragraph_records:
        page_number = int(entry["page_number"])
        lines = [line.strip() for line in str(entry["text"]).splitlines() if line.strip()]

        for line in lines:
            chapter_match = CHAPTER_RE.match(line)
            section_match = SECTION_RE.match(line)
            group_match = GROUP_RE.match(line)
            article_match = ARTICLE_RE.match(line)

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
                current_section = {
                    "section_id": f"{document_id}-dieu-{article_number}",
                    "heading": line,
                    "article_number": article_number,
                    "article_title": article_title,
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
        )
    ]


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

        body_limit = max(max_chars - len(heading) - 2, 100)
        legal_units = build_legal_units(body_paragraphs, body_limit) if body_paragraphs else []

        def compose_chunk(parts: list[str]) -> str:
            if not parts:
                return heading
            return "\n\n".join([heading, *parts]).strip()

        pending_parts: list[str] = []
        pending_clause_ref: str | None = None
        pending_point_ref: str | None = None

        chunk_records: list[dict[str, object]] = []

        if not legal_units:
            chunk_records.append({"clause_ref": None, "point_ref": None, "text": heading})

        def flush_pending() -> None:
            nonlocal pending_parts, pending_clause_ref, pending_point_ref
            if not pending_parts and pending_clause_ref is None and pending_point_ref is None:
                return

            text = compose_chunk(pending_parts)
            chunk_records.append(
                {
                    "clause_ref": pending_clause_ref,
                    "point_ref": pending_point_ref,
                    "text": text,
                }
            )
            pending_parts = []
            pending_clause_ref = None
            pending_point_ref = None

        for unit in legal_units:
            unit_text = str(unit["text"]).strip()
            unit_clause_ref = str(unit["clause_ref"]) if unit.get("clause_ref") else None
            unit_point_ref = str(unit["point_ref"]) if unit.get("point_ref") else None

            if not pending_parts:
                pending_parts = [unit_text] if unit_text else []
                pending_clause_ref = unit_clause_ref
                pending_point_ref = unit_point_ref
                continue

            same_signature = pending_clause_ref == unit_clause_ref and pending_point_ref == unit_point_ref
            candidate_parts = [*pending_parts, unit_text] if unit_text else [*pending_parts]
            candidate_text = compose_chunk(candidate_parts)

            if not same_signature or len(candidate_text) > max_chars:
                flush_pending()
                pending_parts = [unit_text] if unit_text else []
                pending_clause_ref = unit_clause_ref
                pending_point_ref = unit_point_ref
                continue

            if unit_text:
                pending_parts.append(unit_text)

        flush_pending()

        section_chunks: list[dict[str, object]] = []
        for index, record in enumerate(chunk_records, start=1):
            section_chunks.append(
                {
                    "chunk_id": f"{section.section_id}-chunk-{index:02d}",
                    "section_id": section.section_id,
                    "article_number": section.article_number,
                    "article_title": section.article_title,
                    "heading": section.heading,
                    "chapter_heading": section.chapter_heading,
                    "section_heading": section.section_heading,
                    "source_pages": section.source_pages,
                    "chunk_index": index,
                    "char_count": len(str(record["text"])),
                    "text": record["text"],
                    "clause_ref": record["clause_ref"],
                    "point_ref": record["point_ref"],
                }
            )

        chunks.extend(assign_parent_chunk_ids(section_chunks))

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
    }

    manifest_path = metadata_dir / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


__all__ = [
    "ARTICLE_RE",
    "SectionRecord",
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
    "split_by_nearest_whitespace",
    "split_by_sentences",
    "slugify_text",
    "split_sections",
    "split_text_for_chunking",
]
