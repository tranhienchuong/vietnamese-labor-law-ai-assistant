from __future__ import annotations

from dataclasses import asdict, dataclass
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
    return fallback_title


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
        text = str(entry["text"])
        page_number = int(entry["page_number"])
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        first_line = lines[0] if lines else text.strip()
        remainder = "\n".join(lines[1:]).strip()

        chapter_match = CHAPTER_RE.match(first_line)
        if chapter_match:
            current_chapter = first_line
            continue

        heading_group_match = SECTION_RE.match(first_line)
        if heading_group_match:
            current_heading_group = first_line
            continue

        if GROUP_RE.match(first_line):
            current_heading_group = first_line
            continue

        article_match = ARTICLE_RE.match(first_line)
        if article_match:
            flush_current()
            article_number = article_match.group("number")
            article_title = article_match.group("title").strip()
            current_section = {
                "section_id": f"{document_id}-dieu-{article_number}",
                "heading": first_line,
                "article_number": article_number,
                "article_title": article_title,
                "chapter_heading": current_chapter,
                "section_heading": current_heading_group,
                "source_pages": {page_number},
                "paragraphs": [remainder] if remainder else [],
            }
            continue

        if current_section is None:
            preamble.append({"page_number": page_number, "text": text})
            continue

        current_section["paragraphs"].append(text)
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
        raw_paragraphs = split_paragraphs(section.text)
        if not raw_paragraphs:
            continue

        heading = section.heading.strip()
        body_paragraphs = raw_paragraphs
        if raw_paragraphs and raw_paragraphs[0] == heading:
            body_paragraphs = raw_paragraphs[1:]

        if not body_paragraphs:
            body_paragraphs = [heading]

        chunk_texts: list[str] = []
        current_body_parts: list[str] = []

        def compose_chunk(parts: list[str]) -> str:
            if parts == [heading]:
                return heading
            return "\n\n".join([heading, *parts]).strip()

        for paragraph in body_paragraphs:
            candidate_parts = [*current_body_parts, paragraph]
            candidate_text = compose_chunk(candidate_parts)

            if current_body_parts and len(candidate_text) > max_chars:
                chunk_texts.append(compose_chunk(current_body_parts))
                current_body_parts = []

            if len(compose_chunk([paragraph])) > max_chars:
                available = max(max_chars - len(heading) - 2, 50)
                for start in range(0, len(paragraph), available):
                    window = paragraph[start : start + available].strip()
                    if window:
                        chunk_texts.append(compose_chunk([window]))
                continue

            current_body_parts.append(paragraph)

        if current_body_parts:
            chunk_texts.append(compose_chunk(current_body_parts))

        for index, text in enumerate(chunk_texts, start=1):
            chunks.append(
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
                    "char_count": len(text),
                    "text": text,
                }
            )

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
                    "source_path": str(pdf_path.as_posix()),
                    **chunk,
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
                "source_path": str(text_path.as_posix()),
                **chunk,
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
        "pages_with_text": [1],
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

    pdf_paths = sorted(raw_dir.glob("*.pdf"))
    documents = [
        process_document(
            pdf_path=pdf_path,
            cleaned_dir=cleaned_dir,
            chunks_dir=chunks_dir,
            metadata_dir=metadata_dir,
        )
        for pdf_path in pdf_paths
    ]

    for curated_text_path in curated_text_paths or []:
        documents.append(
            process_curated_text(
                text_path=curated_text_path.resolve(),
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
    "normalize_extracted_text",
    "process_document",
    "process_curated_text",
    "slugify_text",
    "split_sections",
]
