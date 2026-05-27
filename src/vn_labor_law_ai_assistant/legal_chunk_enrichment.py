from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, Sequence

from .corpus_pipeline import normalize_for_matching


TOPIC_RULES: dict[str, tuple[str, ...]] = {
    "hop_dong_lao_dong": (
        "hop dong lao dong",
        "quan he lao dong",
    ),
    "giao_ket_hop_dong_lao_dong": (
        "giao ket hop dong lao dong",
        "nghia vu cung cap thong tin khi giao ket",
    ),
    "noi_dung_hop_dong_lao_dong": (
        "noi dung hop dong lao dong",
        "noi dung chu yeu cua hop dong lao dong",
        "phu luc hop dong lao dong",
    ),
    "cham_dut_hop_dong_lao_dong": (
        "cham dut hop dong lao dong",
        "hop dong lao dong cham dut",
        "trach nhiem khi cham dut hop dong lao dong",
    ),
    "don_phuong_cham_dut": (
        "don phuong cham dut",
        "quyen don phuong cham dut",
        "don phuong cham dut hop dong lao dong",
    ),
    "tro_cap_thoi_viec": (
        "tro cap thoi viec",
    ),
    "tro_cap_mat_viec": (
        "tro cap mat viec",
        "tro cap mat viec lam",
    ),
    "tien_luong": (
        "tien luong",
        "muc luong",
        "bang luong",
        "tra luong",
        "luong toi thieu",
    ),
    "thoi_gio_lam_viec_nghi_ngoi": (
        "thoi gio lam viec",
        "thoi gio nghi ngoi",
        "nghi hang nam",
        "nghi le",
    ),
    "lam_them_gio": (
        "lam them gio",
        "gio lam them",
        "so gio lam them",
    ),
    "ky_luat_lao_dong": (
        "ky luat lao dong",
        "xu ly ky luat lao dong",
        "noi quy lao dong",
        "sa thai",
    ),
    "trach_nhiem_vat_chat": (
        "trach nhiem vat chat",
        "boi thuong thiet hai",
        "lam mat dung cu",
        "lam hu hong dung cu",
    ),
    "lao_dong_chua_thanh_nien": (
        "lao dong chua thanh nien",
        "nguoi chua du 18 tuoi",
        "nguoi chua du 15 tuoi",
        "tu du 13 tuoi den chua du 15 tuoi",
    ),
    "lao_dong_nu": (
        "lao dong nu",
        "binh dang gioi",
        "nguoi lao dong nu",
    ),
    "thai_san": (
        "thai san",
        "mang thai",
        "nuoi con duoi 12 thang",
        "nghi thai san",
    ),
    "tuoi_nghi_huu": (
        "tuoi nghi huu",
        "nghi huu",
        "huong luong huu",
    ),
    "an_toan_ve_sinh_lao_dong": (
        "an toan ve sinh lao dong",
        "an toan lao dong",
        "ve sinh lao dong",
        "tai nan lao dong",
        "benh nghe nghiep",
    ),
    "doi_thoai_tai_noi_lam_viec": (
        "doi thoai tai noi lam viec",
        "noi dung doi thoai",
        "to chuc doi thoai",
    ),
    "thuong_luong_tap_the": (
        "thuong luong tap the",
        "thoa uoc lao dong tap the",
        "ky ket thoa uoc lao dong tap the",
    ),
    "tranh_chap_lao_dong": (
        "tranh chap lao dong",
        "tranh chap lao dong ca nhan",
        "tranh chap lao dong tap the",
    ),
    "to_tung_lao_dong": (
        "to tung lao dong",
        "toa an",
        "khoi kien",
        "bo luat to tung dan su",
    ),
}


ACTOR_RULES: dict[str, tuple[str, ...]] = {
    "nguoi_lao_dong": (
        "nguoi lao dong",
        "ben nguoi lao dong",
    ),
    "nguoi_su_dung_lao_dong": (
        "nguoi su dung lao dong",
        "doanh nghiep",
        "co quan, to chuc, hop tac xa",
    ),
    "lao_dong_nu": (
        "lao dong nu",
        "nguoi lao dong nu",
        "mang thai",
        "nuoi con duoi 12 thang",
    ),
    "lao_dong_chua_thanh_nien": (
        "lao dong chua thanh nien",
        "nguoi chua thanh nien",
        "nguoi chua du 18 tuoi",
    ),
    "nguoi_chua_du_15_tuoi": (
        "nguoi chua du 15 tuoi",
        "chua du 15 tuoi",
    ),
    "nguoi_tu_du_13_den_chua_du_15_tuoi": (
        "tu du 13 tuoi den chua du 15 tuoi",
        "du 13 tuoi den chua du 15 tuoi",
    ),
    "nguoi_lao_dong_nuoc_ngoai": (
        "nguoi lao dong nuoc ngoai",
        "la nguoi nuoc ngoai",
        "nguoi nuoc ngoai lam viec",
    ),
    "to_chuc_dai_dien_nguoi_lao_dong": (
        "to chuc dai dien nguoi lao dong",
        "cong doan",
    ),
    "hoa_giai_vien_lao_dong": (
        "hoa giai vien lao dong",
    ),
    "toa_an": (
        "toa an",
        "toa an nhan dan",
    ),
}


ISSUE_TYPE_RULES: dict[str, tuple[str, ...]] = {
    "giao_ket_hop_dong": (
        "giao ket hop dong lao dong",
        "nghia vu giao ket hop dong",
        "nghia vu cung cap thong tin khi giao ket",
    ),
    "noi_dung_hop_dong": (
        "noi dung hop dong lao dong",
        "noi dung chu yeu cua hop dong lao dong",
    ),
    "loai_hop_dong": (
        "loai hop dong lao dong",
        "hop dong lao dong khong xac dinh thoi han",
        "hop dong lao dong xac dinh thoi han",
    ),
    "don_phuong_cham_dut": (
        "don phuong cham dut",
        "quyen don phuong cham dut",
    ),
    "boi_thuong": (
        "boi thuong",
        "boi thuong thiet hai",
        "khoan tien tuong ung",
    ),
    "tro_cap": (
        "tro cap thoi viec",
        "tro cap mat viec",
        "tro cap mat viec lam",
    ),
    "sa_thai": (
        "sa thai",
        "hinh thuc xu ly ky luat sa thai",
    ),
    "ky_luat": (
        "ky luat lao dong",
        "xu ly ky luat lao dong",
        "thoi hieu xu ly ky luat lao dong",
    ),
    "tuoi_nghi_huu": (
        "tuoi nghi huu",
        "nghi huu",
    ),
    "lam_them_gio": (
        "lam them gio",
        "gio lam them",
    ),
    "lao_dong_chua_thanh_nien": (
        "lao dong chua thanh nien",
        "nguoi chua du 18 tuoi",
        "nguoi chua du 15 tuoi",
    ),
    "tranh_chap_lao_dong": (
        "tranh chap lao dong",
        "tranh chap lao dong ca nhan",
        "tranh chap lao dong tap the",
    ),
    "khoi_kien": (
        "khoi kien",
        "quyen khoi kien",
        "thoi hieu khoi kien",
    ),
    "hoa_giai": (
        "hoa giai",
        "hoa giai vien lao dong",
        "thu tuc hoa giai",
    ),
    "tham_quyen_toa_an": (
        "tham quyen cua toa an",
        "tham quyen toa an",
        "toa an co tham quyen",
    ),
}


DOCUMENT_ALIASES: dict[str, str] = {
    "bo luat lao dong": "45-2019-qh14",
    "bo luat lao dong nam 2019": "45-2019-qh14",
    "luat lao dong 2019": "45-2019-qh14",
    "nghi dinh 145": "nghi-dinh-145-2020-nd-cp",
    "nghi dinh 145/2020": "nghi-dinh-145-2020-nd-cp",
    "nghi dinh 135": "nghi-dinh-135-2020-nd-cp",
    "nghi dinh 135/2020": "nghi-dinh-135-2020-nd-cp",
    "thong tu 09": "thong-tu-09-2020-tt-bldtbxh",
    "thong tu 09/2020": "thong-tu-09-2020-tt-bldtbxh",
    "thong tu 10": "thong-tu-10-2020-tt-bldtbxh",
    "thong tu 10/2020": "thong-tu-10-2020-tt-bldtbxh",
    "bo luat to tung dan su": "92-2015-qh13-labor-only",
    "blttds": "92-2015-qh13-labor-only",
}


DOCUMENT_METADATA_BY_TYPE: dict[str, dict[str, object]] = {
    "bo_luat": {
        "issuing_authority": "Quốc hội",
        "normative_rank": 1,
        "rank_label": "highest",
        "effective_level": "national",
        "status": "in_effect",
    },
    "luat": {
        "issuing_authority": "Quốc hội",
        "normative_rank": 1,
        "rank_label": "highest",
        "effective_level": "national",
        "status": "in_effect",
    },
    "nghi_dinh": {
        "issuing_authority": "Chính phủ",
        "normative_rank": 2,
        "rank_label": "middle",
        "effective_level": "national",
        "status": "in_effect",
    },
    "thong_tu": {
        "issuing_authority": "Bộ trưởng / Thủ trưởng cơ quan ngang Bộ",
        "normative_rank": 3,
        "rank_label": "lowest",
        "effective_level": "national",
        "status": "in_effect",
    },
}


DOCUMENT_TYPE_BY_ID: dict[str, str] = {
    "45-2019-qh14": "bo_luat",
    "92-2015-qh13-labor-only": "bo_luat",
    "nghi-dinh-145-2020-nd-cp": "nghi_dinh",
    "nghi-dinh-135-2020-nd-cp": "nghi_dinh",
    "thong-tu-09-2020-tt-bldtbxh": "thong_tu",
    "thong-tu-10-2020-tt-bldtbxh": "thong_tu",
}


def dedupe_preserve_order(values: Sequence[object]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def collect_rule_labels(rule_map: Mapping[str, Sequence[str]], text: str) -> list[str]:
    normalized = normalize_for_matching(text)
    return [
        label
        for label, keywords in rule_map.items()
        if any(normalize_for_matching(keyword) in normalized for keyword in keywords)
    ]


def infer_enriched_taxonomy(chunk: Mapping[str, object]) -> tuple[list[str], list[str], list[str]]:
    context_parts = [
        chunk.get("document_title"),
        chunk.get("part_heading"),
        chunk.get("chapter_heading"),
        chunk.get("section_heading"),
        chunk.get("article_title"),
        chunk.get("appendix_heading"),
        chunk.get("heading"),
        chunk.get("text"),
        chunk.get("retrieval_text"),
    ]
    context = " ".join(str(part) for part in context_parts if part)
    topics = dedupe_preserve_order(
        [*(chunk.get("topic") if isinstance(chunk.get("topic"), list) else []), *collect_rule_labels(TOPIC_RULES, context)]
    )
    actors = dedupe_preserve_order(
        [*(chunk.get("actor") if isinstance(chunk.get("actor"), list) else []), *collect_rule_labels(ACTOR_RULES, context)]
    )
    issue_types = dedupe_preserve_order(
        [
            *(chunk.get("issue_type") if isinstance(chunk.get("issue_type"), list) else []),
            *collect_rule_labels(ISSUE_TYPE_RULES, context),
        ]
    )
    return topics, actors, issue_types


def match_document_alias(text: str) -> str | None:
    normalized = normalize_for_matching(text)
    for alias, document_id in DOCUMENT_ALIASES.items():
        if normalize_for_matching(alias) in normalized:
            return document_id
    return None


def infer_document_type(chunk: Mapping[str, object]) -> str:
    document_id = str(chunk.get("document_id") or "")
    existing_type = str(chunk.get("document_type") or "")
    if existing_type in DOCUMENT_METADATA_BY_TYPE:
        return existing_type
    if document_id in DOCUMENT_TYPE_BY_ID:
        return DOCUMENT_TYPE_BY_ID[document_id]
    matched_document_id = match_document_alias(
        " ".join(str(chunk.get(field) or "") for field in ("document_title", "citation_text", "text"))
    )
    if matched_document_id:
        return DOCUMENT_TYPE_BY_ID[matched_document_id]
    return existing_type or "unknown"


def build_document_hierarchy(chunk: Mapping[str, object]) -> dict[str, object]:
    return {
        "document_id": chunk.get("document_id"),
        "document_title": chunk.get("document_title"),
        "part_number": chunk.get("part_number"),
        "part_heading": chunk.get("part_heading"),
        "chapter_heading": chunk.get("chapter_heading"),
        "section_heading": chunk.get("section_heading"),
        "article_number": chunk.get("article_number"),
        "article_title": chunk.get("article_title"),
        "clause_ref": chunk.get("clause_ref"),
        "point_ref": chunk.get("point_ref"),
        "point_refs": chunk.get("point_refs") or [],
        "appendix_id": chunk.get("appendix_id"),
        "appendix_heading": chunk.get("appendix_heading"),
        "level": chunk.get("level"),
    }


def enrich_legal_chunk(chunk: Mapping[str, object]) -> dict[str, object]:
    enriched = dict(chunk)
    topic, actor, issue_type = infer_enriched_taxonomy(enriched)
    document_type = infer_document_type(enriched)
    document_metadata = DOCUMENT_METADATA_BY_TYPE.get(document_type, {})

    enriched["topic"] = topic
    enriched["actor"] = actor
    enriched["issue_type"] = issue_type
    enriched["document_type"] = document_type
    enriched.update(document_metadata)
    enriched["document_hierarchy"] = build_document_hierarchy(enriched)
    return enriched


def enrich_legal_chunks(chunks: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [enrich_legal_chunk(chunk) for chunk in chunks]


def _distribution(chunks: Sequence[Mapping[str, object]], field: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for chunk in chunks:
        value = chunk.get(field)
        if isinstance(value, list):
            counter.update(str(item) for item in value if item)
        elif value is not None and value != "":
            counter[str(value)] += 1
    return dict(sorted(counter.items()))


def _missing_list(chunks: Sequence[Mapping[str, object]], field: str) -> list[str]:
    missing: list[str] = []
    for chunk in chunks:
        value = chunk.get(field)
        is_missing = value is None or value == "" or value == []
        if is_missing:
            missing.append(str(chunk.get("chunk_id") or ""))
    return missing


def summarize_enriched_chunks(chunks: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chunk_count": len(chunks),
        "chunks_missing_topic": _missing_list(chunks, "topic"),
        "chunks_missing_actor": _missing_list(chunks, "actor"),
        "chunks_missing_issue_type": _missing_list(chunks, "issue_type"),
        "topic_distribution": _distribution(chunks, "topic"),
        "actor_distribution": _distribution(chunks, "actor"),
        "issue_type_distribution": _distribution(chunks, "issue_type"),
        "document_type_distribution": _distribution(chunks, "document_type"),
        "normative_rank_distribution": _distribution(chunks, "normative_rank"),
    }


def render_enriched_summary_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# Enriched Legal Chunks Summary",
        "",
        f"- Chunk count: {summary['chunk_count']}",
        f"- Chunks missing topic: {len(summary['chunks_missing_topic'])}",
        f"- Chunks missing actor: {len(summary['chunks_missing_actor'])}",
        f"- Chunks missing issue_type: {len(summary['chunks_missing_issue_type'])}",
    ]

    table_specs = [
        ("Topic Distribution", "topic_distribution", "Topic"),
        ("Actor Distribution", "actor_distribution", "Actor"),
        ("Issue Type Distribution", "issue_type_distribution", "Issue type"),
        ("Document Type Distribution", "document_type_distribution", "Document type"),
        ("Normative Rank Distribution", "normative_rank_distribution", "Normative rank"),
    ]
    for title, key, label in table_specs:
        lines.extend(["", f"## {title}", "", f"| {label} | Chunks |", "| --- | ---: |"])
        distribution = summary[key]
        if isinstance(distribution, Mapping):
            for value, count in sorted(distribution.items(), key=lambda item: str(item[0])):
                lines.append(f"| {value} | {count} |")
    return "\n".join(lines).strip() + "\n"


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_enriched_chunk_artifacts(
    chunks: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = output_dir / "legal_chunks_enriched.jsonl"
    summary_json_path = output_dir / "legal_chunks_enriched_summary.json"
    summary_md_path = output_dir / "legal_chunks_enriched_summary.md"

    with chunks_path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    summary = summarize_enriched_chunks(chunks)
    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_md_path.write_text(
        render_enriched_summary_markdown(summary),
        encoding="utf-8",
    )
    return chunks_path, summary_json_path, summary_md_path


__all__ = [
    "ACTOR_RULES",
    "DOCUMENT_ALIASES",
    "DOCUMENT_METADATA_BY_TYPE",
    "ISSUE_TYPE_RULES",
    "TOPIC_RULES",
    "enrich_legal_chunk",
    "enrich_legal_chunks",
    "infer_document_type",
    "match_document_alias",
    "read_jsonl",
    "render_enriched_summary_markdown",
    "summarize_enriched_chunks",
    "write_enriched_chunk_artifacts",
]
