from __future__ import annotations

import json
from pathlib import Path

from src.vn_labor_law_ai_assistant.legal_unit_parser import (
    CURATED_LEGAL_UNIT_FILENAMES,
    FULL_BLTTDS_FILENAME,
    build_legal_units_from_directory,
    infer_legal_unit_document_metadata,
    parse_article_spans,
    parse_legal_document,
    resolve_curated_legal_unit_paths,
    write_legal_unit_artifacts,
)


def sample_legal_text() -> str:
    return "\n".join(
        [
            "PHẦN THỨ NHẤT",
            "NHỮNG QUY ĐỊNH CHUNG",
            "",
            "Chương I",
            "QUY ĐỊNH CHUNG",
            "",
            "Mục 1",
            "GIAO KẾT HỢP ĐỒNG LAO ĐỘNG",
            "",
            "Điều 4. Giao kết hợp đồng lao động để sử dụng người chưa đủ 15 tuổi",
            "làm việc",
            "",
            "1. Người sử dụng lao động phải giao kết hợp đồng lao động bằng văn bản.",
            "",
            "a) Điểm a.",
            "",
            "đ) Điểm đ.",
            "",
            "b1) Tiểu điểm b1.",
            "",
            "a.1) Tiểu điểm a.1.",
            "",
            "2. Khoản thứ hai.",
        ]
    )


def write_text(path: Path, text: str = "Điều 1. Phạm vi điều chỉnh\n\n1. Nội dung.") -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_article_detection_and_heading_continuation() -> None:
    articles = parse_article_spans(sample_legal_text())

    assert len(articles) == 1
    assert articles[0].number == "4"
    assert (
        articles[0].title
        == "Giao kết hợp đồng lao động để sử dụng người chưa đủ 15 tuổi làm việc"
    )
    assert articles[0].part and articles[0].part.number == "THỨ NHẤT"
    assert articles[0].chapter and articles[0].chapter.number == "I"
    assert articles[0].section and articles[0].section.number == "1"


def test_quote_prefixed_amended_articles_do_not_split_parent_article() -> None:
    articles = parse_article_spans(
        "\n".join(
            [
                "Điều 219. Sửa đổi, bổ sung một số điều của các luật có liên quan",
                "",
                "“Điều 54. Điều kiện hưởng lương hưu",
                "",
                "“1. Người lao động đủ điều kiện hưởng lương hưu.",
            ]
        )
    )

    assert [article.number for article in articles] == ["219"]
    assert "“Điều 54." in articles[0].text


def test_clause_and_point_detection(tmp_path: Path) -> None:
    path = write_text(tmp_path / "45_2019_QH14.txt", sample_legal_text())
    document = parse_legal_document(path)

    clause_refs = [
        unit["clause_ref"] for unit in document.units if unit["unit_level"] == "clause"
    ]
    point_refs = [
        unit["point_ref"] for unit in document.units if unit["unit_level"] == "point"
    ]

    assert clause_refs == ["1", "2"]
    assert point_refs == ["a", "đ", "b1", "a.1"]
    assert all(unit["article_number"] == "4" for unit in document.units)


def test_document_id_mapping_for_labor_only_blttds(tmp_path: Path) -> None:
    path = tmp_path / "92_2015_QH13_labor_only.txt"
    text = "Điều 32. Những tranh chấp về lao động thuộc thẩm quyền giải quyết của Tòa án"

    metadata = infer_legal_unit_document_metadata(path, text)

    assert metadata.document_id == "92-2015-qh13"
    assert metadata.source_file == "92_2015_QH13_labor_only.txt"
    assert "filtered subset" in metadata.document_title


def test_blttds_labor_only_file_is_used_and_full_file_is_excluded(tmp_path: Path) -> None:
    for filename in CURATED_LEGAL_UNIT_FILENAMES:
        write_text(tmp_path / filename)
    write_text(
        tmp_path / FULL_BLTTDS_FILENAME,
        "Điều 999. Full BLTTDS file should not be selected.",
    )

    selected_paths = resolve_curated_legal_unit_paths(tmp_path)
    selected_names = {path.name for path in selected_paths}
    documents, units = build_legal_units_from_directory(tmp_path)

    assert "92_2015_QH13_labor_only.txt" in selected_names
    assert FULL_BLTTDS_FILENAME not in selected_names
    assert all(unit["source_file"] != FULL_BLTTDS_FILENAME for unit in units)
    assert any(
        document.metadata.source_file == "92_2015_QH13_labor_only.txt"
        for document in documents
    )


def test_legal_unit_artifact_format(tmp_path: Path) -> None:
    input_dir = tmp_path / "curated"
    output_dir = tmp_path / "legal_units"
    input_dir.mkdir()
    for filename in CURATED_LEGAL_UNIT_FILENAMES:
        write_text(input_dir / filename, sample_legal_text())

    documents, units = build_legal_units_from_directory(input_dir)
    jsonl_path, summary_json_path, summary_md_path = write_legal_unit_artifacts(
        documents,
        units,
        output_dir,
        input_dir=input_dir,
    )
    first_record = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    summary = json.loads(summary_json_path.read_text(encoding="utf-8"))

    assert summary_md_path.exists()
    assert summary["document_count"] == len(CURATED_LEGAL_UNIT_FILENAMES)
    assert summary["excluded_files"] == [FULL_BLTTDS_FILENAME]
    for key in [
        "document_id",
        "document_title",
        "document_type",
        "source_file",
        "part_number",
        "part_heading",
        "chapter_number",
        "chapter_heading",
        "section_number",
        "section_heading",
        "article_number",
        "article_title",
        "clause_ref",
        "point_ref",
        "unit_level",
        "text",
    ]:
        assert key in first_record
