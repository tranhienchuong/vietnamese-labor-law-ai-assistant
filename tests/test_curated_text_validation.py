from __future__ import annotations

import json
from pathlib import Path

from src.vn_labor_law_ai_assistant.curated_text_validation import (
    SUGGESTED_CIVIL_PROCEDURE_FILTERED_FILE,
    infer_curated_document_title,
    infer_document_id,
    validate_curated_directory,
    validate_curated_text,
    write_validation_artifacts,
)


def write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_legal_marker_detection_includes_subpoints(tmp_path: Path) -> None:
    path = write_text(
        tmp_path / "thong_tu_10_2020_tt_bldtbxh_clean.txt",
        "\n".join(
            [
                "THÔNG TƯ",
                "Quy định thử nghiệm",
                "Chương I",
                "QUY ĐỊNH CHUNG",
                "Mục 1. NỘI DUNG",
                "Điều 1. Phạm vi điều chỉnh",
                "1. Khoản đầu tiên.",
                "a) Điểm a.",
                "b1) Tiểu điểm b1.",
                "a.1) Tiểu điểm a.1.",
            ]
        ),
    )

    document = validate_curated_text(path)

    assert len(document["detected_chapters"]) == 1
    assert len(document["detected_sections"]) == 1
    assert len(document["detected_articles"]) == 1
    assert len(document["detected_clauses"]) == 1
    assert [point["label"] for point in document["detected_points"]] == ["a", "b1", "a.1"]
    assert [point["kind"] for point in document["detected_points"]] == [
        "point",
        "subpoint",
        "subpoint",
    ]


def test_document_id_inference_uses_legal_identifiers(tmp_path: Path) -> None:
    assert (
        infer_document_id(tmp_path / "45_2019_QH14.txt")
        == "45-2019-qh14"
    )
    assert (
        infer_document_id(tmp_path / "nghi_dinh_145_2020_nd_cp_clean.txt")
        == "nghi-dinh-145-2020-nd-cp"
    )
    assert (
        infer_document_id(tmp_path / "thong_tu_10_2020_tt_bldtbxh_clean.txt")
        == "thong-tu-10-2020-tt-bldtbxh"
    )
    assert (
        infer_document_id(tmp_path / "bo_luat_92_2015_qh13_to_tung_dan_su_2015_clean.txt")
        == "92-2015-qh13"
    )


def test_document_title_inference_uses_signature_and_subtitle(tmp_path: Path) -> None:
    path = tmp_path / "thong_tu_10_2020_tt_bldtbxh_clean.txt"
    text = "\n".join(
        [
            "BỘ LAO ĐỘNG - THƯƠNG BINH VÀ XÃ HỘI",
            "Số: 10/2020/TT-BLĐTBXH",
            "THÔNG TƯ",
            "Quy định chi tiết về hợp đồng lao động",
            "Căn cứ Bộ luật Lao động ngày 20 tháng 11 năm 2019;",
        ]
    )

    assert (
        infer_curated_document_title(text, path)
        == "Thông tư 10/2020/TT-BLĐTBXH - Quy định chi tiết về hợp đồng lao động"
    )


def test_warning_generation_for_civil_procedure_code(tmp_path: Path) -> None:
    path = write_text(
        tmp_path / "bo_luat_92_2015_qh13_to_tung_dan_su_2015_clean.txt",
        "\n".join(
            [
                "Bộ Luật Tố tụng dân sự số 92/2015/QH13, ngày 25/11/2015 của Quốc hội",
                "CHƯƠNG I",
                "NHIỆM VỤ VÀ HIỆU LỰC",
                "Điều 32. Những tranh chấp về lao động thuộc thẩm quyền giải quyết của",
                "Tòa án",
                "1. Tranh chấp lao động cá nhân giữa người lao động với người sử dụng lao động.",
            ]
        ),
    )

    document = validate_curated_text(path)

    assert any("should not be fully included" in warning for warning in document["warnings"])
    assert document["suggested_filtered_file"] == SUGGESTED_CIVIL_PROCEDURE_FILTERED_FILE
    assert document["labor_related_units"]["articles"][0]["article_number"] == "32"


def test_validation_output_format(tmp_path: Path) -> None:
    input_dir = tmp_path / "curated"
    output_dir = tmp_path / "validation"
    input_dir.mkdir()
    write_text(
        input_dir / "45_2019_QH14.txt",
        "\n".join(
            [
                "Chương I",
                "NHỮNG QUY ĐỊNH CHUNG",
                "Điều 1. Phạm vi điều chỉnh",
                "1. Người lao động.",
            ]
        ),
    )

    report = validate_curated_directory(input_dir, expected_filenames=("45_2019_QH14.txt",))
    json_path, markdown_path = write_validation_artifacts(report, output_dir)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert markdown_path.exists()
    assert payload["document_count"] == 1
    document = payload["documents"][0]
    for key in [
        "document_id",
        "document_title",
        "document_type",
        "character_count",
        "detected_chapters",
        "detected_sections",
        "detected_articles",
        "detected_clauses",
        "detected_points",
        "possible_ocr_errors",
        "possible_broken_headings",
        "warnings",
    ]:
        assert key in document
