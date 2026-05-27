from __future__ import annotations

from src.vn_labor_law_ai_assistant.legal_reference_edges import (
    build_reference_edge_records,
    parse_legal_references,
    summarize_reference_edges,
)
from src.vn_labor_law_ai_assistant.unresolved_reference_report import (
    build_unresolved_reference_report,
)


def chunk(
    *,
    chunk_id: str = "45_2019_QH14_Dieu_1",
    document_id: str = "45-2019-qh14",
    document_type: str = "bo_luat",
    article_number: str = "1",
    clause_ref: str | None = None,
    point_ref: str | None = None,
    level: str = "article",
    text: str = "",
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "document_title": "Bộ luật Lao động 2019",
        "document_type": document_type,
        "article_number": article_number,
        "article_title": "Test",
        "heading": f"Điều {article_number}. Test",
        "clause_ref": clause_ref,
        "point_ref": point_ref,
        "point_refs": [point_ref] if point_ref else [],
        "level": level,
        "citation_text": f"{document_id}, Điều {article_number}",
        "text": text,
        "retrieval_text": "",
    }


def test_blld_article_reference_parsing() -> None:
    refs = parse_legal_references(
        chunk(text="Quy định tại Điều 169 của Bộ luật Lao động.")
    )

    assert refs[0].target_document_id == "45-2019-qh14"
    assert refs[0].target_id == "article:45-2019-qh14:dieu-169"
    assert refs[0].reference_level == "article"


def test_clause_level_reference_parsing() -> None:
    refs = parse_legal_references(
        chunk(text="Theo khoản 1 Điều 21 của Bộ luật Lao động.")
    )

    assert refs[0].target_id == "clause:45-2019-qh14:dieu-21:khoan-1"
    assert refs[0].reference_level == "clause"


def test_point_level_reference_parsing() -> None:
    refs = parse_legal_references(
        chunk(text="Áp dụng theo điểm a khoản 1 Điều 32.")
    )

    assert refs[0].target_id == "point:45-2019-qh14:dieu-32:khoan-1:diem-a"
    assert refs[0].reference_level == "point"


def test_nghi_dinh_reference_parsing() -> None:
    refs = parse_legal_references(
        chunk(text="Thực hiện theo Nghị định 145/2020/NĐ-CP.")
    )

    assert refs[0].target_document_id == "nghi-dinh-145-2020-nd-cp"
    assert refs[0].target_id == "document:nghi-dinh-145-2020-nd-cp"


def test_thong_tu_reference_parsing() -> None:
    refs = parse_legal_references(
        chunk(text="Hồ sơ theo Thông tư 09/2020/TT-BLĐTBXH và Thông tư 10/2020/TT-BLĐTBXH.")
    )

    assert {ref.target_document_id for ref in refs} == {
        "thong-tu-09-2020-tt-bldtbxh",
        "thong-tu-10-2020-tt-bldtbxh",
    }


def test_blttds_reference_parsing() -> None:
    refs = parse_legal_references(
        chunk(text="Thẩm quyền theo Điều 32 Bộ luật Tố tụng dân sự.")
    )

    assert refs[0].target_document_id == "92-2015-qh13-labor-only"
    assert refs[0].target_id == "article:92-2015-qh13-labor-only:dieu-32"


def test_luat_nay_inside_labor_code_article_219_is_external() -> None:
    refs = parse_legal_references(
        chunk(
            article_number="219",
            text="Người lao động quy định tại điểm e khoản 1 Điều 2 của Luật này.",
        )
    )

    assert refs[0].target_document_id == "external-quoted-amended-law"
    assert refs[0].target_id == "point:external-quoted-amended-law:dieu-2:khoan-1:diem-e"


def test_tt09_scope_article_bare_reference_defaults_to_labor_code() -> None:
    refs = parse_legal_references(
        chunk(
            chunk_id="TT09_2020_Dieu_1_Khoan_1",
            document_id="thong-tu-09-2020-tt-bldtbxh",
            document_type="thong_tu",
            article_number="1",
            clause_ref="1",
            level="clause",
            text="Sử dụng người chưa đủ 15 tuổi làm việc theo quy định tại khoản 4 Điều 145.",
        )
    )

    assert refs[0].target_document_id == "45-2019-qh14"
    assert refs[0].target_id == "clause:45-2019-qh14:dieu-145:khoan-4"


def test_tt10_scope_article_multi_clause_reference_defaults_to_labor_code() -> None:
    refs = parse_legal_references(
        chunk(
            chunk_id="TT10_2020_Dieu_1_Khoan_1",
            document_id="thong-tu-10-2020-tt-bldtbxh",
            document_type="thong_tu",
            article_number="1",
            clause_ref="1",
            level="clause",
            text="Nội dung của hợp đồng lao động theo khoản 1, 2 và 3 Điều 21.",
        )
    )

    assert {ref.target_id for ref in refs} == {
        "clause:45-2019-qh14:dieu-21:khoan-1",
        "clause:45-2019-qh14:dieu-21:khoan-2",
        "clause:45-2019-qh14:dieu-21:khoan-3",
    }


def test_nd145_scope_article_bare_reference_defaults_to_labor_code() -> None:
    refs = parse_legal_references(
        chunk(
            chunk_id="ND145_2020_Dieu_1_Khoan_2",
            document_id="nghi-dinh-145-2020-nd-cp",
            document_type="nghi_dinh",
            article_number="1",
            clause_ref="2",
            level="clause",
            text="Hợp đồng lao động theo khoản 1, khoản 2, khoản 3 Điều 21.",
        )
    )

    assert refs[0].target_document_id == "45-2019-qh14"
    assert refs[0].target_id == "clause:45-2019-qh14:dieu-21:khoan-3"


def test_current_decree_reference_overrides_nearby_labor_code_context() -> None:
    refs = parse_legal_references(
        chunk(
            chunk_id="ND145_2020_Dieu_71_Khoan_2_Diem_a",
            document_id="nghi-dinh-145-2020-nd-cp",
            document_type="nghi_dinh",
            article_number="71",
            clause_ref="2",
            point_ref="a",
            level="point",
            text=(
                "Thiệt hại tại khoản 2 Điều 130 của Bộ luật Lao động được quy định như sau: "
                "2. Trong thời hiệu xử lý bồi thường thiệt hại quy định tại Điều 72 Nghị định này."
            ),
        )
    )

    assert any(ref.target_id == "clause:45-2019-qh14:dieu-130:khoan-2" for ref in refs)
    assert any(ref.target_id == "article:nghi-dinh-145-2020-nd-cp:dieu-72" for ref in refs)


def test_current_circular_reference_overrides_nearby_labor_code_context() -> None:
    refs = parse_legal_references(
        chunk(
            chunk_id="TT09_2020_Dieu_3_Khoan_5_Diem_a_b",
            document_id="thong-tu-09-2020-tt-bldtbxh",
            document_type="thong_tu",
            article_number="3",
            clause_ref="5",
            point_ref="b",
            level="point",
            text=(
                "Nơi làm việc không thuộc các trường hợp quy định tại khoản 2 "
                "Điều 147 của Bộ luật Lao động và khoản 2 Điều 9 của Thông tư này."
            ),
        )
    )

    assert any(ref.target_id == "clause:45-2019-qh14:dieu-147:khoan-2" for ref in refs)
    assert any(ref.target_id == "clause:thong-tu-09-2020-tt-bldtbxh:dieu-9:khoan-2" for ref in refs)


def test_list_suffix_document_applies_to_prior_article_references() -> None:
    refs = parse_legal_references(
        chunk(
            chunk_id="ND135_2020_Dieu_7_Khoan_1",
            document_id="nghi-dinh-135-2020-nd-cp",
            document_type="nghi_dinh",
            article_number="7",
            clause_ref="1",
            level="clause",
            text=(
                "Các quy định của chế độ hưu trí theo Điều 54, Điều 55, khoản 1 "
                "Điều 73 của Luật bảo hiểm xã hội được thực hiện theo tuổi nghỉ hưu "
                "tại Điều 169, khoản 1 Điều 219 của Bộ luật Lao động."
            ),
        )
    )

    assert any(ref.target_id == "article:external-luat-bao-hiem-xa-hoi:dieu-54" for ref in refs)
    assert any(ref.target_id == "article:external-luat-bao-hiem-xa-hoi:dieu-55" for ref in refs)
    assert any(ref.target_id == "clause:external-luat-bao-hiem-xa-hoi:dieu-73:khoan-1" for ref in refs)
    assert any(ref.target_id == "article:45-2019-qh14:dieu-169" for ref in refs)
    assert any(ref.target_id == "clause:45-2019-qh14:dieu-219:khoan-1" for ref in refs)


def test_labor_code_suffix_applies_across_reference_list() -> None:
    refs = parse_legal_references(
        chunk(
            chunk_id="ND145_2020_Dieu_89_Khoan_1_Diem_a",
            document_id="nghi-dinh-145-2020-nd-cp",
            document_type="nghi_dinh",
            article_number="89",
            clause_ref="1",
            point_ref="a",
            level="point",
            text=(
                "Quy định về hình thức hợp đồng lao động theo Điều 14 và khoản 1 "
                "Điều 162; nghĩa vụ cung cấp thông tin theo Điều 16; trợ cấp thôi "
                "việc theo Điều 46 của Bộ luật Lao động được thực hiện như sau."
            ),
        )
    )

    assert any(ref.target_id == "article:45-2019-qh14:dieu-14" for ref in refs)
    assert any(ref.target_id == "clause:45-2019-qh14:dieu-162:khoan-1" for ref in refs)
    assert any(ref.target_id == "article:45-2019-qh14:dieu-16" for ref in refs)
    assert any(ref.target_id == "article:45-2019-qh14:dieu-46" for ref in refs)


def test_details_and_guided_by_inverse_edge_creation() -> None:
    chunks = [
        chunk(article_number="169"),
        chunk(
            chunk_id="ND135_2020_Dieu_1",
            document_id="nghi-dinh-135-2020-nd-cp",
            document_type="nghi_dinh",
            article_number="1",
            text="Nghị định này quy định chi tiết Điều 169 của Bộ luật Lao động.",
        ),
    ]

    edges, duplicates = build_reference_edge_records(chunks)

    assert duplicates == 0
    assert any(edge["edge_type"] == "DETAILS" for edge in edges)
    assert any(edge["edge_type"] == "GUIDED_BY" for edge in edges)


def test_self_reference_prevention() -> None:
    edges, _ = build_reference_edge_records(
        [chunk(article_number="32", text="Theo khoản 1 Điều 32.", clause_ref="1", level="clause")]
    )

    assert edges == []


def test_retrieval_citation_prefix_is_not_parsed_as_reference() -> None:
    source = chunk(
        chunk_id="45_2019_QH14_Dieu_2_Khoan_1",
        article_number="2",
        clause_ref="1",
        level="clause",
        text="1. Người lao động.",
    )
    source["retrieval_text"] = (
        "Trong Chương I, Bộ luật Lao động 2019, Điều 2, khoản 1 quy định: 1. Người lao động."
    )

    edges, _ = build_reference_edge_records([source])

    assert edges == []


def test_duplicate_edge_prevention() -> None:
    chunks = [
        chunk(article_number="21", clause_ref="1", level="clause"),
        chunk(
            chunk_id="TT10_2020_Dieu_1",
            document_id="thong-tu-10-2020-tt-bldtbxh",
            document_type="thong_tu",
            article_number="1",
            text="Theo khoản 1 Điều 21 của Bộ luật Lao động. Theo khoản 1 Điều 21 của Bộ luật Lao động.",
        ),
    ]

    edges, duplicates = build_reference_edge_records(chunks)

    assert duplicates > 0
    assert sum(1 for edge in edges if edge["edge_type"] == "DETAILS") == 1


def test_unresolved_target_reporting() -> None:
    edges, duplicates = build_reference_edge_records(
        [chunk(text="Theo khoản 99 Điều 999 của Bộ luật Lao động.")]
    )
    summary = summarize_reference_edges(edges, duplicates)

    assert edges[0]["resolved"] is False
    assert summary["unresolved_edges"] == 1
    assert summary["unresolved_edge_ids"] == [edges[0]["edge_id"]]


def test_unresolved_report_classifies_external_reference() -> None:
    chunks = [
        chunk(
            article_number="219",
            text="Người lao động quy định tại điểm e khoản 1 Điều 2 của Luật này.",
        )
    ]
    edges, _ = build_reference_edge_records(chunks)
    report = build_unresolved_reference_report(edges, chunks)

    assert report["total_unresolved_edges"] == 1
    assert report["critical_unresolved_references"] == 0
    assert report["unresolved_by_reason"] == {"target document outside current corpus": 1}
    assert (
        report["unresolved_edges"][0]["recommended_action"]
        == "acceptable_external_reference"
    )


def test_unresolved_report_flags_wrong_targeted_critical_guiding_reference() -> None:
    report = build_unresolved_reference_report(
        [
            {
                "edge_id": "wrong",
                "source_id": "clause:thong-tu-09-2020-tt-bldtbxh:dieu-1:khoan-1",
                "target_id": "clause:thong-tu-09-2020-tt-bldtbxh:dieu-145:khoan-4",
                "edge_type": "REFERENCES",
                "source_chunk_id": "TT09_2020_Dieu_1_Khoan_1",
                "source_document_id": "thong-tu-09-2020-tt-bldtbxh",
                "target_document_id": "thong-tu-09-2020-tt-bldtbxh",
                "citation_text": "Thông tư 09, Điều 1, khoản 1",
                "original_matched_text": "khoản 4 Điều 145",
                "normalized_matched_text": "khoan 4 dieu 145",
                "extraction_method": "test",
                "confidence": 1.0,
                "resolved": True,
                "target_article": "145",
                "target_clause": "4",
                "target_point": None,
                "reference_level": "clause",
            }
        ],
        [],
    )

    assert report["critical_unresolved_references"] == 1
    assert report["critical_wrong_target_references"] == 1
