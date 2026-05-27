from __future__ import annotations

from src.vn_labor_law_ai_assistant.legal_chunk_enrichment import (
    enrich_legal_chunk,
    match_document_alias,
    summarize_enriched_chunks,
)


def make_chunk(**overrides: object) -> dict[str, object]:
    chunk: dict[str, object] = {
        "document_id": "45-2019-qh14",
        "document_title": "Bộ luật Lao động 2019",
        "document_type": "bo_luat",
        "chunk_id": "45_2019_QH14_Dieu_35",
        "section_id": "45-2019-qh14-dieu-35",
        "article_number": "35",
        "article_title": "Quyền đơn phương chấm dứt hợp đồng lao động của người lao động",
        "heading": "Điều 35. Quyền đơn phương chấm dứt hợp đồng lao động của người lao động",
        "part_number": None,
        "part_heading": None,
        "chapter_heading": "Chương III. HỢP ĐỒNG LAO ĐỘNG",
        "section_heading": "Mục 3. CHẤM DỨT HỢP ĐỒNG LAO ĐỘNG",
        "clause_ref": "1",
        "point_ref": None,
        "point_refs": [],
        "level": "clause",
        "text": "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động nhưng phải báo trước cho người sử dụng lao động.",
        "citation_text": "Bộ luật Lao động 2019, Điều 35, khoản 1",
        "retrieval_text": "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động.",
        "topic": [],
        "actor": [],
        "issue_type": [],
    }
    chunk.update(overrides)
    return chunk


def test_topic_detection_adds_required_labels() -> None:
    enriched = enrich_legal_chunk(make_chunk())

    assert "cham_dut_hop_dong_lao_dong" in enriched["topic"]
    assert "don_phuong_cham_dut" in enriched["topic"]
    assert "hop_dong_lao_dong" in enriched["topic"]


def test_actor_detection_adds_required_labels() -> None:
    enriched = enrich_legal_chunk(make_chunk())

    assert "nguoi_lao_dong" in enriched["actor"]
    assert "nguoi_su_dung_lao_dong" in enriched["actor"]


def test_issue_type_detection_adds_required_labels() -> None:
    enriched = enrich_legal_chunk(make_chunk())

    assert "don_phuong_cham_dut" in enriched["issue_type"]


def test_document_alias_matching() -> None:
    assert match_document_alias("Theo Bộ luật lao động năm 2019") == "45-2019-qh14"
    assert match_document_alias("Nghị định 145/2020 quy định chi tiết") == "nghi-dinh-145-2020-nd-cp"
    assert match_document_alias("BLTTDS về thẩm quyền tòa án") == "92-2015-qh13-labor-only"


def test_normative_rank_assignment() -> None:
    law = enrich_legal_chunk(make_chunk(document_type="bo_luat"))
    decree = enrich_legal_chunk(
        make_chunk(document_id="nghi-dinh-145-2020-nd-cp", document_type="nghi_dinh")
    )
    circular = enrich_legal_chunk(
        make_chunk(document_id="thong-tu-10-2020-tt-bldtbxh", document_type="thong_tu")
    )

    assert law["issuing_authority"] == "Quốc hội"
    assert law["normative_rank"] == 1
    assert law["rank_label"] == "highest"
    assert decree["issuing_authority"] == "Chính phủ"
    assert decree["normative_rank"] == 2
    assert decree["rank_label"] == "middle"
    assert circular["issuing_authority"] == "Bộ trưởng / Thủ trưởng cơ quan ngang Bộ"
    assert circular["normative_rank"] == 3
    assert circular["rank_label"] == "lowest"


def test_enriched_output_schema_preserves_existing_fields_and_adds_summary_fields() -> None:
    original = make_chunk(topic=["existing_topic"])
    enriched = enrich_legal_chunk(original)
    summary = summarize_enriched_chunks([enriched])

    for key, value in original.items():
        if key not in {"topic", "actor", "issue_type", "document_type"}:
            assert enriched[key] == value

    assert "existing_topic" in enriched["topic"]
    assert enriched["effective_level"] == "national"
    assert enriched["status"] == "in_effect"
    assert enriched["document_hierarchy"]["article_number"] == "35"
    assert enriched["document_hierarchy"]["clause_ref"] == "1"
    assert summary["chunk_count"] == 1
    assert summary["chunks_missing_topic"] == []
    assert summary["chunks_missing_actor"] == []
    assert summary["chunks_missing_issue_type"] == []
    assert summary["document_type_distribution"] == {"bo_luat": 1}
    assert summary["normative_rank_distribution"] == {"1": 1}
