from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.vn_labor_law_ai_assistant.corpus_pipeline import (
    ARTICLE_RE,
    CURATED_LEGAL_CHUNK_FILENAMES,
    FULL_BLTTDS_FILENAME,
    PageRecord,
    build_curated_chunk_records,
    build_corpus,
    chunk_sections,
    enrich_chunk,
    infer_document_title,
    infer_chunk_taxonomy,
    normalize_extracted_text,
    resolve_curated_legal_chunk_paths,
    slugify_text,
    split_main_text_and_appendix_text,
    split_sections,
    summarize_legal_chunks,
)


class CorpusPipelineTests(unittest.TestCase):
    def test_slugify_text_handles_vietnamese(self) -> None:
        self.assertEqual(slugify_text("Nghị định 145/2020/NĐ-CP"), "nghi-dinh-145-2020-nd-cp")

    def test_normalize_extracted_text_applies_spacing_fixes(self) -> None:
        raw = "Bộluật Lao động từ12 tháng ; Trợcấp sửdụng"
        cleaned = normalize_extracted_text(raw)
        self.assertIn("Bộ luật Lao động từ 12 tháng;", cleaned)
        self.assertIn("Trợ cấp sử dụng", cleaned)

    def test_split_sections_extracts_article_context(self) -> None:
        page_records = [
            PageRecord(
                page_number=1,
                text=(
                    "Mục 2. CHẤM DỨT HỢP ĐỒNG LAO ĐỘNG\n\n"
                    "Điều 7. Thời hạn báo trước khi đơn phương chấm dứt hợp đồng lao động\n\n"
                    "1. Người lao động phải báo trước."
                ),
            ),
            PageRecord(
                page_number=2,
                text=(
                    "Điều 8. Trợ cấp thôi việc, trợ cấp mất việc làm\n\n"
                    "1. Người sử dụng lao động có trách nhiệm chi trả."
                ),
            ),
        ]

        sections = split_sections(
            page_records=page_records,
            document_id="nghi-dinh-145-2020-nd-cp",
            document_title="Nghị định 145/2020/NĐ-CP",
        )

        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].article_number, "7")
        self.assertEqual(sections[0].section_heading, "Mục 2. CHẤM DỨT HỢP ĐỒNG LAO ĐỘNG")
        self.assertEqual(sections[1].article_number, "8")
        self.assertEqual(sections[1].source_pages, [2])

    def test_split_sections_uses_group_heading_for_curated_text(self) -> None:
        page_records = [
            PageRecord(
                page_number=1,
                text=(
                    "II. NHÓM ĐIỀU CỐT LÕI VỀ CHẤM DỨT HỢP ĐỒNG LAO ĐỘNG\n\n"
                    "Điều 34. Các trường hợp chấm dứt hợp đồng lao động\n\n"
                    "1. Hết hạn hợp đồng lao động."
                ),
            )
        ]

        sections = split_sections(
            page_records=page_records,
            document_id="bo-luat-45-2019-qh14",
            document_title="Bộ luật số 45/2019/QH14",
        )

        self.assertEqual(sections[0].section_heading, "II. NHÓM ĐIỀU CỐT LÕI VỀ CHẤM DỨT HỢP ĐỒNG LAO ĐỘNG")

    def test_split_sections_handles_heading_and_body_in_same_paragraph(self) -> None:
        page_records = [
            PageRecord(
                page_number=1,
                text="Điều 35. Quyền đơn phương chấm dứt hợp đồng lao động của người lao động\n1. Người lao động có quyền đơn phương chấm dứt hợp đồng lao động.",
            )
        ]

        sections = split_sections(
            page_records=page_records,
            document_id="bo-luat-45-2019-qh14",
            document_title="Bộ luật số 45/2019/QH14",
        )

        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].article_number, "35")
        self.assertIn("1. Người lao động có quyền", sections[0].text)

    def test_split_sections_captures_embedded_chapter_heading_transitions(self) -> None:
        page_records = [
            PageRecord(
                page_number=1,
                text=(
                    "III. QUY ĐỊNH BỔ TRỢ KHÁC LIÊN QUAN\n"
                    "Điều 62. Hợp đồng đào tạo nghề\n"
                    "1. Nội dung đào tạo.\n"
                    "Chương V\n"
                    "ĐỐI THOẠI TẠI NƠI LÀM VIỆC\n"
                    "Mục 1. ĐỐI THOẠI TẠI NƠI LÀM VIỆC\n"
                    "Điều 63. Tổ chức đối thoại tại nơi làm việc\n"
                    "1. Tổ chức đối thoại."
                ),
            )
        ]

        sections = split_sections(
            page_records=page_records,
            document_id="bo-luat-45-2019-qh14",
            document_title="Bộ luật số 45/2019/QH14",
        )

        dieu_62 = next(section for section in sections if section.article_number == "62")
        dieu_63 = next(section for section in sections if section.article_number == "63")

        self.assertEqual(dieu_62.section_heading, "III. QUY ĐỊNH BỔ TRỢ KHÁC LIÊN QUAN")
        self.assertEqual(dieu_63.chapter_heading, "Chương V. ĐỐI THOẠI TẠI NƠI LÀM VIỆC")
        self.assertEqual(dieu_63.section_heading, "Mục 1. ĐỐI THOẠI TẠI NƠI LÀM VIỆC")

    def test_infer_document_title_uses_source_hint(self) -> None:
        text = (
            "BỘ DỮ LIỆU TRÍCH XUẤT TỪ BỘ LUẬT LAO ĐỘNG 2019\n"
            "Nguồn: Bộ luật số 45/2019/QH14\n"
        )
        self.assertEqual(infer_document_title(text, "fallback"), "Bộ luật số 45/2019/QH14")

    def test_infer_document_title_uses_first_meaningful_line(self) -> None:
        text = "Nghị định 145/2020/NĐ-CP\n\nMục 2. CHẤM DỨT HỢP ĐỒNG LAO ĐỘNG\n"
        self.assertEqual(infer_document_title(text, "fallback"), "Nghị định 145/2020/NĐ-CP")

    def test_infer_document_title_prefers_legal_identifier_from_fallback_name(self) -> None:
        text = (
            "CHAPTER I\n\n"
            "GENERAL PROVISIONS\n\n"
            "Article 1. Scope\n"
        )

        self.assertEqual(
            infer_document_title(text, "45_2019_QH14"),
            "Bộ luật số 45/2019/QH14",
        )

    def test_chunk_sections_splits_long_text(self) -> None:
        long_body = " ".join(["trợ cấp thôi việc"] * 200)
        section = split_sections(
            page_records=[
                PageRecord(
                    page_number=1,
                    text=f"Điều 8. Trợ cấp thôi việc\n\n{long_body}",
                )
            ],
            document_id="test-doc",
            document_title="Test Document",
        )[0]

        chunks = chunk_sections([section], max_chars=300)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(chunk["char_count"] <= 300 for chunk in chunks))

    def test_chunk_sections_preserves_article_full_behavior_with_stable_chunk_id(self) -> None:
        section = split_sections(
            page_records=[
                PageRecord(
                    page_number=1,
                    text=(
                        "Điều 4. Tuổi nghỉ hưu trong điều kiện lao động bình thường\n\n"
                        "Tuổi nghỉ hưu trong điều kiện lao động bình thường được điều chỉnh theo lộ trình."
                    ),
                )
            ],
            document_id="nghi-dinh-135-2020-nd-cp",
            document_title="Nghị định 135/2020/NĐ-CP",
        )[0]

        chunks = chunk_sections([section], max_chars=1200)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_type"], "article_full")
        self.assertEqual(chunks[0]["chunk_id"], "ND135_2020_Dieu_4")

    def test_chunk_sections_preserves_article_intro_behavior(self) -> None:
        section = split_sections(
            page_records=[
                PageRecord(
                    page_number=1,
                    text=(
                        "Điều 3. Nội dung chủ yếu của hợp đồng lao động\n"
                        "Hợp đồng lao động phải có các nội dung chủ yếu sau đây:\n"
                        "1. Tên, địa chỉ của người sử dụng lao động."
                    ),
                )
            ],
            document_id="thong-tu-10-2020-tt-bldtbxh",
            document_title="Thông tư 10/2020/TT-BLĐTBXH",
        )[0]

        chunks = chunk_sections([section], max_chars=1200)

        self.assertEqual(chunks[0]["chunk_type"], "article_intro")
        self.assertEqual(chunks[0]["chunk_id"], "TT10_2020_Dieu_3_Intro")
        self.assertEqual(chunks[1]["chunk_type"], "clause")
        self.assertEqual(chunks[1]["chunk_id"], "TT10_2020_Dieu_3_Khoan_1")

    def test_chunk_sections_prefers_legal_boundaries_before_whitespace_fallback(self) -> None:
        page_records = [
            PageRecord(
                page_number=1,
                text=(
                    "Điều 36. Quyền đơn phương chấm dứt hợp đồng lao động của người sử dụng lao động\n"
                    "1. Người sử dụng lao động có quyền đơn phương chấm dứt hợp đồng lao động trong trường hợp sau đây:\n"
                    "a) Người lao động thường xuyên không hoàn thành công việc theo hợp đồng lao động;\n"
                    "b) Người lao động bị ốm đau, tai nạn đã điều trị trong thời gian dài mà khả năng lao động chưa hồi phục;\n"
                    "c) Do thiên tai, hỏa hoạn, dịch bệnh nguy hiểm, địch họa hoặc di dời, thu hẹp sản xuất, kinh doanh theo yêu cầu của cơ quan nhà nước có thẩm quyền mà người sử dụng lao động đã tìm mọi biện pháp khắc phục nhưng vẫn buộc phải giảm chỗ làm việc;\n"
                    "d) Người lao động không có mặt tại nơi làm việc sau thời hạn quy định."
                ),
            )
        ]

        section = split_sections(
            page_records=page_records,
            document_id="bo-luat-45-2019-qh14",
            document_title="Bộ luật số 45/2019/QH14",
        )[0]

        chunks = chunk_sections([section], max_chars=420)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(any("địch họa hoặc di dời" in chunk["text"] for chunk in chunks))
        self.assertTrue(all("\n\nịch họa" not in chunk["text"] for chunk in chunks))

    def test_chunk_sections_groups_clause_intro_with_points(self) -> None:
        section = split_sections(
            page_records=[
                PageRecord(
                    page_number=1,
                    text=(
                        "Điều 107. Làm thêm giờ\n"
                        "2. Người sử dụng lao động được sử dụng người lao động làm thêm giờ khi đáp ứng đầy đủ các yêu cầu sau đây:\n"
                        "a) Phải được sự đồng ý của người lao động;\n"
                        "b) Bảo đảm số giờ làm thêm của người lao động không quá 50% số giờ làm việc bình thường trong 01 ngày;\n"
                        "c) Bảo đảm số giờ làm thêm của người lao động không quá 200 giờ trong 01 năm."
                    ),
                )
            ],
            document_id="45-2019-qh14",
            document_title="Bộ luật số 45/2019/QH14",
        )[0]

        chunks = chunk_sections([section], max_chars=1200)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["clause_ref"], "2")
        self.assertEqual(chunks[0]["point_refs"], ["a", "b", "c"])
        self.assertEqual(chunks[0]["chunk_id"], "45_2019_QH14_Dieu_107_Khoan_2_Diem_a_b_c")
        self.assertIn("khi đáp ứng đầy đủ các yêu cầu", chunks[0]["text"])
        self.assertIn("a) Phải được sự đồng ý", chunks[0]["text"])
        self.assertIn("c) Bảo đảm số giờ làm thêm", chunks[0]["text"])

    def test_enrich_chunk_cites_multiple_point_refs(self) -> None:
        section = split_sections(
            page_records=[
                PageRecord(
                    page_number=1,
                    text=(
                        "Điều 3. Nội dung chủ yếu của hợp đồng lao động\n"
                        "5. Nội dung về công việc gồm:\n"
                        "a) Tên công việc;\n"
                        "b) Địa điểm làm việc."
                    ),
                )
            ],
            document_id="thong-tu-09-2020-tt-bldtbxh",
            document_title="Thông tư 09/2020/TT-BLĐTBXH",
        )[0]

        chunks = chunk_sections([section], max_chars=1200)
        enriched = enrich_chunk(
            chunks[0],
            document_title="Thông tư 09/2020/TT-BLĐTBXH",
            source_kind="curated_text",
        )

        self.assertEqual(chunks[0]["point_refs"], ["a", "b"])
        self.assertEqual(chunks[0]["chunk_id"], "TT09_2020_Dieu_3_Khoan_5_Diem_a_b")
        self.assertIn("các điểm a, b", enriched["citation_text"])

    def test_human_readable_chunk_id_distinguishes_d_and_dd_points(self) -> None:
        section = split_sections(
            page_records=[
                PageRecord(
                    page_number=1,
                    text=(
                        "Điều 5. Nhóm điểm kiểm tra\n"
                        "1. Các nội dung gồm:\n"
                        "d) Điểm d;\n"
                        "đ) Điểm đ."
                    ),
                )
            ],
            document_id="45-2019-qh14",
            document_title="Bộ luật số 45/2019/QH14",
        )[0]

        chunks = chunk_sections([section], max_chars=1200)

        self.assertEqual(chunks[0]["point_refs"], ["d", "đ"])
        self.assertEqual(chunks[0]["chunk_id"], "45_2019_QH14_Dieu_5_Khoan_1_Diem_d_dd")

    def test_chunk_sections_creates_clause_part_chunks_for_large_point_sets(self) -> None:
        long_a = " ".join(["Tên công việc và địa điểm làm việc"] * 20)
        long_b = " ".join(["Thời hạn của hợp đồng lao động"] * 20)
        section = split_sections(
            page_records=[
                PageRecord(
                    page_number=1,
                    text=(
                        "Điều 3. Nội dung chủ yếu của hợp đồng lao động\n"
                        "5. Nội dung về công việc gồm:\n"
                        f"a) {long_a};\n"
                        f"b) {long_b}."
                    ),
                )
            ],
            document_id="thong-tu-10-2020-tt-bldtbxh",
            document_title="Thông tư 10/2020/TT-BLĐTBXH",
        )[0]

        chunks = chunk_sections([section], max_chars=300)

        self.assertTrue(any(chunk["chunk_type"] == "clause_part" for chunk in chunks))
        self.assertTrue(
            any(
                str(chunk["chunk_id"]).startswith("TT10_2020_Dieu_3_Khoan_5_Diem_")
                for chunk in chunks
            )
        )

    def test_chunk_sections_uses_sequential_chunks_for_complex_amendment_article(self) -> None:
        section = split_sections(
            page_records=[
                PageRecord(
                    page_number=1,
                    text=(
                        "Điều 219. Sửa đổi, bổ sung một số điều của các luật có liên quan đến lao động\n"
                        "1. Sửa đổi, bổ sung một số điều của Luật Bảo hiểm xã hội:\n"
                        "a) Sửa đổi, bổ sung Điều 54 như sau:\n"
                        "“Điều 54. Điều kiện hưởng lương hưu\n"
                        "1. Người lao động khi nghỉ việc có đủ 20 năm đóng bảo hiểm xã hội trở lên thì được hưởng lương hưu.\n"
                        "a) Đủ tuổi theo quy định tại khoản 2 Điều 169 của Bộ luật Lao động;”"
                    ),
                )
            ],
            document_id="45-2019-qh14",
            document_title="Bộ luật số 45/2019/QH14",
        )[0]

        chunks = chunk_sections([section], max_chars=400)

        self.assertTrue(chunks)
        self.assertTrue(all(chunk["chunk_type"] == "article_sequential" for chunk in chunks))
        self.assertTrue(all(chunk["clause_ref"] is None for chunk in chunks))
        self.assertTrue(all(chunk["point_refs"] == [] for chunk in chunks))

    def test_chunk_sections_preserves_clause_and_point_hierarchy(self) -> None:
        page_records = [
            PageRecord(
                page_number=1,
                text=(
                    "Điều 8. Trợ cấp thôi việc, trợ cấp mất việc làm\n"
                    "2. Người sử dụng lao động có trách nhiệm chi trả trợ cấp mất việc làm.\n"
                    "Trường hợp người lao động có thời gian làm việc thực tế đủ 12 tháng thì vẫn được hưởng.\n"
                    "4. Xác định thời gian người lao động đã làm việc thực tế trong một số trường hợp đặc biệt:\n"
                    "c) Trường hợp người lao động tiếp tục làm việc theo phương án sử dụng lao động.\n"
                    "c.2) Trường hợp hợp đồng lao động chấm dứt theo khoản 11 Điều 34 thì được tính trợ cấp mất việc làm.\n"
                    "c.3) Người sử dụng lao động có trách nhiệm chi trả trợ cấp thôi việc cho thời gian làm việc trước đó."
                ),
            )
        ]

        section = split_sections(
            page_records=page_records,
            document_id="nghi-dinh-145-2020-nd-cp",
            document_title="Nghị định 145/2020/NĐ-CP",
        )[0]

        chunks = chunk_sections([section], max_chars=500)
        clause_chunk = next(chunk for chunk in chunks if chunk["clause_ref"] == "2")
        clause_with_points = next(chunk for chunk in chunks if chunk["clause_ref"] == "4")

        self.assertEqual(clause_chunk["clause_ref"], "2")
        self.assertIsNone(clause_chunk["point_ref"])
        self.assertEqual(clause_with_points["point_refs"], ["c", "c.2", "c.3"])
        self.assertIsNone(clause_with_points["point_ref"])
        self.assertIn("4. Xác định thời gian", clause_with_points["text"])
        self.assertIn("c.2) Trường hợp hợp đồng lao động", clause_with_points["text"])
        self.assertIn("c.3) Người sử dụng lao động", clause_with_points["text"])

    def test_split_sections_tracks_part_heading(self) -> None:
        page_records = [
            PageRecord(
                page_number=1,
                text=(
                    "PHẦN THỨ SÁU\n"
                    "THỦ TỤC GIẢI QUYẾT VIỆC DÂN SỰ\n"
                    "Chương XXX\n"
                    "THỦ TỤC GIẢI QUYẾT YÊU CẦU TUYÊN BỐ HỢP ĐỒNG LAO ĐỘNG VÔ HIỆU\n"
                    "Điều 401. Yêu cầu tuyên bố hợp đồng lao động vô hiệu\n"
                    "1. Người lao động có quyền yêu cầu Tòa án."
                ),
            )
        ]

        section = split_sections(
            page_records=page_records,
            document_id="92-2015-qh13-labor-only",
            document_title="Bộ luật Tố tụng dân sự 2015",
        )[0]
        chunks = chunk_sections([section], max_chars=1200)

        self.assertEqual(section.part_number, "THỨ SÁU")
        self.assertEqual(
            section.part_heading,
            "PHẦN THỨ SÁU. THỦ TỤC GIẢI QUYẾT VIỆC DÂN SỰ",
        )
        self.assertEqual(chunks[0]["part_number"], "THỨ SÁU")
        self.assertEqual(chunks[0]["chunk_id"], "BLTTDS_2015_Dieu_401_Khoan_1")

    def test_chunk_sections_assigns_nearest_parent_chunk_id(self) -> None:
        page_records = [
            PageRecord(
                page_number=1,
                text=(
                    "Điều 7. Thời hạn báo trước\n"
                    "Nội dung mở đầu điều luật.\n"
                    "1. Ngành, nghề đặc thù gồm:\n"
                    "a) Thành viên tổ lái tàu bay;\n"
                    "2. Thời hạn báo trước như sau:\n"
                    "c) Trường hợp tổng quát.\n"
                    "c.2) Trường hợp cụ thể."
                ),
            )
        ]

        section = split_sections(
            page_records=page_records,
            document_id="nghi-dinh-145-2020-nd-cp",
            document_title="Nghị định 145/2020/NĐ-CP",
        )[0]

        chunks = chunk_sections([section], max_chars=500)
        article_chunk = next(chunk for chunk in chunks if chunk["chunk_type"] == "article_intro")
        clause_one_chunk = next(chunk for chunk in chunks if chunk["clause_ref"] == "1" and chunk["point_ref"] is None)
        clause_two_chunk = next(chunk for chunk in chunks if chunk["clause_ref"] == "2" and chunk["point_ref"] is None)

        self.assertEqual(clause_one_chunk["parent_chunk_id"], article_chunk["chunk_id"])
        self.assertEqual(clause_two_chunk["parent_chunk_id"], article_chunk["chunk_id"])
        self.assertEqual(clause_one_chunk["point_refs"], ["a"])
        self.assertEqual(clause_two_chunk["point_refs"], ["c", "c.2"])
        self.assertIn("Nội dung mở đầu điều luật.", clause_one_chunk["text"])
        self.assertIn("Nội dung mở đầu điều luật.", clause_two_chunk["text"])

    def test_chunk_sections_whitespace_fallback_does_not_split_words(self) -> None:
        body = " ".join(["alpha", "beta", "gamma"] * 80)
        section = split_sections(
            page_records=[
                PageRecord(
                    page_number=1,
                    text=f"Điều 99. Kiểm tra fallback\n{body}",
                )
            ],
            document_id="test-doc",
            document_title="Test Document",
        )[0]

        chunks = chunk_sections([section], max_chars=180)

        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            body_text = chunk["text"].split("\n\n", 1)[1]
            tokens = body_text.replace("\n", " ").split()
            self.assertTrue(all(token in {"alpha", "beta", "gamma"} for token in tokens))

    def test_chunk_sections_skips_preamble_sections(self) -> None:
        page_records = [
            PageRecord(
                page_number=1,
                text=(
                    "BỘ DỮ LIỆU TRÍCH XUẤT TỪ BỘ LUẬT LAO ĐỘNG 2019\n\n"
                    "Điều 34. Các trường hợp chấm dứt hợp đồng lao động\n\n"
                    "1. Hết hạn hợp đồng lao động."
                ),
            )
        ]

        sections = split_sections(
            page_records=page_records,
            document_id="bo-luat-45-2019-qh14",
            document_title="Bộ luật số 45/2019/QH14",
        )
        chunks = chunk_sections(sections)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["article_number"], "34")
        self.assertFalse(any("preamble" in chunk["section_id"] for chunk in chunks))

    def test_enrich_chunk_adds_rag_metadata(self) -> None:
        page_records = [
            PageRecord(
                page_number=1,
                text=(
                    "Điều 36. Quyền đơn phương chấm dứt hợp đồng lao động của người sử dụng lao động\n"
                    "1. Người sử dụng lao động có quyền đơn phương chấm dứt hợp đồng lao động.\n"
                    "2. Người sử dụng lao động phải báo trước cho người lao động."
                ),
            )
        ]

        section = split_sections(
            page_records=page_records,
            document_id="bo-luat-45-2019-qh14",
            document_title="Bộ luật số 45/2019/QH14",
        )[0]
        chunk = chunk_sections([section])[0]
        enriched = enrich_chunk(
            chunk=chunk,
            document_title="Bộ luật số 45/2019/QH14",
            source_kind="curated_text",
        )

        self.assertNotIn("source_pages", enriched)
        self.assertEqual(enriched["clause_ref"], "1")
        self.assertIsNone(enriched["point_ref"])
        self.assertEqual(enriched["level"], "clause")
        self.assertIsNone(enriched["parent_chunk_id"])
        self.assertIn("Điều 36", enriched["citation_text"])
        self.assertIn("khoản 1", enriched["citation_text"])
        self.assertIn("quy định:", enriched["retrieval_text"])
        self.assertNotIn("Văn bản:", enriched["retrieval_text"])
        self.assertIn("don_phuong_cham_dut", enriched["topic"])
        self.assertIn("nguoi_su_dung_lao_dong", enriched["actor"])
        self.assertIn("quyen_don_phuong_cham_dut", enriched["issue_type"])

    def test_enrich_chunk_builds_specific_citation_for_point_chunks(self) -> None:
        chunk = {
            "chunk_id": "nghi-dinh-145-2020-nd-cp-dieu-8-chunk-08",
            "section_id": "nghi-dinh-145-2020-nd-cp-dieu-8",
            "article_number": "8",
            "article_title": "Trợ cấp thôi việc, trợ cấp mất việc làm",
            "heading": "Điều 8. Trợ cấp thôi việc, trợ cấp mất việc làm",
            "chapter_heading": None,
            "section_heading": "Mục 2. CHẤM DỨT HỢP ĐỒNG LAO ĐỘNG",
            "source_pages": [3],
            "chunk_index": 8,
            "char_count": 160,
            "text": (
                "Điều 8. Trợ cấp thôi việc, trợ cấp mất việc làm\n\n"
                "c.2) Trường hợp hợp đồng lao động chấm dứt theo khoản 11 Điều 34 thì được tính trợ cấp mất việc làm."
            ),
            "clause_ref": "4",
            "point_ref": "c.2",
        }

        enriched = enrich_chunk(
            chunk=chunk,
            document_title="Nghị định 145/2020/NĐ-CP",
            source_kind="raw_pdf",
        )

        self.assertEqual(enriched["level"], "point")
        self.assertIn("khoản 4", enriched["citation_text"])
        self.assertIn("điểm c.2", enriched["citation_text"])
        self.assertEqual(enriched["source_pages"], [3])

    def test_infer_chunk_taxonomy_does_not_tag_maternity_benefit_as_tro_cap(self) -> None:
        topics, actors, issue_types = infer_chunk_taxonomy(
            document_title="Bộ luật số 45/2019/QH14",
            section_heading="Mục 2. LAO ĐỘNG NỮ VÀ BẢO ĐẢM BÌNH ĐẲNG GIỚI",
            article_title="Nghỉ thai sản",
            body_text="Trong thời gian nghỉ thai sản, lao động nữ được hưởng chế độ thai sản theo quy định của pháp luật về bảo hiểm xã hội.",
        )

        self.assertIn("bao_ve_thai_san", topics)
        self.assertIn("bao_ve_thai_san", issue_types)
        self.assertNotIn("tro_cap", topics)
        self.assertIn("lao_dong_nu", actors)

    def test_build_corpus_curated_text_overrides_same_document_id_from_raw_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw"
            cleaned_dir = root / "cleaned"
            chunks_dir = root / "chunks"
            metadata_dir = root / "metadata"
            raw_dir.mkdir()
            cleaned_dir.mkdir()

            (raw_dir / "Nghị định 145／2020／NĐ-CP.pdf").write_bytes(b"%PDF-1.4\n")
            curated_path = cleaned_dir / "nghi-dinh-145-2020-nd-cp.txt"
            curated_path.write_text(
                "Nghị định 145/2020/NĐ-CP\n\nĐiều 7. Thời hạn báo trước\n\n1. Nội dung mẫu.",
                encoding="utf-8",
            )

            manifest = build_corpus(
                raw_dir=raw_dir,
                cleaned_dir=cleaned_dir,
                chunks_dir=chunks_dir,
                metadata_dir=metadata_dir,
                curated_text_paths=[curated_path],
            )

            self.assertEqual(manifest["document_count"], 1)
            self.assertEqual(manifest["documents"][0]["source_kind"], "curated_text")
            chunk_payload = json.loads((chunks_dir / "nghi-dinh-145-2020-nd-cp.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(chunk_payload["source_kind"], "curated_text")
            self.assertEqual(chunk_payload["document_title"], "Nghị định 145/2020/NĐ-CP")
            self.assertNotIn("source_pages", chunk_payload)


    def test_build_corpus_prefers_full_curated_text_over_partial_duplicate_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw"
            cleaned_dir = root / "cleaned"
            chunks_dir = root / "chunks"
            metadata_dir = root / "metadata"
            raw_dir.mkdir()
            cleaned_dir.mkdir()

            article_prefix = ARTICLE_RE.pattern.split(r"\s+", 1)[0].lstrip("^")
            full_law_path = cleaned_dir / "45_2019_QH14.txt"
            full_law_path.write_text(
                (
                    "Äiá»u 1. Pháº¡m vi Ä‘iá»u chá»‰nh\n\n"
                    "1. Ná»™i dung toÃ n vÄƒn.\n\n"
                    "Äiá»u 34. CÃ¡c trÆ°á»ng há»£p cháº¥m dá»©t há»£p Ä‘á»“ng lao Ä‘á»™ng\n\n"
                    "1. Háº¿t háº¡n há»£p Ä‘á»“ng lao Ä‘á»™ng."
                ),
                encoding="utf-8",
            )
            full_law_path.write_text(
                (
                    f"{article_prefix} 1. Scope\n\n"
                    "1. Full curated source body.\n\n"
                    f"{article_prefix} 34. Termination\n\n"
                    "1. End of contract."
                ),
                encoding="utf-8",
            )
            partial_extract_path = cleaned_dir / "45_2019_QH14_extract.txt"
            partial_extract_path.write_text(
                (
                    "BO DU LIEU TRICH XUAT TU BO LUAT LAO DONG 2019\n"
                    "Nguá»“n: Bá»™ luáº­t sá»‘ 45/2019/QH14\n"
                    "Pham vi trich xuat: cham dut hop dong lao dong\n\n"
                    "Partial extract content."
                ),
                encoding="utf-8",
            )
            partial_extract_path.write_text(
                (
                    "BO DU LIEU TRICH XUAT TU BO LUAT LAO DONG 2019\n"
                    "Pham vi trich xuat: cham dut hop dong lao dong\n\n"
                    "Partial extract content."
                ),
                encoding="utf-8",
            )

            manifest = build_corpus(
                raw_dir=raw_dir,
                cleaned_dir=cleaned_dir,
                chunks_dir=chunks_dir,
                metadata_dir=metadata_dir,
                curated_text_paths=[full_law_path, partial_extract_path],
            )

            self.assertEqual(manifest["document_count"], 1)
            self.assertEqual(manifest["documents"][0]["source_path"], str(full_law_path.as_posix()))
            self.assertTrue(manifest["warnings"])
            self.assertIn("skipped", manifest["warnings"][0].lower())
            chunk_payload = json.loads(
                Path(manifest["documents"][0]["chunks_path"]).read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(chunk_payload["document_title"], "Bộ luật số 45/2019/QH14")

    def test_curated_chunk_build_uses_blttds_labor_only_and_excludes_full_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for filename in CURATED_LEGAL_CHUNK_FILENAMES:
                (root / filename).write_text(
                    "Điều 1. Điều khoản kiểm tra\n\n1. Nội dung kiểm tra.",
                    encoding="utf-8",
                )
            full_blttds_path = root / FULL_BLTTDS_FILENAME
            full_blttds_path.write_text(
                "Điều 1. Phạm vi điều chỉnh\n\n1. Nội dung toàn văn.",
                encoding="utf-8",
            )

            paths = resolve_curated_legal_chunk_paths(root)
            chunks, warnings = build_curated_chunk_records([*paths, full_blttds_path])
            source_names = {Path(str(chunk["source_path"])).name for chunk in chunks}

            self.assertIn("92_2015_QH13_labor_only.txt", [path.name for path in paths])
            self.assertTrue(warnings)
            self.assertIn("92_2015_QH13_labor_only.txt", source_names)
            self.assertNotIn(FULL_BLTTDS_FILENAME, source_names)

    def test_curated_chunk_build_uses_canonical_document_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            text_path = root / "thong_tu_09_2020_tt_bldtbxh_clean.txt"
            text_path.write_text(
                (
                    "BỘ LAO ĐỘNG - THƯƠNG BINH VÀ XÃ HỘI\n\n"
                    "Điều 1. Phạm vi điều chỉnh\n"
                    "1. Nội dung kiểm tra."
                ),
                encoding="utf-8",
            )

            chunks, _ = build_curated_chunk_records([text_path])

            self.assertEqual(chunks[0]["document_id"], "thong-tu-09-2020-tt-bldtbxh")
            self.assertEqual(chunks[0]["document_title"], "Thông tư 09/2020/TT-BLĐTBXH")
            self.assertEqual(chunks[0]["document_type"], "thong_tu")
            self.assertNotIn("-clean", chunks[0]["document_id"])
            self.assertIn("Thông tư 09/2020/TT-BLĐTBXH", chunks[0]["citation_text"])

    def test_appendix_boundary_detection_splits_footer_and_appendix(self) -> None:
        main_text, appendix_text = split_main_text_and_appendix_text(
            (
                "Điều 9. Trách nhiệm hướng dẫn thi hành\n"
                "Các Bộ trưởng chịu trách nhiệm thi hành Nghị định này.\n"
                "TM. CHÍNH PHỦ\n"
                "THỦ TƯỚNG\n"
                "Nơi nhận:\n"
                "- Lưu: VT.\n"
                "PHỤ LỤC I\n"
                "LỘ TRÌNH TUỔI NGHỈ HƯU\n"
                "Lao động nam\n"
                "Tháng sinh | Năm sinh\n"
            )
        )

        self.assertIn("Điều 9. Trách nhiệm hướng dẫn thi hành", main_text)
        self.assertNotIn("TM. CHÍNH PHỦ", main_text)
        self.assertNotIn("Nơi nhận", main_text)
        self.assertIn("PHỤ LỤC I", appendix_text)

    def test_nd135_appendices_are_not_attached_to_article_9(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            text_path = Path(tmpdir) / "nghi_dinh_135_2020_nd_cp_clean.txt"
            text_path.write_text(
                (
                    "Điều 9. Trách nhiệm hướng dẫn thi hành\n"
                    "Các Bộ trưởng chịu trách nhiệm thi hành Nghị định này.\n"
                    "TM. CHÍNH PHỦ\n"
                    "THỦ TƯỚNG\n"
                    "Nơi nhận:\n"
                    "- Lưu: VT.\n"
                    "PHỤ LỤC I\n"
                    "LỘ TRÌNH TUỔI NGHỈ HƯU TRONG ĐIỀU KIỆN LAO ĐỘNG BÌNH THƯỜNG\n"
                    "Lao động nam\n"
                    "Tháng sinh | Năm sinh | Tuổi nghỉ hưu\n"
                    "1 | 1961 | 60 tuổi 3 tháng\n"
                    "Lao động nữ\n"
                    "Tháng sinh | Năm sinh | Tuổi nghỉ hưu\n"
                    "1 | 1966 | 55 tuổi 4 tháng\n"
                    "PHỤ LỤC III\n"
                    "DANH MỤC CÔNG VIỆC KHAI THÁC THAN TRONG HẦM LÒ\n"
                    "1. Khai thác mỏ hầm lò.\n"
                ),
                encoding="utf-8",
            )

            chunks, _ = build_curated_chunk_records([text_path])
            article_9 = next(chunk for chunk in chunks if chunk.get("article_number") == "9")
            appendix_chunks = [chunk for chunk in chunks if chunk.get("level") == "appendix"]

            self.assertNotIn("PHỤ LỤC", article_9["text"])
            self.assertNotIn("TM. CHÍNH PHỦ", article_9["text"])
            self.assertTrue(appendix_chunks)
            self.assertTrue(
                any(
                    chunk["chunk_id"].startswith("ND135_2020_Phu_Luc_I_Bang_Nam")
                    for chunk in appendix_chunks
                )
            )
            self.assertTrue(all(chunk.get("appendix_id") for chunk in appendix_chunks))

    def test_nd145_forms_are_not_attached_to_article_115(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            text_path = Path(tmpdir) / "nghi_dinh_145_2020_nd_cp_clean.txt"
            text_path.write_text(
                (
                    "Điều 115. Trách nhiệm thi hành\n"
                    "Các Bộ trưởng, Thủ trưởng cơ quan ngang bộ chịu trách nhiệm thi hành Nghị định này.\n"
                    "Nơi nhận: CHÍNH PHỦ\n"
                    "- Lưu: VT.\n"
                    "Mẫu số 01/PLI | Báo cáo tình hình sử dụng lao động\n"
                    "TÊN DOANH NGHIỆP\n"
                    "Nội dung mẫu báo cáo.\n"
                ),
                encoding="utf-8",
            )

            chunks, _ = build_curated_chunk_records([text_path])
            article_115 = next(chunk for chunk in chunks if chunk.get("article_number") == "115")
            appendix_chunk = next(chunk for chunk in chunks if chunk.get("level") == "appendix")

            self.assertNotIn("Mẫu số", article_115["text"])
            self.assertNotIn("Nơi nhận", article_115["text"])
            self.assertEqual(appendix_chunk["chunk_id"], "ND145_2020_Phu_Luc_I_Mau_01_PLI")
            self.assertEqual(appendix_chunk["appendix_id"], "ND145_2020_Phu_Luc_I_Mau_01_PLI")

    def test_legal_chunks_summary_reports_required_quality_fields(self) -> None:
        chunks = [
            {
                "chunk_id": "dup",
                "document_id": "doc-a",
                "level": "article",
                "chunk_type": "article_full",
                "article_number": "1",
                "citation_text": "Điều 1",
                "text": "ngắn",
            },
            {
                "chunk_id": "dup",
                "document_id": "doc-a",
                "level": "clause",
                "chunk_type": "clause",
                "article_number": "1",
                "citation_text": "",
                "text": " ".join(["dài"] * 1200),
            },
            {
                "chunk_id": "missing-article",
                "document_id": "doc-b",
                "level": "clause",
                "chunk_type": "clause",
                "article_number": None,
                "citation_text": "Preamble",
                "text": "preamble text",
            },
            {
                "chunk_id": "appendix",
                "document_id": "doc-b",
                "level": "appendix",
                "chunk_type": "appendix_table",
                "article_number": None,
                "citation_text": "Phụ lục",
                "text": "appendix text",
            },
        ]

        summary = summarize_legal_chunks(chunks)

        self.assertEqual(summary["chunk_count_by_document"]["doc-a"], 2)
        self.assertEqual(summary["chunk_count_by_level"]["clause"], 2)
        self.assertEqual(summary["chunk_count_by_chunk_type"]["article_full"], 1)
        self.assertEqual(summary["duplicate_chunk_id_count"], 1)
        self.assertEqual(len(summary["chunks_missing_citation_text"]), 1)
        self.assertEqual(len(summary["chunks_missing_article_number"]), 1)
        self.assertEqual(len(summary["very_short_chunks"]), 3)
        self.assertEqual(len(summary["very_long_chunks"]), 1)
        self.assertEqual(len(summary["very_long_normal_chunks"]), 1)


if __name__ == "__main__":
    unittest.main()
