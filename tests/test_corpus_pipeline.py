from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.vn_labor_law_ai_assistant.corpus_pipeline import (
    PageRecord,
    build_corpus,
    chunk_sections,
    enrich_chunk,
    infer_document_title,
    normalize_extracted_text,
    slugify_text,
    split_sections,
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
        body_lookup = {
            chunk["text"].split("\n\n", 1)[1]: chunk
            for chunk in chunks
            if "\n\n" in chunk["text"]
        }

        clause_chunk = next(chunk for body, chunk in body_lookup.items() if "Trường hợp người lao động có thời gian" in body)
        point_chunk = next(chunk for body, chunk in body_lookup.items() if body.startswith("c.2)"))
        subpoint_chunk = next(chunk for body, chunk in body_lookup.items() if body.startswith("c.3)"))

        self.assertEqual(clause_chunk["clause_ref"], "2")
        self.assertIsNone(clause_chunk["point_ref"])
        self.assertEqual(point_chunk["clause_ref"], "4")
        self.assertEqual(point_chunk["point_ref"], "c.2")
        self.assertEqual(subpoint_chunk["clause_ref"], "4")
        self.assertEqual(subpoint_chunk["point_ref"], "c.3")

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
        article_chunk = next(chunk for chunk in chunks if chunk["clause_ref"] is None and chunk["point_ref"] is None)
        clause_one_chunk = next(chunk for chunk in chunks if chunk["clause_ref"] == "1" and chunk["point_ref"] is None)
        clause_two_chunk = next(chunk for chunk in chunks if chunk["clause_ref"] == "2" and chunk["point_ref"] is None)
        point_a_chunk = next(chunk for chunk in chunks if chunk["point_ref"] == "a")
        point_c_chunk = next(chunk for chunk in chunks if chunk["point_ref"] == "c")
        point_c2_chunk = next(chunk for chunk in chunks if chunk["point_ref"] == "c.2")

        self.assertEqual(clause_one_chunk["parent_chunk_id"], article_chunk["chunk_id"])
        self.assertEqual(clause_two_chunk["parent_chunk_id"], article_chunk["chunk_id"])
        self.assertEqual(point_a_chunk["parent_chunk_id"], clause_one_chunk["chunk_id"])
        self.assertEqual(point_c_chunk["parent_chunk_id"], clause_two_chunk["chunk_id"])
        self.assertEqual(point_c2_chunk["parent_chunk_id"], point_c_chunk["chunk_id"])

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


if __name__ == "__main__":
    unittest.main()
