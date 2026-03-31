from __future__ import annotations

import unittest

from src.vn_labor_law_ai_assistant.corpus_pipeline import (
    PageRecord,
    chunk_sections,
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

    def test_infer_document_title_uses_source_hint(self) -> None:
        text = (
            "BỘ DỮ LIỆU TRÍCH XUẤT TỪ BỘ LUẬT LAO ĐỘNG 2019\n"
            "Nguồn: Bộ luật số 45/2019/QH14\n"
        )
        self.assertEqual(infer_document_title(text, "fallback"), "Bộ luật số 45/2019/QH14")

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


if __name__ == "__main__":
    unittest.main()
