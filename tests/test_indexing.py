from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.vn_labor_law_ai_assistant.indexing import (
    SparseBM25Encoder,
    build_index_records,
    extract_legal_hint_tokens,
    make_qdrant_point_id,
    write_records_sqlite,
)


class FakeSegmenter:
    def segment(self, text: str) -> list[str]:
        return text.lower().replace("\n", " ").split()


class IndexingTests(unittest.TestCase):
    def test_extract_legal_hint_tokens_captures_article_clause_point(self) -> None:
        text = "Trợ cấp thôi việc được tính theo Điều 46 khoản 1 điểm c.2 của Bộ luật Lao động."
        tokens = extract_legal_hint_tokens(text)

        self.assertIn("dieu_46", tokens)
        self.assertIn("khoan_1", tokens)
        self.assertIn("diem_c.2", tokens)

    def test_build_index_records_adds_sparse_hint_tokens(self) -> None:
        chunk = {
            "chunk_id": "bo-luat-dieu-46-chunk-01",
            "document_id": "bo-luat-45-2019-qh14",
            "document_title": "Bộ luật số 45/2019/QH14",
            "source_kind": "curated_text",
            "source_path": "corpus/cleaned/du_lieu.txt",
            "section_id": "bo-luat-dieu-46",
            "article_number": "46",
            "article_title": "Trợ cấp thôi việc",
            "heading": "Điều 46. Trợ cấp thôi việc",
            "chapter_heading": "Chương III. Hợp đồng lao động",
            "section_heading": "Mục 3. Chấm dứt hợp đồng lao động",
            "level": "clause",
            "clause_ref": "1",
            "point_ref": "c.2",
            "parent_chunk_id": "bo-luat-dieu-46-parent",
            "citation_text": "Bộ luật số 45/2019/QH14, Điều 46, khoản 1, điểm c.2",
            "topic": ["tro_cap"],
            "actor": ["nguoi_lao_dong"],
            "issue_type": ["tro_cap_thoi_viec"],
            "retrieval_text": "Bộ luật số 45/2019/QH14, Điều 46, khoản 1, điểm c.2 quy định: Người lao động được trợ cấp thôi việc.",
            "text": "Điều 46. Trợ cấp thôi việc\n\n1. Người lao động được trợ cấp thôi việc.",
        }

        record = build_index_records([chunk], segmenter=FakeSegmenter())[0]

        self.assertEqual(record.chunk_id, "bo-luat-dieu-46-chunk-01")
        self.assertEqual(record.parent_chunk_id, "bo-luat-dieu-46-parent")
        self.assertEqual(record.payload["article_number"], "46")
        self.assertIn("dieu_46", record.sparse_tokens)
        self.assertIn("khoan_1", record.sparse_tokens)
        self.assertIn("diem_c.2", record.sparse_tokens)
        self.assertIn("người", record.dense_text.lower())

    def test_sparse_bm25_encoder_aligns_query_and_document_indices(self) -> None:
        encoder = SparseBM25Encoder.fit(
            [
                ["tro_cap", "dieu_46", "thoi_viec"],
                ["bao_truoc", "dieu_36"],
            ]
        )

        doc_vector = encoder.encode_document(["tro_cap", "dieu_46", "thoi_viec"])
        query_vector = encoder.encode_query(["tro_cap", "dieu_46"])

        self.assertTrue(doc_vector.indices)
        self.assertTrue(query_vector.indices)

        shared_indices = set(doc_vector.indices).intersection(query_vector.indices)
        self.assertTrue(shared_indices)
        self.assertTrue(all(value > 0 for value in doc_vector.values))
        self.assertTrue(all(value > 0 for value in query_vector.values))

    def test_make_qdrant_point_id_is_deterministic(self) -> None:
        first = make_qdrant_point_id("nghi-dinh-dieu-7-chunk-03")
        second = make_qdrant_point_id("nghi-dinh-dieu-7-chunk-03")
        third = make_qdrant_point_id("nghi-dinh-dieu-7-chunk-04")

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_write_records_sqlite_persists_parent_lookup_fields(self) -> None:
        chunk = {
            "chunk_id": "nghi-dinh-dieu-7-chunk-03",
            "document_id": "nghi-dinh-145-2020-nd-cp",
            "document_title": "Nghị định 145/2020/NĐ-CP",
            "source_kind": "curated_text",
            "source_path": "corpus/cleaned/nghi-dinh-145-2020-nd-cp.txt",
            "section_id": "nghi-dinh-dieu-7",
            "article_number": "7",
            "article_title": "Thời hạn báo trước",
            "heading": "Điều 7. Thời hạn báo trước",
            "chapter_heading": None,
            "section_heading": "Mục 2. CHẤM DỨT HỢP ĐỒNG LAO ĐỘNG",
            "level": "point",
            "clause_ref": "1",
            "point_ref": "a",
            "parent_chunk_id": "nghi-dinh-dieu-7-chunk-02",
            "citation_text": "Nghị định 145/2020/NĐ-CP, Điều 7, khoản 1, điểm a",
            "topic": ["bao_truoc"],
            "actor": ["nguoi_lao_dong"],
            "issue_type": ["thoi_han_bao_truoc"],
            "retrieval_text": "Nghị định 145/2020/NĐ-CP, Điều 7, khoản 1, điểm a quy định: Thành viên tổ lái tàu bay.",
            "text": "Điều 7. Thời hạn báo trước\n\na) Thành viên tổ lái tàu bay.",
        }
        record = build_index_records([chunk], segmenter=FakeSegmenter())[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "records.db"
            write_records_sqlite([record], db_path)

            connection = sqlite3.connect(db_path)
            try:
                row = connection.execute(
                    "SELECT parent_chunk_id, citation_text, sparse_text FROM records WHERE chunk_id = ?",
                    (record.chunk_id,),
                ).fetchone()
            finally:
                connection.close()

        self.assertEqual(row[0], "nghi-dinh-dieu-7-chunk-02")
        self.assertEqual(row[1], "Nghị định 145/2020/NĐ-CP, Điều 7, khoản 1, điểm a")
        self.assertIn("dieu_7", row[2])


if __name__ == "__main__":
    unittest.main()
