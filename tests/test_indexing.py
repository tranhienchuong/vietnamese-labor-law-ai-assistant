from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from src.vn_labor_law_ai_assistant.indexing import (
    PyViWordSegmenter,
    SparseBM25Encoder,
    build_index_records,
    build_qdrant_client,
    build_qdrant_payload,
    embed_dense_texts,
    ensure_qdrant_payload_indexes,
    extract_legal_hint_tokens,
    is_sparse_stopword,
    make_qdrant_point_id,
    write_records_sqlite,
)


class FakeSegmenter:
    def segment(self, text: str) -> list[str]:
        return text.lower().replace("\n", " ").split()


class IndexingTests(unittest.TestCase):
    def test_build_qdrant_client_uses_cloud_url_when_configured(self) -> None:
        class FakeQdrantClient:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        with patch.dict(
            "os.environ",
            {
                "QDRANT_URL": "https://example.qdrant.io",
                "QDRANT_API_KEY": "secret",
            },
        ):
            client = build_qdrant_client(FakeQdrantClient, Path("ignored"))

        self.assertEqual(client.kwargs["url"], "https://example.qdrant.io")
        self.assertEqual(client.kwargs["api_key"], "secret")
        self.assertNotIn("path", client.kwargs)

    def test_build_qdrant_client_falls_back_to_local_path(self) -> None:
        class FakeQdrantClient:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        with patch.dict("os.environ", {"QDRANT_URL": "", "QDRANT_API_KEY": ""}):
            client = build_qdrant_client(FakeQdrantClient, Path("artifacts/index/qdrant"))

        self.assertEqual(client.kwargs, {"path": str(Path("artifacts/index/qdrant"))})

    def test_embed_dense_texts_uses_custom_http_provider(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {
                    "EMBEDDING_PROVIDER": "custom_http",
                    "EMBEDDING_API_URL": "https://embedding.example/v1/embeddings",
                },
                clear=False,
            ),
            patch(
                "src.vn_labor_law_ai_assistant.indexing.embed_texts_via_http",
                return_value=[[1.0, 2.0]],
            ) as embed_http,
            patch("src.vn_labor_law_ai_assistant.indexing.require_sentence_transformers") as require_local,
        ):
            vectors = embed_dense_texts(["mot cau"], model_name="ignored", batch_size=7)

        self.assertEqual(vectors, [[1.0, 2.0]])
        embed_http.assert_called_once_with(["mot cau"], batch_size=7)
        require_local.assert_not_called()

    def test_ensure_qdrant_payload_indexes_creates_keyword_indexes(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str, str, bool]] = []

            def create_payload_index(
                self,
                *,
                collection_name: str,
                field_name: str,
                field_schema: str,
                wait: bool,
            ) -> None:
                self.calls.append((collection_name, field_name, field_schema, wait))

        class FakeModels:
            class PayloadSchemaType:
                KEYWORD = "keyword"

        client = FakeClient()

        ensure_qdrant_payload_indexes(
            client,
            FakeModels,
            collection_name="labor_law_hybrid",
        )

        indexed_fields = {call[1] for call in client.calls}
        self.assertTrue(
            {
                "chunk_id",
                "document_id",
                "article_number",
                "clause_ref",
                "point_ref",
                "issue_type",
                "topic",
                "actor",
            }.issubset(indexed_fields)
        )
        self.assertTrue(all(call[2] == "keyword" and call[3] for call in client.calls))

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
        self.assertIn("issue_tro_cap_thoi_viec", record.sparse_tokens)
        self.assertIn("topic_tro_cap", record.sparse_tokens)
        self.assertIn("actor_nguoi_lao_dong", record.sparse_tokens)
        self.assertIn("issue_severance", record.sparse_tokens)
        self.assertIn("formula_half_month_salary", record.sparse_tokens)
        self.assertIn("người", record.dense_text.lower())

    def test_build_qdrant_payload_contains_runtime_text_fields(self) -> None:
        chunk = {
            "chunk_id": "bo-luat-dieu-46-chunk-01",
            "document_id": "bo-luat-45-2019-qh14",
            "document_title": "Bo luat so 45/2019/QH14",
            "source_kind": "curated_text",
            "source_path": "corpus/cleaned/du_lieu.txt",
            "section_id": "bo-luat-dieu-46",
            "article_number": "46",
            "article_title": "Tro cap thoi viec",
            "heading": "Dieu 46. Tro cap thoi viec",
            "chapter_heading": None,
            "section_heading": "Muc 3. Cham dut hop dong lao dong",
            "level": "clause",
            "clause_ref": "1",
            "point_ref": None,
            "parent_chunk_id": "bo-luat-dieu-46-parent",
            "citation_text": "Bo luat so 45/2019/QH14, Dieu 46, khoan 1",
            "topic": ["tro_cap"],
            "actor": ["nguoi_lao_dong"],
            "issue_type": ["tro_cap_thoi_viec"],
            "text": "Moi nam lam viec duoc tro cap mot nua thang tien luong.",
        }

        record = build_index_records([chunk], segmenter=FakeSegmenter())[0]
        payload = build_qdrant_payload(record)

        self.assertEqual(payload["chunk_id"], record.chunk_id)
        self.assertEqual(payload["text"], record.text)
        self.assertEqual(payload["dense_text"], record.dense_text)
        self.assertEqual(payload["sparse_text"], record.sparse_text)
        self.assertEqual(payload["citation_text"], record.citation_text)
        self.assertEqual(payload["parent_chunk_id"], record.parent_chunk_id)

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

    def test_is_sparse_stopword_filters_common_function_words(self) -> None:
        self.assertTrue(is_sparse_stopword("theo"))
        self.assertTrue(is_sparse_stopword("của"))
        self.assertFalse(is_sparse_stopword("trợ_cấp"))
        self.assertFalse(is_sparse_stopword("dieu_46"))

    def test_pyvi_segmenter_removes_stopwords(self) -> None:
        segmenter = PyViWordSegmenter()

        tokens = segmenter.segment("Theo quy định của pháp luật, trợ cấp thôi việc được tính như thế nào?")

        self.assertIn("trợ_cấp", tokens)
        self.assertIn("thôi_việc", tokens)
        self.assertNotIn("theo", tokens)
        self.assertNotIn("của", tokens)
        self.assertNotIn("được", tokens)

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
