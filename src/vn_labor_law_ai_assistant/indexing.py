from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
from typing import Sequence
import uuid

from .corpus_pipeline import extract_chunk_body, normalize_for_matching


LEGAL_ARTICLE_RE = re.compile(r"\bdieu\s+(?P<value>\d+[a-z]?)")
LEGAL_CLAUSE_RE = re.compile(r"\bkhoan\s+(?P<value>\d+)")
LEGAL_POINT_RE = re.compile(r"\bdiem\s+(?P<value>[a-z](?:\.\d+)?)")
TOKEN_RE = re.compile(r"[\w.]+", re.UNICODE)
SPARSE_STOPWORDS = {
    "a",
    "anh",
    "ba",
    "bị",
    "boi",
    "bởi",
    "các",
    "cả",
    "cần",
    "cho",
    "có",
    "của",
    "cùng",
    "cũng",
    "chỉ",
    "đã",
    "đang",
    "để",
    "đến",
    "đều",
    "được",
    "đó",
    "đây",
    "em",
    "hay",
    "khi",
    "không",
    "là",
    "lại",
    "lên",
    "mà",
    "mỗi",
    "một",
    "ngay",
    "này",
    "nên",
    "nếu",
    "những",
    "như",
    "nhiều",
    "ơi",
    "phải",
    "qua",
    "ra",
    "rằng",
    "rất",
    "rồi",
    "sau",
    "sẽ",
    "so",
    "sự",
    "the",
    "theo",
    "thì",
    "trên",
    "trong",
    "tôi",
    "tại",
    "từ",
    "và",
    "vẫn",
    "về",
    "vì",
    "với",
}
SPARSE_STOPWORD_NORMALIZED = {
    normalize_for_matching(word).replace(" ", "")
    for word in SPARSE_STOPWORDS
}


def require_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required for dense embedding builds.") from exc
    return SentenceTransformer


def require_qdrant():
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise RuntimeError("qdrant-client is required for hybrid index builds.") from exc
    return QdrantClient, models


def require_pyvi():
    try:
        from pyvi import ViTokenizer
    except ImportError as exc:
        raise RuntimeError("pyvi is required for Vietnamese sparse tokenization.") from exc
    return ViTokenizer


def resolve_device(requested_device: str | None = None) -> str:
    if requested_device:
        return requested_device

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_chunk_paths(chunks_dir: Path, chunk_files: Sequence[Path] | None = None) -> list[Path]:
    if chunk_files:
        return [path.resolve() for path in chunk_files]
    return sorted(path.resolve() for path in chunks_dir.glob("*.jsonl"))


def load_chunk_payloads(chunk_paths: Sequence[Path]) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for path in chunk_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payloads.append(json.loads(line))
    return payloads


def normalize_reference_token(prefix: str, value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip().lower()
    cleaned = re.sub(r"\s+", "", cleaned)
    return f"{prefix}_{cleaned}"


def extract_legal_hint_tokens(text: str) -> list[str]:
    normalized = normalize_for_matching(text)
    tokens: list[str] = []

    for match in LEGAL_ARTICLE_RE.finditer(normalized):
        tokens.append(f"dieu_{match.group('value').lower()}")
    for match in LEGAL_CLAUSE_RE.finditer(normalized):
        tokens.append(f"khoan_{match.group('value').lower()}")
    for match in LEGAL_POINT_RE.finditer(normalized):
        tokens.append(f"diem_{match.group('value').lower()}")

    return tokens


class PyViWordSegmenter:
    def __init__(self) -> None:
        self._tokenizer = require_pyvi()

    def segment(self, text: str) -> list[str]:
        segmented_text = self._tokenizer.tokenize(text)
        tokens: list[str] = []
        for token in TOKEN_RE.findall(segmented_text):
            lowered_token = token.lower()
            if is_sparse_stopword(lowered_token):
                continue
            tokens.append(lowered_token)
        return tokens


def is_sparse_stopword(token: str) -> bool:
    normalized_token = normalize_for_matching(token.replace("_", " ")).replace(" ", "")
    return normalized_token in SPARSE_STOPWORD_NORMALIZED


def build_dense_text(chunk: dict[str, object]) -> str:
    retrieval_text = str(chunk.get("retrieval_text") or "").strip()
    if retrieval_text:
        return retrieval_text

    heading = str(chunk.get("heading") or "").strip()
    text = str(chunk.get("text") or "").strip()
    body_text = extract_chunk_body(text, heading)
    citation_text = str(chunk.get("citation_text") or "").strip()
    parts = [part for part in [citation_text, body_text] if part]
    return "\n".join(parts).strip()


def build_sparse_tokens(
    chunk: dict[str, object],
    segmenter: PyViWordSegmenter,
) -> list[str]:
    heading = str(chunk.get("heading") or "").strip()
    body_text = extract_chunk_body(str(chunk.get("text") or ""), heading)
    citation_text = str(chunk.get("citation_text") or "").strip()

    text_for_sparse = "\n".join(part for part in [citation_text, heading, body_text] if part).strip()
    tokens = segmenter.segment(text_for_sparse)

    explicit_tokens = [
        normalize_reference_token("dieu", str(chunk["article_number"])) if chunk.get("article_number") else None,
        normalize_reference_token("khoan", str(chunk["clause_ref"])) if chunk.get("clause_ref") else None,
        normalize_reference_token("diem", str(chunk["point_ref"])) if chunk.get("point_ref") else None,
    ]

    tokens.extend(token for token in explicit_tokens if token)
    tokens.extend(extract_legal_hint_tokens(text_for_sparse))
    return tokens


def build_sparse_text(tokens: Sequence[str]) -> str:
    return " ".join(tokens).strip()


def make_qdrant_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


@dataclass(frozen=True)
class IndexRecord:
    chunk_id: str
    dense_text: str
    sparse_text: str
    sparse_tokens: tuple[str, ...]
    parent_chunk_id: str | None
    payload: dict[str, object]
    text: str
    citation_text: str

    @property
    def document_id(self) -> str:
        return str(self.payload["document_id"])

    @property
    def source_path(self) -> str:
        return str(self.payload["source_path"])


@dataclass
class SparseVectorData:
    indices: list[int]
    values: list[float]


@dataclass
class SparseBM25Encoder:
    vocabulary: dict[str, int]
    idf_by_token: dict[str, float]
    avg_doc_length: float
    document_count: int
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def fit(
        cls,
        tokenized_documents: Sequence[Sequence[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> SparseBM25Encoder:
        document_frequency: Counter[str] = Counter()
        vocabulary: dict[str, int] = {}
        total_doc_length = 0

        for tokens in tokenized_documents:
            total_doc_length += len(tokens)
            unique_tokens = set(tokens)
            document_frequency.update(unique_tokens)
            for token in unique_tokens:
                if token not in vocabulary:
                    vocabulary[token] = len(vocabulary)

        document_count = len(tokenized_documents)
        avg_doc_length = total_doc_length / max(document_count, 1)
        idf_by_token: dict[str, float] = {}

        for token, doc_freq in document_frequency.items():
            numerator = document_count - doc_freq + 0.5
            denominator = doc_freq + 0.5
            idf_by_token[token] = math.log(1.0 + (numerator / denominator))

        return cls(
            vocabulary=vocabulary,
            idf_by_token=idf_by_token,
            avg_doc_length=avg_doc_length,
            document_count=document_count,
            k1=k1,
            b=b,
        )

    def encode_document(self, tokens: Sequence[str]) -> SparseVectorData:
        term_frequency = Counter(token for token in tokens if token in self.vocabulary)
        doc_length = len(tokens)
        denominator_norm = self.k1 * (1.0 - self.b + self.b * (doc_length / max(self.avg_doc_length, 1e-9)))

        indices: list[int] = []
        values: list[float] = []

        for token, frequency in term_frequency.items():
            numerator = frequency * (self.k1 + 1.0)
            denominator = frequency + denominator_norm
            weight = self.idf_by_token[token] * (numerator / denominator)
            indices.append(self.vocabulary[token])
            values.append(weight)

        order = sorted(range(len(indices)), key=lambda idx: indices[idx])
        return SparseVectorData(
            indices=[indices[idx] for idx in order],
            values=[values[idx] for idx in order],
        )

    def encode_query(self, tokens: Sequence[str]) -> SparseVectorData:
        term_frequency = Counter(token for token in tokens if token in self.vocabulary)
        indices: list[int] = []
        values: list[float] = []

        for token, frequency in term_frequency.items():
            weight = self.idf_by_token[token] * (1.0 + math.log(1.0 + frequency))
            indices.append(self.vocabulary[token])
            values.append(weight)

        order = sorted(range(len(indices)), key=lambda idx: indices[idx])
        return SparseVectorData(
            indices=[indices[idx] for idx in order],
            values=[values[idx] for idx in order],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "vocabulary": self.vocabulary,
            "idf_by_token": self.idf_by_token,
            "avg_doc_length": self.avg_doc_length,
            "document_count": self.document_count,
            "k1": self.k1,
            "b": self.b,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> SparseBM25Encoder:
        return cls(
            vocabulary={str(key): int(value) for key, value in dict(payload["vocabulary"]).items()},
            idf_by_token={str(key): float(value) for key, value in dict(payload["idf_by_token"]).items()},
            avg_doc_length=float(payload["avg_doc_length"]),
            document_count=int(payload["document_count"]),
            k1=float(payload.get("k1", 1.5)),
            b=float(payload.get("b", 0.75)),
        )


def build_index_records(
    chunk_payloads: Sequence[dict[str, object]],
    *,
    segmenter: PyViWordSegmenter | None = None,
) -> list[IndexRecord]:
    segmenter = segmenter or PyViWordSegmenter()

    records: list[IndexRecord] = []
    for chunk in chunk_payloads:
        dense_text = build_dense_text(chunk)
        sparse_tokens = build_sparse_tokens(chunk, segmenter)
        payload = {
            "chunk_id": chunk["chunk_id"],
            "qdrant_point_id": make_qdrant_point_id(str(chunk["chunk_id"])),
            "document_id": chunk["document_id"],
            "document_title": chunk["document_title"],
            "source_kind": chunk["source_kind"],
            "source_path": chunk["source_path"],
            "section_id": chunk["section_id"],
            "article_number": chunk.get("article_number"),
            "article_title": chunk.get("article_title"),
            "heading": chunk["heading"],
            "chapter_heading": chunk.get("chapter_heading"),
            "section_heading": chunk.get("section_heading"),
            "level": chunk.get("level"),
            "clause_ref": chunk.get("clause_ref"),
            "point_ref": chunk.get("point_ref"),
            "parent_chunk_id": chunk.get("parent_chunk_id"),
            "citation_text": chunk.get("citation_text"),
            "topic": list(chunk.get("topic") or []),
            "actor": list(chunk.get("actor") or []),
            "issue_type": list(chunk.get("issue_type") or []),
        }
        records.append(
            IndexRecord(
                chunk_id=str(chunk["chunk_id"]),
                dense_text=dense_text,
                sparse_text=build_sparse_text(sparse_tokens),
                sparse_tokens=tuple(sparse_tokens),
                parent_chunk_id=str(chunk["parent_chunk_id"]) if chunk.get("parent_chunk_id") else None,
                payload=payload,
                text=str(chunk["text"]),
                citation_text=str(chunk.get("citation_text") or ""),
            )
        )

    return records


def embed_dense_texts(
    texts: Sequence[str],
    *,
    model_name: str,
    batch_size: int = 32,
    device: str | None = None,
) -> list[list[float]]:
    sentence_transformer_cls = require_sentence_transformers()
    model = sentence_transformer_cls(model_name, device=resolve_device(device))
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return [embedding.tolist() for embedding in embeddings]


def write_records_jsonl(records: Sequence[IndexRecord], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "chunk_id": record.chunk_id,
                "dense_text": record.dense_text,
                "sparse_text": record.sparse_text,
                "parent_chunk_id": record.parent_chunk_id,
                "citation_text": record.citation_text,
                "text": record.text,
                "payload": record.payload,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_records_sqlite(records: Sequence[IndexRecord], output_path: Path) -> None:
    connection = sqlite3.connect(output_path)
    try:
        connection.execute(
            """
            CREATE TABLE records (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                document_title TEXT NOT NULL,
                section_id TEXT NOT NULL,
                article_number TEXT,
                article_title TEXT,
                heading TEXT NOT NULL,
                chapter_heading TEXT,
                section_heading TEXT,
                level TEXT,
                clause_ref TEXT,
                point_ref TEXT,
                citation_text TEXT NOT NULL,
                dense_text TEXT NOT NULL,
                sparse_text TEXT NOT NULL,
                text TEXT NOT NULL,
                parent_chunk_id TEXT,
                source_kind TEXT NOT NULL,
                source_path TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        connection.execute("CREATE INDEX idx_records_parent_chunk_id ON records(parent_chunk_id)")
        connection.execute("CREATE INDEX idx_records_document_id ON records(document_id)")

        rows = [
            (
                record.chunk_id,
                str(record.payload["document_id"]),
                str(record.payload["document_title"]),
                str(record.payload["section_id"]),
                record.payload.get("article_number"),
                record.payload.get("article_title"),
                str(record.payload["heading"]),
                record.payload.get("chapter_heading"),
                record.payload.get("section_heading"),
                record.payload.get("level"),
                record.payload.get("clause_ref"),
                record.payload.get("point_ref"),
                record.citation_text,
                record.dense_text,
                record.sparse_text,
                record.text,
                record.parent_chunk_id,
                str(record.payload["source_kind"]),
                str(record.payload["source_path"]),
                json.dumps(record.payload, ensure_ascii=False),
            )
            for record in records
        ]
        connection.executemany(
            """
            INSERT INTO records (
                chunk_id, document_id, document_title, section_id, article_number, article_title,
                heading, chapter_heading, section_heading, level, clause_ref, point_ref,
                citation_text, dense_text, sparse_text, text, parent_chunk_id, source_kind,
                source_path, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()
    finally:
        connection.close()


def write_sparse_encoder(encoder: SparseBM25Encoder, output_path: Path) -> None:
    output_path.write_text(json.dumps(encoder.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_sparse_encoder(input_path: Path) -> SparseBM25Encoder:
    return SparseBM25Encoder.from_dict(json.loads(input_path.read_text(encoding="utf-8")))


def build_corpus_signature(chunk_paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in chunk_paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_qdrant_collection(
    *,
    records: Sequence[IndexRecord],
    dense_vectors: Sequence[Sequence[float]],
    sparse_vectors: Sequence[SparseVectorData],
    qdrant_path: Path,
    collection_name: str,
    dense_vector_name: str = "dense",
    sparse_vector_name: str = "sparse",
    batch_size: int = 128,
) -> None:
    if not records:
        raise ValueError("No records available for Qdrant indexing.")

    qdrant_client_cls, models = require_qdrant()
    client = qdrant_client_cls(path=str(qdrant_path))
    try:
        dense_size = len(dense_vectors[0])
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config={
                dense_vector_name: models.VectorParams(
                    size=dense_size,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                sparse_vector_name: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )

        for start in range(0, len(records), batch_size):
            end = start + batch_size
            batch_points = []
            for record, dense_vector, sparse_vector in zip(
                records[start:end],
                dense_vectors[start:end],
                sparse_vectors[start:end],
            ):
                batch_points.append(
                    models.PointStruct(
                        id=make_qdrant_point_id(record.chunk_id),
                        vector={
                            dense_vector_name: list(dense_vector),
                            sparse_vector_name: models.SparseVector(
                                indices=sparse_vector.indices,
                                values=sparse_vector.values,
                            ),
                        },
                        payload=record.payload,
                    )
                )

            client.upsert(collection_name=collection_name, points=batch_points, wait=True)
    finally:
        client.close()


def write_build_manifest(manifest: dict[str, object], output_path: Path) -> None:
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def update_current_pointer(artifacts_dir: Path, manifest: dict[str, object]) -> None:
    current_manifest_path = artifacts_dir / "current.json"
    temp_manifest_path = artifacts_dir / ".current.json.tmp"
    temp_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_manifest_path.replace(current_manifest_path)


def build_hybrid_index(
    *,
    chunk_paths: Sequence[Path],
    artifacts_dir: Path,
    dense_model_name: str,
    collection_name: str,
    batch_size: int = 32,
    device: str | None = None,
    build_id: str | None = None,
) -> dict[str, object]:
    timestamp = build_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifacts_dir = artifacts_dir.resolve()
    builds_dir = artifacts_dir / "builds"
    temp_build_dir = builds_dir / f".build_{timestamp}.tmp"
    final_build_dir = builds_dir / f"build_{timestamp}"

    if temp_build_dir.exists():
        shutil.rmtree(temp_build_dir)
    if final_build_dir.exists():
        raise FileExistsError(f"Build directory already exists: {final_build_dir}")

    builds_dir.mkdir(parents=True, exist_ok=True)
    temp_build_dir.mkdir(parents=True, exist_ok=True)

    chunk_payloads = load_chunk_payloads(chunk_paths)
    records = build_index_records(chunk_payloads)
    dense_vectors = embed_dense_texts(
        [record.dense_text for record in records],
        model_name=dense_model_name,
        batch_size=batch_size,
        device=device,
    )

    sparse_encoder = SparseBM25Encoder.fit([record.sparse_tokens for record in records])
    sparse_vectors = [sparse_encoder.encode_document(record.sparse_tokens) for record in records]

    qdrant_path = temp_build_dir / "qdrant"
    records_db_path = temp_build_dir / "records.db"
    records_jsonl_path = temp_build_dir / "records.jsonl"
    sparse_encoder_path = temp_build_dir / "sparse_encoder.json"
    manifest_path = temp_build_dir / "index_manifest.json"

    write_records_sqlite(records, records_db_path)
    write_records_jsonl(records, records_jsonl_path)
    write_sparse_encoder(sparse_encoder, sparse_encoder_path)
    build_qdrant_collection(
        records=records,
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
        qdrant_path=qdrant_path,
        collection_name=collection_name,
        batch_size=batch_size,
    )

    manifest = {
        "build_id": timestamp,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "collection_name": collection_name,
        "dense_model_name": dense_model_name,
        "dense_vector_name": "dense",
        "sparse_vector_name": "sparse",
        "record_count": len(records),
        "corpus_signature": build_corpus_signature(chunk_paths),
        "chunk_paths": [path.as_posix() for path in chunk_paths],
        "build_dir": final_build_dir.as_posix(),
        "qdrant_path": (final_build_dir / "qdrant").as_posix(),
        "records_db_path": (final_build_dir / "records.db").as_posix(),
        "records_jsonl_path": (final_build_dir / "records.jsonl").as_posix(),
        "sparse_encoder_path": (final_build_dir / "sparse_encoder.json").as_posix(),
        "batch_size": batch_size,
        "device": resolve_device(device),
    }
    write_build_manifest(manifest, manifest_path)

    temp_build_dir.replace(final_build_dir)
    update_current_pointer(artifacts_dir, manifest)
    return manifest


__all__ = [
    "IndexRecord",
    "PyViWordSegmenter",
    "SparseBM25Encoder",
    "SparseVectorData",
    "build_hybrid_index",
    "build_index_records",
    "build_sparse_text",
    "build_sparse_tokens",
    "extract_legal_hint_tokens",
    "is_sparse_stopword",
    "load_sparse_encoder",
    "load_chunk_payloads",
    "make_qdrant_point_id",
    "normalize_reference_token",
    "resolve_chunk_paths",
]
