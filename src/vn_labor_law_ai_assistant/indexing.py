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
import warnings

from .core.config import REPO_ROOT, load_repo_env, load_settings
from .corpus_pipeline import extract_chunk_body, normalize_for_matching
from .embeddings import embed_texts_via_http, is_custom_http_embedding_provider


load_repo_env()

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
QDRANT_KEYWORD_PAYLOAD_INDEX_FIELDS = (
    "chunk_id",
    "document_id",
    "article_number",
    "clause_ref",
    "point_ref",
    "point_refs",
    "chunk_type",
    "issue_type",
    "topic",
    "actor",
)
ARTICLE_SPARSE_HINT_TOKENS = {
    "29": ("issue_transfer_work", "temporary_transfer", "different_work_than_contract"),
    "35": ("issue_employee_unilateral", "notice_period", "no_notice_termination"),
    "36": ("issue_employer_unilateral", "poor_performance", "absent_without_reason"),
    "37": ("issue_no_employer_unilateral", "protected_leave", "maternity_protection"),
    "39": ("issue_unlawful_unilateral", "unlawful_termination"),
    "40": ("issue_employee_unlawful_termination", "training_cost_reimbursement"),
    "41": ("issue_unlawful_termination", "issue_compensation", "issue_reinstatement", "remedy_2_month_salary"),
    "46": (
        "issue_severance",
        "formula_half_month_salary",
        "condition_12_months",
        "exclude_unemployment_insurance_time",
    ),
    "47": ("issue_job_loss_allowance", "minimum_2_month_salary", "structural_change", "economic_reason"),
    "48": ("issue_post_termination", "return_social_insurance_book", "final_payment"),
    "97": ("issue_late_wage", "late_payment_interest", "wage_payment_deadline"),
    "98": ("issue_overtime_pay", "overtime_percentage", "night_work_pay"),
    "99": ("issue_work_stoppage_wage", "work_stoppage", "salary_during_stoppage"),
    "104": ("issue_bonus", "thirteenth_month_salary", "bonus_policy"),
    "113": ("issue_annual_leave", "unused_annual_leave_payment", "annual_leave_days"),
    "114": ("issue_travel_annual_leave", "annual_leave_travel_time"),
    "115": ("issue_personal_leave", "paid_personal_leave", "unpaid_leave"),
    "122": ("issue_discipline_procedure", "discipline_principles", "disciplinary_meeting"),
    "124": ("issue_discipline_forms", "dismissal_discipline_form"),
    "125": ("issue_dismissal", "dismissal_discipline", "absent_5_days"),
    "128": ("issue_temporary_suspension", "suspension_salary_advance"),
    "129": ("issue_material_liability", "damage_compensation", "salary_deduction"),
    "137": ("issue_maternity_protection", "pregnant_worker_protection"),
}
REQUIRED_INDEX_INPUT_FIELDS = (
    "chunk_id",
    "retrieval_text",
    "citation_text",
    "document_id",
    "normative_rank",
)
REQUIRED_VECTOR_PAYLOAD_FIELDS = (
    "chunk_id",
    "document_id",
    "document_title",
    "document_type",
    "normative_rank",
    "rank_label",
    "article_number",
    "article_title",
    "clause_ref",
    "point_ref",
    "point_refs",
    "level",
    "chunk_type",
    "parent_chunk_id",
    "topic",
    "actor",
    "issue_type",
    "citation_text",
    "source_file",
    "document_hierarchy",
)


def normalize_metadata_token(prefix: str, value: object) -> str | None:
    cleaned = normalize_for_matching(str(value or "")).strip().replace(" ", "_")
    cleaned = re.sub(r"[^a-z0-9_.]+", "_", cleaned).strip("_")
    return f"{prefix}_{cleaned}" if cleaned else None


def require_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required for dense embedding builds.") from exc
    return SentenceTransformer


def require_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required for cross-encoder reranking.") from exc
    return CrossEncoder


def require_qdrant():
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise RuntimeError("qdrant-client is required for hybrid index builds.") from exc
    return QdrantClient, models


def qdrant_storage_mode() -> str:
    return "cloud" if load_settings().qdrant_url.strip() else "local"


def build_qdrant_client(qdrant_client_cls, qdrant_path: Path | None = None):
    settings = load_settings()
    qdrant_url = settings.qdrant_url.strip()
    qdrant_api_key = settings.optional_secret_value(settings.qdrant_api_key)
    qdrant_timeout = float(settings.qdrant_timeout)

    if qdrant_url:
        return qdrant_client_cls(
            url=qdrant_url,
            api_key=qdrant_api_key or None,
            timeout=qdrant_timeout,
        )

    if qdrant_path is None:
        raise ValueError("qdrant_path is required when QDRANT_URL is not set.")

    return qdrant_client_cls(path=str(qdrant_path), timeout=qdrant_timeout)


def require_pyvi():
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*invalid escape sequence.*",
                category=SyntaxWarning,
                module=r"pyvi(\.|$)",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
                category=Warning,
            )
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

    page_content = str(chunk.get("page_content") or "").strip()
    heading = str(chunk.get("heading") or "").strip()
    text = str(chunk.get("text") or "").strip()
    body_text = extract_chunk_body(text, heading)
    citation_text = str(chunk.get("citation_text") or "").strip()
    document_title = str(chunk.get("document_title") or "").strip()
    article_title = str(chunk.get("article_title") or "").strip()
    issue_terms = " ".join(str(value) for value in (chunk.get("issue_type") or []))
    topic_terms = " ".join(str(value) for value in (chunk.get("topic") or []))
    metadata_text = " ".join(
        part
        for part in [document_title, citation_text, article_title, issue_terms, topic_terms]
        if part
    )
    parts = [part for part in [metadata_text, retrieval_text or page_content, body_text] if part]
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
    tokens.extend(
        token
        for value in chunk.get("point_refs") or []
        if (token := normalize_reference_token("diem", str(value)))
    )
    tokens.extend(extract_legal_hint_tokens(text_for_sparse))
    tokens.extend(
        token
        for prefix, values in (
            ("topic", chunk.get("topic") or []),
            ("issue", chunk.get("issue_type") or []),
            ("actor", chunk.get("actor") or []),
        )
        for value in values
        if (token := normalize_metadata_token(prefix, value))
    )
    article_number = str(chunk.get("article_number") or "").strip()
    tokens.extend(ARTICLE_SPARSE_HINT_TOKENS.get(article_number, ()))
    return tokens


def build_sparse_text(tokens: Sequence[str]) -> str:
    return " ".join(tokens).strip()


def make_qdrant_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def source_file_for_chunk(chunk: dict[str, object]) -> str:
    source_file = str(chunk.get("source_file") or "").strip()
    if source_file:
        return source_file
    source_path = str(chunk.get("source_path") or "").strip()
    if not source_path:
        return ""
    return Path(source_path).name


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


def build_qdrant_payload(record: IndexRecord) -> dict[str, object]:
    return {
        **record.payload,
        "text": record.text,
        "dense_text": record.dense_text,
        "sparse_text": record.sparse_text,
        "citation_text": record.citation_text,
        "parent_chunk_id": record.parent_chunk_id,
    }


def ensure_qdrant_payload_indexes(
    client,
    models,
    *,
    collection_name: str,
) -> None:
    for field_name in QDRANT_KEYWORD_PAYLOAD_INDEX_FIELDS:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )


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
            "document_type": chunk.get("document_type"),
            "normative_rank": chunk.get("normative_rank"),
            "rank_label": chunk.get("rank_label"),
            "source_kind": chunk["source_kind"],
            "source_path": chunk["source_path"],
            "source_file": source_file_for_chunk(chunk),
            "document_hierarchy": dict(chunk.get("document_hierarchy") or {}),
            "section_id": chunk["section_id"],
            "article_number": chunk.get("article_number"),
            "article_title": chunk.get("article_title"),
            "heading": chunk["heading"],
            "chapter_heading": chunk.get("chapter_heading"),
            "section_heading": chunk.get("section_heading"),
            "level": chunk.get("level"),
            "chunk_type": chunk.get("chunk_type"),
            "clause_ref": chunk.get("clause_ref"),
            "point_ref": chunk.get("point_ref"),
            "point_refs": list(chunk.get("point_refs") or []),
            "parent_chunk_id": chunk.get("parent_chunk_id"),
            "citation_text": chunk.get("citation_text"),
            "retrieval_text": str(chunk.get("retrieval_text") or ""),
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
    text_list = list(texts)
    if is_custom_http_embedding_provider():
        return embed_texts_via_http(text_list, batch_size=batch_size)

    sentence_transformer_cls = require_sentence_transformers()
    model = sentence_transformer_cls(model_name, device=resolve_device(device))
    embeddings = model.encode(
        text_list,
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
                chunk_type TEXT,
                clause_ref TEXT,
                point_ref TEXT,
                point_refs TEXT,
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
                record.payload.get("chunk_type"),
                record.payload.get("clause_ref"),
                record.payload.get("point_ref"),
                "|" + "|".join(str(value) for value in record.payload.get("point_refs") or []) + "|"
                if record.payload.get("point_refs")
                else "",
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
                heading, chapter_heading, section_heading, level, chunk_type, clause_ref, point_ref, point_refs,
                citation_text, dense_text, sparse_text, text, parent_chunk_id, source_kind,
                source_path, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_index_inputs(chunk_payloads: Sequence[dict[str, object]]) -> dict[str, object]:
    chunk_ids = [str(chunk.get("chunk_id") or "") for chunk in chunk_payloads]
    duplicate_chunk_ids = sorted(
        chunk_id for chunk_id, count in Counter(chunk_ids).items() if chunk_id and count > 1
    )
    missing: dict[str, list[str]] = {}
    for field in REQUIRED_INDEX_INPUT_FIELDS:
        missing[field] = [
            str(chunk.get("chunk_id") or f"row-{index + 1}")
            for index, chunk in enumerate(chunk_payloads)
            if chunk.get(field) in (None, "", [])
        ]

    return {
        "chunk_count": len(chunk_payloads),
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "duplicate_chunk_id_count": len(duplicate_chunk_ids),
        "missing_retrieval_text_count": len(missing["retrieval_text"]),
        "missing_citation_text_count": len(missing["citation_text"]),
        "missing_document_id_count": len(missing["document_id"]),
        "missing_normative_rank_count": len(missing["normative_rank"]),
        "missing_fields": missing,
        "passed": (
            not duplicate_chunk_ids
            and all(not values for values in missing.values())
        ),
    }


def raise_for_index_input_errors(report: dict[str, object]) -> None:
    errors: list[str] = []
    if report["duplicate_chunk_id_count"]:
        errors.append(
            "duplicate chunk_id values: "
            + ", ".join(str(value) for value in report["duplicate_chunk_ids"])
        )
    for field in REQUIRED_INDEX_INPUT_FIELDS:
        missing = dict(report["missing_fields"]).get(field, [])
        if missing:
            errors.append(f"missing {field}: {len(missing)} chunk(s)")
    if errors:
        raise ValueError("Cannot build vector index: " + "; ".join(errors))


def validate_index_build(
    *,
    chunk_payloads: Sequence[dict[str, object]],
    records: Sequence[IndexRecord],
    dense_vectors: Sequence[Sequence[float]],
    sparse_vectors: Sequence[SparseVectorData],
) -> dict[str, object]:
    input_report = validate_index_inputs(chunk_payloads)
    record_chunk_ids = [record.chunk_id for record in records]
    input_chunk_ids = [str(chunk.get("chunk_id") or "") for chunk in chunk_payloads]
    missing_indexed_chunk_ids = sorted(set(input_chunk_ids) - set(record_chunk_ids))
    extra_indexed_chunk_ids = sorted(set(record_chunk_ids) - set(input_chunk_ids))
    empty_vector_payloads = [
        record.chunk_id
        for record in records
        if not build_qdrant_payload(record)
        or any(field not in build_qdrant_payload(record) for field in REQUIRED_VECTOR_PAYLOAD_FIELDS)
    ]
    payload_missing_by_field = {
        field: [
            record.chunk_id
            for record in records
            if field not in build_qdrant_payload(record)
            or (
                field
                in {"chunk_id", "document_id", "document_title", "document_type", "normative_rank", "rank_label", "citation_text", "source_file"}
                and build_qdrant_payload(record).get(field) in (None, "", [])
            )
        ]
        for field in REQUIRED_VECTOR_PAYLOAD_FIELDS
    }
    source_document_types = sorted(
        {str(chunk.get("document_type") or "") for chunk in chunk_payloads if chunk.get("document_type")}
    )
    indexed_document_types = sorted(
        {str(record.payload.get("document_type") or "") for record in records if record.payload.get("document_type")}
    )
    source_normative_ranks = sorted(
        {int(chunk["normative_rank"]) for chunk in chunk_payloads if chunk.get("normative_rank") not in (None, "", [])}
    )
    indexed_normative_ranks = sorted(
        {int(record.payload["normative_rank"]) for record in records if record.payload.get("normative_rank") not in (None, "", [])}
    )
    dense_vector_dimensions = sorted({len(vector) for vector in dense_vectors})
    validation = {
        **input_report,
        "indexed_chunk_count": len(records),
        "all_chunks_indexed": (
            len(records) == len(chunk_payloads)
            and not missing_indexed_chunk_ids
            and not extra_indexed_chunk_ids
        ),
        "missing_indexed_chunk_ids": missing_indexed_chunk_ids,
        "extra_indexed_chunk_ids": extra_indexed_chunk_ids,
        "dense_vector_count": len(dense_vectors),
        "sparse_vector_count": len(sparse_vectors),
        "dense_vector_dimensions": dense_vector_dimensions,
        "empty_vector_payload_count": len(empty_vector_payloads),
        "empty_vector_payload_chunk_ids": empty_vector_payloads,
        "payload_missing_by_field": payload_missing_by_field,
        "source_document_types": source_document_types,
        "indexed_document_types": indexed_document_types,
        "all_document_types_preserved": source_document_types == indexed_document_types,
        "source_normative_ranks": source_normative_ranks,
        "indexed_normative_ranks": indexed_normative_ranks,
        "all_normative_ranks_preserved": source_normative_ranks == indexed_normative_ranks,
    }
    validation["passed"] = (
        bool(input_report["passed"])
        and bool(validation["all_chunks_indexed"])
        and len(dense_vectors) == len(records)
        and len(sparse_vectors) == len(records)
        and len(empty_vector_payloads) == 0
        and all(not values for values in payload_missing_by_field.values())
        and bool(validation["all_document_types_preserved"])
        and bool(validation["all_normative_ranks_preserved"])
        and len(dense_vector_dimensions) == 1
    )
    return validation


def raise_for_index_build_errors(report: dict[str, object]) -> None:
    raise_for_index_input_errors(report)
    errors: list[str] = []
    if not report["all_chunks_indexed"]:
        errors.append("not all chunks were indexed")
    if report["dense_vector_count"] != report["indexed_chunk_count"]:
        errors.append("dense vector count does not match indexed chunk count")
    if report["sparse_vector_count"] != report["indexed_chunk_count"]:
        errors.append("sparse vector count does not match indexed chunk count")
    if report["empty_vector_payload_count"]:
        errors.append(f"empty or incomplete vector payloads: {report['empty_vector_payload_count']}")
    if not report["all_document_types_preserved"]:
        errors.append("document types were not preserved in vector payloads")
    if not report["all_normative_ranks_preserved"]:
        errors.append("normative ranks were not preserved in vector payloads")
    if len(report["dense_vector_dimensions"]) != 1:
        errors.append("dense vectors have inconsistent dimensions")
    if errors:
        raise ValueError("Invalid vector index build: " + "; ".join(errors))


def indexed_documents_for_manifest(records: Sequence[IndexRecord]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for record in records:
        document_id = str(record.payload.get("document_id") or "")
        if not document_id:
            continue
        item = grouped.setdefault(
            document_id,
            {
                "document_id": document_id,
                "document_title": record.payload.get("document_title"),
                "document_type": record.payload.get("document_type"),
                "normative_rank": record.payload.get("normative_rank"),
                "rank_label": record.payload.get("rank_label"),
                "chunk_count": 0,
            },
        )
        item["chunk_count"] = int(item["chunk_count"]) + 1
    return sorted(grouped.values(), key=lambda item: str(item["document_id"]))


def warnings_for_index_build(
    validation_report: dict[str, object],
    records: Sequence[IndexRecord],
) -> list[str]:
    warnings_list: list[str] = []
    if not records:
        warnings_list.append("no records were generated")
    if not validation_report["passed"]:
        warnings_list.append("index validation did not pass")
    return warnings_list


def render_vector_index_summary_markdown(manifest: dict[str, object]) -> str:
    validation = dict(manifest.get("validation") or {})
    lines = [
        "# Vector Index Summary",
        "",
        f"- Build ID: {manifest['build_id']}",
        f"- Generated at: {manifest['generated_at']}",
        f"- Embedding model: {manifest['embedding_model']}",
        f"- Chunk count: {manifest['chunk_count']}",
        f"- Document count: {manifest['document_count']}",
        f"- Vector dimension: {manifest['vector_dimension']}",
        f"- Collection: {manifest['collection_name']}",
        f"- Qdrant storage: {manifest['qdrant_storage']}",
        f"- Source chunks file: {manifest['source_chunks_file']}",
        f"- Source chunks file hash: {manifest['source_chunks_file_hash']}",
        "",
        "## Validation",
        "",
        f"- All chunks indexed: {validation.get('all_chunks_indexed')}",
        f"- Duplicate chunk IDs: {validation.get('duplicate_chunk_id_count')}",
        f"- Missing retrieval_text: {validation.get('missing_retrieval_text_count')}",
        f"- Missing citation_text: {validation.get('missing_citation_text_count')}",
        f"- Missing document_id: {validation.get('missing_document_id_count')}",
        f"- Missing normative_rank: {validation.get('missing_normative_rank_count')}",
        f"- Empty vector payloads: {validation.get('empty_vector_payload_count')}",
        f"- All document types preserved: {validation.get('all_document_types_preserved')}",
        f"- All normative ranks preserved: {validation.get('all_normative_ranks_preserved')}",
        f"- Passed: {validation.get('passed')}",
        "",
        "## Indexed Documents",
        "",
        "| Document | Type | Rank | Chunks |",
        "| --- | --- | ---: | ---: |",
    ]
    for document in manifest.get("indexed_documents") or []:
        if not isinstance(document, dict):
            continue
        lines.append(
            "| {document_id} | {document_type} | {normative_rank} | {chunk_count} |".format(
                document_id=document.get("document_id"),
                document_type=document.get("document_type"),
                normative_rank=document.get("normative_rank"),
                chunk_count=document.get("chunk_count"),
            )
        )
    warnings_list = manifest.get("warnings") or []
    lines.extend(["", "## Warnings", ""])
    if warnings_list:
        for warning in warnings_list:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")
    return "\n".join(lines).strip() + "\n"


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
    client = build_qdrant_client(qdrant_client_cls, qdrant_path)
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
        ensure_qdrant_payload_indexes(
            client,
            models,
            collection_name=collection_name,
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
                        payload=build_qdrant_payload(record),
                    )
                )

            client.upsert(collection_name=collection_name, points=batch_points, wait=True)
    finally:
        client.close()


def write_build_manifest(manifest: dict[str, object], output_path: Path) -> None:
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def path_for_manifest(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def update_current_pointer(artifacts_dir: Path, manifest: dict[str, object]) -> None:
    current_manifest_path = artifacts_dir / "current.json"
    temp_manifest_path = artifacts_dir / ".current.json.tmp"
    temp_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_manifest_path.replace(current_manifest_path)


def write_root_index_artifacts(artifacts_dir: Path, manifest: dict[str, object]) -> None:
    manifest_path = artifacts_dir / "manifest.json"
    summary_json_path = artifacts_dir / "vector_index_summary.json"
    summary_md_path = artifacts_dir / "vector_index_summary.md"
    temp_manifest_path = artifacts_dir / ".manifest.json.tmp"
    temp_summary_json_path = artifacts_dir / ".vector_index_summary.json.tmp"
    temp_summary_md_path = artifacts_dir / ".vector_index_summary.md.tmp"
    temp_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_summary_json_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_summary_md_path.write_text(render_vector_index_summary_markdown(manifest), encoding="utf-8")
    temp_manifest_path.replace(manifest_path)
    temp_summary_json_path.replace(summary_json_path)
    temp_summary_md_path.replace(summary_md_path)


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
    input_validation_report = validate_index_inputs(chunk_payloads)
    raise_for_index_input_errors(input_validation_report)
    records = build_index_records(chunk_payloads)
    dense_vectors = embed_dense_texts(
        [record.dense_text for record in records],
        model_name=dense_model_name,
        batch_size=batch_size,
        device=device,
    )

    sparse_encoder = SparseBM25Encoder.fit([record.sparse_tokens for record in records])
    sparse_vectors = [sparse_encoder.encode_document(record.sparse_tokens) for record in records]
    validation_report = validate_index_build(
        chunk_payloads=chunk_payloads,
        records=records,
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
    )
    raise_for_index_build_errors(validation_report)

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

    storage_mode = qdrant_storage_mode()
    vector_dimension = len(dense_vectors[0]) if dense_vectors else 0
    source_chunks_file = (
        path_for_manifest(chunk_paths[0])
        if len(chunk_paths) == 1
        else [path_for_manifest(path) for path in chunk_paths]
    )
    source_chunks_file_hash = (
        file_sha256(chunk_paths[0])
        if len(chunk_paths) == 1
        else build_corpus_signature(chunk_paths)
    )
    manifest = {
        "build_id": timestamp,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": dense_model_name,
        "chunk_count": len(records),
        "document_count": len({record.document_id for record in records}),
        "source_chunks_file": source_chunks_file,
        "source_chunks_file_hash": source_chunks_file_hash,
        "vector_dimension": vector_dimension,
        "indexed_documents": indexed_documents_for_manifest(records),
        "warnings": warnings_for_index_build(validation_report, records),
        "collection_name": collection_name,
        "qdrant_storage": storage_mode,
        "record_source": "qdrant_payload" if storage_mode == "cloud" else "sqlite",
        "dense_model_name": dense_model_name,
        "dense_vector_name": "dense",
        "sparse_vector_name": "sparse",
        "record_count": len(records),
        "corpus_signature": build_corpus_signature(chunk_paths),
        "chunk_paths": [path_for_manifest(path) for path in chunk_paths],
        "build_dir": path_for_manifest(final_build_dir),
        "qdrant_path": path_for_manifest(final_build_dir / "qdrant"),
        "records_db_path": path_for_manifest(final_build_dir / "records.db"),
        "records_jsonl_path": path_for_manifest(final_build_dir / "records.jsonl"),
        "sparse_encoder_path": path_for_manifest(final_build_dir / "sparse_encoder.json"),
        "batch_size": batch_size,
        "device": resolve_device(device),
        "validation": validation_report,
    }
    write_build_manifest(manifest, manifest_path)

    temp_build_dir.replace(final_build_dir)
    update_current_pointer(artifacts_dir, manifest)
    write_root_index_artifacts(artifacts_dir, manifest)
    return manifest


__all__ = [
    "IndexRecord",
    "PyViWordSegmenter",
    "SparseBM25Encoder",
    "SparseVectorData",
    "build_qdrant_client",
    "build_hybrid_index",
    "build_index_records",
    "build_qdrant_payload",
    "build_sparse_text",
    "build_sparse_tokens",
    "file_sha256",
    "ensure_qdrant_payload_indexes",
    "extract_legal_hint_tokens",
    "indexed_documents_for_manifest",
    "is_sparse_stopword",
    "load_sparse_encoder",
    "load_chunk_payloads",
    "make_qdrant_point_id",
    "normalize_reference_token",
    "raise_for_index_build_errors",
    "raise_for_index_input_errors",
    "render_vector_index_summary_markdown",
    "qdrant_storage_mode",
    "require_cross_encoder",
    "require_qdrant",
    "resolve_chunk_paths",
    "source_file_for_chunk",
    "validate_index_build",
    "validate_index_inputs",
]
