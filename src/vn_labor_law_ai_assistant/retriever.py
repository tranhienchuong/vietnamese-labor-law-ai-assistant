from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import re
from typing import Sequence

from .corpus_pipeline import normalize_for_matching
from .indexing import (
    PyViWordSegmenter,
    SparseBM25Encoder,
    extract_legal_hint_tokens,
    load_sparse_encoder,
    require_qdrant,
    require_sentence_transformers,
)


ARTICLE_REF_RE = re.compile(r"\bdieu\s+(?P<value>\d+[a-z]?)")
CLAUSE_REF_RE = re.compile(r"\bkhoan\s+(?P<value>\d+)")
POINT_REF_RE = re.compile(r"\bdiem\s+(?P<value>[a-z](?:\.\d+)?)")

DOCUMENT_KEYWORDS = {
    "du-lieu-cham-dut-hop-dong-lao-dong": (
        "bo luat lao dong",
        "bo luat 2019",
        "45/2019",
        "qh 14",
        "qh14",
    ),
    "nghi-dinh-145-2020-nd-cp": (
        "nghi dinh 145",
        "145/2020",
        "nd-cp",
    ),
}

ACTOR_KEYWORDS = {
    "nguoi_lao_dong": (
        "nguoi lao dong",
        "nhan vien",
        "cong nhan",
        "toi ",
        " em ",
        " minh ",
        "bi cong ty",
        "nghi viec",
        "xin nghi",
    ),
    "nguoi_su_dung_lao_dong": (
        "nguoi su dung lao dong",
        "cong ty",
        "doanh nghiep",
        "chu su dung",
        "nguoi su dung",
    ),
    "lao_dong_nu": (
        "lao dong nu",
        "mang thai",
        "thai san",
        "nuoi con duoi 12 thang",
    ),
    "nguoi_lao_dong_nuoc_ngoai": (
        "lao dong nuoc ngoai",
        "nguoi nuoc ngoai",
        "giay phep lao dong",
    ),
}

TOPIC_KEYWORDS = {
    "cham_dut_hop_dong_lao_dong": (
        "cham dut hop dong",
        "ket thuc hop dong",
        "duoi viec",
        "cho nghi viec",
        "nghi viec",
        "cham dut hdld",
    ),
    "don_phuong_cham_dut": (
        "don phuong",
        "tu nghi",
        "nghi ngang",
        "co duoc nghi ngay",
    ),
    "tro_cap": (
        "tro cap",
        "thoi viec",
        "mat viec",
    ),
    "bao_truoc": (
        "bao truoc",
        "thoi han bao truoc",
        "bao truoc bao lau",
    ),
    "ky_luat_sa_thai": (
        "sa thai",
        "ky luat",
        "noi quy lao dong",
    ),
    "thay_doi_co_cau_kinh_te": (
        "thay doi co cau",
        "ly do kinh te",
        "sap nhap",
        "chia tach",
    ),
    "tam_hoan_hop_dong": (
        "tam hoan",
        "tam dung hop dong",
    ),
    "bao_ve_thai_san": (
        "thai san",
        "mang thai",
        "nuoi con duoi 12 thang",
    ),
    "dao_tao_nghe": (
        "dao tao",
        "hoc nghe",
        "chi phi dao tao",
    ),
    "hop_dong_lao_dong": (
        "hop dong lao dong",
        "hdld",
        "giao ket hop dong",
    ),
}

ISSUE_KEYWORDS = {
    "can_cu_cham_dut": (
        "truong hop nao",
        "khi nao duoc",
        "co duoc cham dut",
        "can cu",
        "ly do",
    ),
    "quyen_don_phuong_cham_dut": (
        "don phuong",
        "tu nghi",
        "co duoc nghi ngay",
    ),
    "thoi_han_bao_truoc": (
        "bao truoc",
        "bao truoc bao lau",
        "thoi han bao truoc",
    ),
    "tro_cap_thoi_viec": (
        "tro cap thoi viec",
        "tinh tro cap thoi viec",
    ),
    "tro_cap_mat_viec": (
        "tro cap mat viec",
        "mat viec lam",
    ),
    "nghia_vu_khi_cham_dut": (
        "thanh toan",
        "tra so",
        "xac nhan thoi gian dong bhxh",
        "nghia vu",
    ),
    "trai_phap_luat": (
        "trai luat",
        "trai phap luat",
        "sai luat",
    ),
    "boi_thuong": (
        "boi thuong",
        "den bu",
    ),
    "sa_thai": (
        "sa thai",
    ),
    "noi_quy_lao_dong": (
        "noi quy",
        "ky luat",
    ),
    "thong_bao_cham_dut": (
        "thong bao",
        "bao truoc",
    ),
}


@dataclass(frozen=True)
class QueryIntent:
    raw_query: str
    normalized_query: str
    actor_filters: tuple[str, ...]
    topic_filters: tuple[str, ...]
    issue_filters: tuple[str, ...]
    document_filters: tuple[str, ...]
    article_number: str | None = None
    clause_ref: str | None = None
    point_ref: str | None = None

    @property
    def legal_reference_filters(self) -> tuple[tuple[str, str], ...]:
        filters: list[tuple[str, str]] = []
        if self.article_number:
            filters.append(("article_number", self.article_number))
        if self.clause_ref:
            filters.append(("clause_ref", self.clause_ref))
        if self.point_ref:
            filters.append(("point_ref", self.point_ref))
        return tuple(filters)


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    qdrant_point_id: str
    score: float
    citation_text: str
    payload: dict[str, object]


@dataclass(frozen=True)
class RetrievedRecord:
    chunk_id: str
    parent_chunk_id: str | None
    citation_text: str
    text: str
    dense_text: str
    sparse_text: str
    payload: dict[str, object]


@dataclass(frozen=True)
class RetrievalContext:
    chunk_id: str
    citation_text: str
    text: str
    payload: dict[str, object]
    score: float
    matched_chunk_ids: tuple[str, ...]
    matched_citations: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    intent: QueryIntent
    hits: tuple[SearchHit, ...]
    contexts: tuple[RetrievalContext, ...]


def dedupe_preserve_order(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def parse_reference_value(pattern: re.Pattern[str], normalized_query: str) -> str | None:
    match = pattern.search(normalized_query)
    if not match:
        return None
    return match.group("value").lower()


def collect_keyword_matches(normalized_query: str, mapping: dict[str, Sequence[str]]) -> tuple[str, ...]:
    matches = [
        label
        for label, keywords in mapping.items()
        if any(keyword in normalized_query for keyword in keywords)
    ]
    return tuple(matches)


def route_query(query: str) -> QueryIntent:
    normalized_query = normalize_for_matching(f" {query} ")
    return QueryIntent(
        raw_query=query,
        normalized_query=normalized_query,
        actor_filters=collect_keyword_matches(normalized_query, ACTOR_KEYWORDS),
        topic_filters=collect_keyword_matches(normalized_query, TOPIC_KEYWORDS),
        issue_filters=collect_keyword_matches(normalized_query, ISSUE_KEYWORDS),
        document_filters=collect_keyword_matches(normalized_query, DOCUMENT_KEYWORDS),
        article_number=parse_reference_value(ARTICLE_REF_RE, normalized_query),
        clause_ref=parse_reference_value(CLAUSE_REF_RE, normalized_query),
        point_ref=parse_reference_value(POINT_REF_RE, normalized_query),
    )


def load_manifest(index_path: Path) -> dict[str, object]:
    if index_path.is_dir():
        manifest_path = index_path / "current.json"
    else:
        manifest_path = index_path
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def format_intent_summary(intent: QueryIntent) -> str:
    parts: list[str] = []
    if intent.document_filters:
        parts.append(f"document={', '.join(intent.document_filters)}")
    if intent.actor_filters:
        parts.append(f"actor={', '.join(intent.actor_filters)}")
    if intent.topic_filters:
        parts.append(f"topic={', '.join(intent.topic_filters)}")
    if intent.issue_filters:
        parts.append(f"issue={', '.join(intent.issue_filters)}")
    if intent.article_number:
        parts.append(f"dieu={intent.article_number}")
    if intent.clause_ref:
        parts.append(f"khoan={intent.clause_ref}")
    if intent.point_ref:
        parts.append(f"diem={intent.point_ref}")
    return "; ".join(parts) if parts else "khong co filter heuristic"


def format_context_for_prompt(contexts: Sequence[RetrievalContext]) -> str:
    blocks: list[str] = []
    for index, context in enumerate(contexts, start=1):
        matched_citations = "\n".join(f"- {citation}" for citation in context.matched_citations)
        blocks.append(
            "\n".join(
                [
                    f"[NGU CANH {index}]",
                    f"Co so phap ly: {context.citation_text}",
                    f"Match goc:",
                    matched_citations,
                    "Noi dung:",
                    context.text.strip(),
                ]
            ).strip()
        )
    return "\n\n".join(blocks).strip()


class HybridRetriever:
    def __init__(
        self,
        *,
        index_path: Path = Path("artifacts/index"),
        device: str | None = None,
    ) -> None:
        self._manifest = load_manifest(index_path)
        self._device = device or str(self._manifest.get("device") or "cpu")
        self._records_db_path = Path(str(self._manifest["records_db_path"]))
        self._qdrant_path = Path(str(self._manifest["qdrant_path"]))
        self._collection_name = str(self._manifest["collection_name"])
        self._dense_model_name = str(self._manifest["dense_model_name"])
        self._dense_vector_name = str(self._manifest["dense_vector_name"])
        self._sparse_vector_name = str(self._manifest["sparse_vector_name"])
        self._segmenter = PyViWordSegmenter()
        self._sparse_encoder = load_sparse_encoder(Path(str(self._manifest["sparse_encoder_path"])))
        self._sqlite = sqlite3.connect(self._records_db_path)
        self._sqlite.row_factory = sqlite3.Row
        qdrant_client_cls, self._qdrant_models = require_qdrant()
        self._qdrant = qdrant_client_cls(path=str(self._qdrant_path))
        self._dense_model = None

    @property
    def manifest(self) -> dict[str, object]:
        return dict(self._manifest)

    def close(self) -> None:
        self._qdrant.close()
        self._sqlite.close()

    def _get_dense_model(self):
        if self._dense_model is None:
            sentence_transformer_cls = require_sentence_transformers()
            self._dense_model = sentence_transformer_cls(self._dense_model_name, device=self._device)
        return self._dense_model

    def _encode_dense_query(self, query: str) -> list[float]:
        model = self._get_dense_model()
        vector = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return vector.tolist()

    def _encode_sparse_query(self, intent: QueryIntent) -> tuple[list[str], object]:
        tokens = self._segmenter.segment(intent.raw_query)
        tokens.extend(extract_legal_hint_tokens(intent.raw_query))
        if intent.article_number:
            tokens.append(f"dieu_{intent.article_number}")
        if intent.clause_ref:
            tokens.append(f"khoan_{intent.clause_ref}")
        if intent.point_ref:
            tokens.append(f"diem_{intent.point_ref}")
        sparse_query = self._sparse_encoder.encode_query(tokens)
        sparse_vector = self._qdrant_models.SparseVector(
            indices=sparse_query.indices,
            values=sparse_query.values,
        )
        return tokens, sparse_vector

    def _build_query_filter(self, intent: QueryIntent):
        must_conditions: list[object] = []
        ranked_conditions: list[object] = []
        models = self._qdrant_models

        if intent.document_filters:
            must_conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=list(intent.document_filters)),
                )
            )

        for field_name, value in intent.legal_reference_filters:
            must_conditions.append(
                models.FieldCondition(
                    key=field_name,
                    match=models.MatchAny(any=[value]),
                )
            )

        if intent.actor_filters:
            ranked_conditions.append(
                models.FieldCondition(
                    key="actor",
                    match=models.MatchAny(any=list(intent.actor_filters)),
                )
            )
        if intent.topic_filters:
            ranked_conditions.append(
                models.FieldCondition(
                    key="topic",
                    match=models.MatchAny(any=list(intent.topic_filters)),
                )
            )
        if intent.issue_filters:
            ranked_conditions.append(
                models.FieldCondition(
                    key="issue_type",
                    match=models.MatchAny(any=list(intent.issue_filters)),
                )
            )

        if not must_conditions and not ranked_conditions:
            return None

        if ranked_conditions:
            return models.Filter(
                must=must_conditions or None,
                min_should=models.MinShould(conditions=ranked_conditions, min_count=1),
            )

        return models.Filter(must=must_conditions)

    def _fetch_records(self, chunk_ids: Sequence[str]) -> dict[str, RetrievedRecord]:
        ordered_ids = dedupe_preserve_order(chunk_ids)
        if not ordered_ids:
            return {}

        placeholders = ", ".join("?" for _ in ordered_ids)
        rows = self._sqlite.execute(
            f"""
            SELECT chunk_id, parent_chunk_id, citation_text, text, dense_text, sparse_text, payload_json
            FROM records
            WHERE chunk_id IN ({placeholders})
            """,
            ordered_ids,
        ).fetchall()

        records: dict[str, RetrievedRecord] = {}
        for row in rows:
            payload = json.loads(row["payload_json"])
            records[str(row["chunk_id"])] = RetrievedRecord(
                chunk_id=str(row["chunk_id"]),
                parent_chunk_id=str(row["parent_chunk_id"]) if row["parent_chunk_id"] else None,
                citation_text=str(row["citation_text"]),
                text=str(row["text"]),
                dense_text=str(row["dense_text"]),
                sparse_text=str(row["sparse_text"]),
                payload=payload,
            )
        return records

    def _assemble_contexts(self, hits: Sequence[SearchHit]) -> tuple[RetrievalContext, ...]:
        direct_ids = [hit.chunk_id for hit in hits]
        direct_records = self._fetch_records(direct_ids)
        parent_ids = [
            record.parent_chunk_id
            for record in direct_records.values()
            if record.parent_chunk_id and record.parent_chunk_id not in direct_records
        ]
        parent_records = self._fetch_records(parent_ids)
        record_map = {**direct_records, **parent_records}

        grouped: dict[str, dict[str, object]] = {}
        for rank, hit in enumerate(hits):
            matched_record = direct_records.get(hit.chunk_id)
            if matched_record is None:
                continue

            context_id = matched_record.parent_chunk_id or matched_record.chunk_id
            context_record = record_map.get(context_id, matched_record)
            group = grouped.setdefault(
                context_record.chunk_id,
                {
                    "record": context_record,
                    "score": hit.score,
                    "rank": rank,
                    "matched_chunk_ids": [],
                    "matched_citations": [],
                },
            )
            group["score"] = max(float(group["score"]), hit.score)
            group["rank"] = min(int(group["rank"]), rank)
            group["matched_chunk_ids"].append(hit.chunk_id)
            group["matched_citations"].append(hit.citation_text)

        ordered_groups = sorted(
            grouped.values(),
            key=lambda item: (-float(item["score"]), int(item["rank"])),
        )

        contexts: list[RetrievalContext] = []
        for item in ordered_groups:
            record = item["record"]
            assert isinstance(record, RetrievedRecord)
            matched_chunk_ids = dedupe_preserve_order(item["matched_chunk_ids"])
            matched_citations = dedupe_preserve_order(item["matched_citations"])
            contexts.append(
                RetrievalContext(
                    chunk_id=record.chunk_id,
                    citation_text=record.citation_text,
                    text=record.text,
                    payload=record.payload,
                    score=float(item["score"]),
                    matched_chunk_ids=matched_chunk_ids,
                    matched_citations=matched_citations,
                )
            )
        return tuple(contexts)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 8,
        prefetch_limit: int = 24,
    ) -> RetrievalResult:
        intent = route_query(query)
        query_filter = self._build_query_filter(intent)
        dense_query = self._encode_dense_query(query)
        _, sparse_query = self._encode_sparse_query(intent)
        models = self._qdrant_models

        response = self._qdrant.query_points(
            collection_name=self._collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_query,
                    using=self._dense_vector_name,
                    filter=query_filter,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=sparse_query,
                    using=self._sparse_vector_name,
                    filter=query_filter,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        hits = tuple(
            SearchHit(
                chunk_id=str(point.payload["chunk_id"]),
                qdrant_point_id=str(point.payload["qdrant_point_id"]),
                score=float(point.score),
                citation_text=str(point.payload.get("citation_text") or ""),
                payload=dict(point.payload),
            )
            for point in response.points
        )
        contexts = self._assemble_contexts(hits)
        return RetrievalResult(
            query=query,
            intent=intent,
            hits=hits,
            contexts=contexts,
        )


__all__ = [
    "HybridRetriever",
    "QueryIntent",
    "RetrievalContext",
    "RetrievalResult",
    "SearchHit",
    "RetrievedRecord",
    "dedupe_preserve_order",
    "format_context_for_prompt",
    "format_intent_summary",
    "load_manifest",
    "route_query",
]
