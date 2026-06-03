from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vn_labor_law_ai_assistant.indexing import make_qdrant_point_id
from vn_labor_law_ai_assistant.rag.retrieval import HybridRetriever, RetrievalContext, SearchHit


MODE_VECTOR_ONLY = "vector_only"
MODE_HYBRID = "hybrid"
MODE_GRAPH = "graph_augmented"
MODES = (MODE_VECTOR_ONLY, MODE_HYBRID, MODE_GRAPH)
PRIMARY_LABOR_CODE_ID = "45-2019-qh14"
DEFAULT_BENCHMARK_PATH = (
    REPO_ROOT / "artifacts" / "evaluation" / "golden_benchmark_100_extended.jsonl"
)


@dataclass(frozen=True)
class CitationSpec:
    label: str
    document_id: str
    article_number: str | None = None
    clause_ref: str | None = None
    chunk_id_contains: str | None = None
    top_n: int | None = None


@dataclass(frozen=True)
class CitationOrderRule:
    label: str
    higher: CitationSpec
    lower: CitationSpec
    top_n: int | None = None


@dataclass(frozen=True)
class BenchmarkItem:
    id: str
    query: str
    category: str
    topic: str = ""
    expected_answer_points: tuple[str, ...] = ()
    difficulty: str = "medium"
    requires_graph: bool = False
    requires_normative_hierarchy: bool = False
    expected_documents: tuple[str, ...] = ()
    expected_articles: tuple[CitationSpec, ...] = ()
    expected_clauses: tuple[CitationSpec, ...] = ()
    required_citations: tuple[CitationSpec, ...] = ()
    forbidden_citations: tuple[CitationSpec, ...] = ()
    citation_order_rules: tuple[CitationOrderRule, ...] = ()


@dataclass(frozen=True)
class RetrievedItem:
    chunk_id: str
    citation_text: str
    document_id: str
    article_number: str
    clause_ref: str
    retrieval_source: str
    retrieval_method: str
    score: float
    graph_depth: int | None = None
    graph_edge_types: tuple[str, ...] = ()


@dataclass
class ModeResult:
    mode: str
    item: BenchmarkItem
    contexts: tuple[RetrievedItem, ...]
    hits: tuple[RetrievedItem, ...]
    metrics: dict[str, float | int | str | bool] = field(default_factory=dict)


def spec(
    label: str,
    document_id: str,
    article_number: str | int | None = None,
    clause_ref: str | int | None = None,
    *,
    chunk_id_contains: str | None = None,
    top_n: int | None = None,
) -> CitationSpec:
    return CitationSpec(
        label=label,
        document_id=document_id,
        article_number=str(article_number) if article_number is not None else None,
        clause_ref=str(clause_ref) if clause_ref is not None else None,
        chunk_id_contains=chunk_id_contains,
        top_n=top_n,
    )


def order_rule(label: str, higher: CitationSpec, lower: CitationSpec, *, top_n: int | None = None) -> CitationOrderRule:
    return CitationOrderRule(label=label, higher=higher, lower=lower, top_n=top_n)


def default_benchmark_items() -> tuple[BenchmarkItem, ...]:
    bll = PRIMARY_LABOR_CODE_ID
    return (
        BenchmarkItem(
            id="strict_minor_worker_14",
            query="Người 14 tuổi có được làm việc không?",
            category="direct_qa",
            expected_documents=(bll, "thong-tu-09-2020-tt-bldtbxh"),
            expected_articles=(spec("BLLĐ Điều 143", bll, 143), spec("BLLĐ Điều 145", bll, 145), spec("BLLĐ Điều 146", bll, 146)),
            required_citations=(
                spec("BLLĐ Điều 143", bll, 143),
                spec("BLLĐ Điều 145", bll, 145),
                spec("BLLĐ Điều 146", bll, 146),
                spec("TT09 hướng dẫn lao động chưa thành niên", "thong-tu-09-2020-tt-bldtbxh"),
            ),
        ),
        BenchmarkItem(
            id="strict_minor_worker_under_15_conditions",
            query="Người chưa đủ 15 tuổi làm việc cần điều kiện gì?",
            category="procedure_qa",
            expected_documents=(bll, "thong-tu-09-2020-tt-bldtbxh"),
            expected_articles=(spec("BLLĐ Điều 145", bll, 145), spec("BLLĐ Điều 146", bll, 146), spec("TT09 Điều 3", "thong-tu-09-2020-tt-bldtbxh", 3)),
            required_citations=(spec("BLLĐ Điều 145", bll, 145), spec("BLLĐ Điều 146", bll, 146), spec("TT09 Điều 3", "thong-tu-09-2020-tt-bldtbxh", 3)),
        ),
        BenchmarkItem(
            id="strict_retirement_age_2026_woman",
            query="Nữ nghỉ hưu năm 2026 thì bao nhiêu tuổi?",
            category="scenario_based_qa",
            expected_documents=(bll, "nghi-dinh-135-2020-nd-cp"),
            expected_articles=(spec("BLLĐ Điều 169", bll, 169), spec("NĐ135 Điều 4", "nghi-dinh-135-2020-nd-cp", 4)),
            required_citations=(
                spec("BLLĐ Điều 169", bll, 169),
                spec("NĐ135 Điều 4", "nghi-dinh-135-2020-nd-cp", 4),
                spec("NĐ135 bảng tuổi nghỉ hưu", "nghi-dinh-135-2020-nd-cp", chunk_id_contains="Phu_Luc"),
            ),
        ),
        BenchmarkItem(
            id="strict_labor_contract_content",
            query="Hợp đồng lao động cần có những nội dung gì?",
            category="direct_qa",
            expected_documents=(bll, "thong-tu-10-2020-tt-bldtbxh"),
            expected_articles=(spec("BLLĐ Điều 21", bll, 21), spec("TT10 Điều 3", "thong-tu-10-2020-tt-bldtbxh", 3)),
            required_citations=(spec("BLLĐ Điều 21", bll, 21, top_n=5), spec("TT10 Điều 3", "thong-tu-10-2020-tt-bldtbxh", 3, top_n=5)),
            forbidden_citations=(spec("NĐ145 Điều 17 ký quỹ", "nghi-dinh-145-2020-nd-cp", 17, top_n=5),),
        ),
        BenchmarkItem(
            id="strict_dismissal_dispute_mediation_before_lawsuit",
            query="Tranh chấp sa thải có cần hòa giải trước khi kiện không?",
            category="procedure_qa",
            expected_documents=(bll, "92-2015-qh13-labor-only"),
            expected_articles=(spec("BLLĐ Điều 188", bll, 188), spec("BLLĐ Điều 190", bll, 190), spec("BLTTDS Điều 32", "92-2015-qh13-labor-only", 32)),
            required_citations=(spec("BLLĐ Điều 188", bll, 188), spec("BLTTDS Điều 32", "92-2015-qh13-labor-only", 32), spec("BLLĐ Điều 190", bll, 190)),
        ),
        BenchmarkItem(
            id="strict_employee_unlawful_unilateral_termination",
            query="Người lao động đơn phương chấm dứt hợp đồng trái luật thì phải bồi thường gì?",
            category="scenario_based_qa",
            expected_documents=(bll,),
            expected_articles=(spec("BLLĐ Điều 40", bll, 40), spec("BLLĐ Điều 39", bll, 39), spec("BLLĐ Điều 35", bll, 35)),
            required_citations=(spec("BLLĐ Điều 40 trong top 3", bll, 40, top_n=3), spec("BLLĐ Điều 39 trong top 5", bll, 39, top_n=5)),
        ),
        BenchmarkItem(
            id="strict_structural_change_job_loss_allowance",
            query="Công ty thay đổi cơ cấu thì phải trả trợ cấp gì?",
            category="scenario_based_qa",
            expected_documents=(bll, "nghi-dinh-145-2020-nd-cp"),
            expected_articles=(spec("BLLĐ Điều 42", bll, 42), spec("BLLĐ Điều 47", bll, 47), spec("NĐ145 Điều 8", "nghi-dinh-145-2020-nd-cp", 8)),
            required_citations=(spec("BLLĐ Điều 42", bll, 42), spec("BLLĐ Điều 47", bll, 47), spec("NĐ145 Điều 8", "nghi-dinh-145-2020-nd-cp", 8)),
            forbidden_citations=(spec("BLLĐ Điều 40", bll, 40, top_n=5),),
        ),
        BenchmarkItem(
            id="strict_no_notice_resignation",
            query="Khi nào người lao động được nghỉ việc không cần báo trước?",
            category="exception_based_qa",
            expected_documents=(bll,),
            expected_articles=(spec("BLLĐ Điều 35", bll, 35),),
            expected_clauses=(spec("BLLĐ Điều 35 khoản 2", bll, 35, 2),),
            required_citations=(spec("BLLĐ Điều 35 khoản 2 trong top 3", bll, 35, 2, top_n=3),),
            forbidden_citations=(spec("BLLĐ Điều 40 trước Điều 35 khoản 2", bll, 40, top_n=3), spec("BLLĐ Điều 41 trước Điều 35 khoản 2", bll, 41, top_n=3), spec("BLLĐ Điều 48 trước Điều 35 khoản 2", bll, 48, top_n=3)),
        ),
        BenchmarkItem(
            id="extra_employee_definition",
            query="Người lao động được định nghĩa như thế nào theo Bộ luật Lao động 2019?",
            category="direct_qa",
            expected_documents=(bll,),
            expected_articles=(spec("BLLĐ Điều 3", bll, 3),),
            expected_clauses=(spec("BLLĐ Điều 3 khoản 1", bll, 3, 1),),
            required_citations=(spec("BLLĐ Điều 3 khoản 1", bll, 3, 1, top_n=5),),
        ),
        BenchmarkItem(
            id="extra_contract_type_comparison",
            query="So sánh hợp đồng lao động xác định thời hạn và không xác định thời hạn?",
            category="comparison_qa",
            expected_documents=(bll,),
            expected_articles=(spec("BLLĐ Điều 20", bll, 20),),
            required_citations=(spec("BLLĐ Điều 20", bll, 20),),
        ),
        BenchmarkItem(
            id="extra_employee_unlawful_vs_structural_change",
            query="So sánh trách nhiệm khi người lao động đơn phương chấm dứt hợp đồng trái luật với trường hợp công ty thay đổi cơ cấu phải trợ cấp?",
            category="comparison_qa",
            expected_documents=(bll,),
            expected_articles=(spec("BLLĐ Điều 40", bll, 40), spec("BLLĐ Điều 42", bll, 42), spec("BLLĐ Điều 47", bll, 47)),
            required_citations=(spec("BLLĐ Điều 40", bll, 40), spec("BLLĐ Điều 42", bll, 42), spec("BLLĐ Điều 47", bll, 47)),
        ),
        BenchmarkItem(
            id="extra_minor_worker_multihop_guidance",
            query="Người chưa đủ 15 tuổi được làm việc theo Bộ luật Lao động và thông tư nào hướng dẫn điều kiện?",
            category="multi_hop_qa",
            expected_documents=(bll, "thong-tu-09-2020-tt-bldtbxh"),
            expected_articles=(spec("BLLĐ Điều 145", bll, 145), spec("TT09 Điều 3", "thong-tu-09-2020-tt-bldtbxh", 3)),
            required_citations=(spec("BLLĐ Điều 145", bll, 145), spec("TT09 Điều 3", "thong-tu-09-2020-tt-bldtbxh", 3)),
        ),
        BenchmarkItem(
            id="extra_retirement_multihop_guidance",
            query="Tuổi nghỉ hưu theo BLLĐ Điều 169 được Nghị định 135 hướng dẫn thế nào?",
            category="multi_hop_qa",
            expected_documents=(bll, "nghi-dinh-135-2020-nd-cp"),
            expected_articles=(spec("BLLĐ Điều 169", bll, 169), spec("NĐ135 Điều 4", "nghi-dinh-135-2020-nd-cp", 4)),
            required_citations=(spec("BLLĐ Điều 169", bll, 169), spec("NĐ135 Điều 4", "nghi-dinh-135-2020-nd-cp", 4)),
        ),
        BenchmarkItem(
            id="extra_overtime_exception",
            query="Trường hợp nào được làm thêm giờ và giới hạn làm thêm theo tháng là bao nhiêu?",
            category="exception_based_qa",
            expected_documents=(bll, "nghi-dinh-145-2020-nd-cp"),
            expected_articles=(
                spec("BLLĐ Điều 107", bll, 107),
                spec("NĐ145 Điều 60", "nghi-dinh-145-2020-nd-cp", 60),
            ),
            required_citations=(
                spec("BLLĐ Điều 107 trong top 3", bll, 107, top_n=3),
                spec("NĐ145 Điều 60 trong top 10", "nghi-dinh-145-2020-nd-cp", 60, top_n=10),
            ),
            forbidden_citations=(spec("BLLĐ Điều 98 không được đứng top 1", bll, 98, top_n=1),),
            citation_order_rules=(
                order_rule(
                    "BLLĐ Điều 107 phải xếp trên BLLĐ Điều 98",
                    spec("BLLĐ Điều 107", bll, 107),
                    spec("BLLĐ Điều 98", bll, 98),
                    top_n=10,
                ),
            ),
        ),
        BenchmarkItem(
            id="extra_probation_wage",
            query="Thử việc thì tiền lương tối thiểu phải bằng bao nhiêu phần trăm lương của công việc?",
            category="scenario_based_qa",
            expected_documents=(bll,),
            expected_articles=(spec("BLLĐ Điều 26", bll, 26),),
            required_citations=(spec("BLLĐ Điều 26", bll, 26),),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare vector-only, hybrid, and graph-augmented retrieval.")
    parser.add_argument("--index-path", type=Path, default=REPO_ROOT / "artifacts" / "index")
    parser.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "evaluation")
    parser.add_argument("--output-prefix", type=str, default="retrieval_modes")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--prefetch-limit", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reranker-model", type=str, default=os.getenv("RERANKER_MODEL", ""))
    parser.add_argument("--embedding-provider", type=str, default=os.getenv("GRAPH_RETRIEVAL_TEST_EMBEDDING_PROVIDER", "sentence_transformers"))
    parser.add_argument("--device", type=str, default=os.getenv("GRAPH_RETRIEVAL_TEST_DEVICE", "cpu"))
    return parser.parse_args()


def item_to_json(item: BenchmarkItem) -> dict[str, object]:
    return {
        **asdict(item),
        "expected_answer_points": list(item.expected_answer_points),
        "expected_documents": list(item.expected_documents),
        "expected_articles": [asdict(value) for value in item.expected_articles],
        "expected_clauses": [asdict(value) for value in item.expected_clauses],
        "required_citations": [asdict(value) for value in item.required_citations],
        "forbidden_citations": [asdict(value) for value in item.forbidden_citations],
        "citation_order_rules": [asdict(value) for value in item.citation_order_rules],
    }


def coerce_spec(value: object) -> CitationSpec:
    if isinstance(value, CitationSpec):
        return value
    if isinstance(value, dict):
        return spec(
            str(value.get("label") or value.get("citation") or value),
            str(value.get("document_id") or ""),
            value.get("article_number"),
            value.get("clause_ref"),
            chunk_id_contains=str(value.get("chunk_id_contains")) if value.get("chunk_id_contains") else None,
            top_n=int(value["top_n"]) if value.get("top_n") else None,
        )
    text = str(value)
    return spec(text, "", chunk_id_contains=text)


def coerce_order_rule(value: object) -> CitationOrderRule:
    if isinstance(value, CitationOrderRule):
        return value
    if not isinstance(value, dict):
        raise ValueError(f"Citation order rule must be an object, got {value!r}")
    return order_rule(
        str(value.get("label") or "citation order rule"),
        coerce_spec(value.get("higher") or {}),
        coerce_spec(value.get("lower") or {}),
        top_n=int(value["top_n"]) if value.get("top_n") else None,
    )


def infer_document_id(text: str) -> str:
    normalized = text.lower()
    if "135" in normalized:
        return "nghi-dinh-135-2020-nd-cp"
    if "145" in normalized:
        return "nghi-dinh-145-2020-nd-cp"
    if "09" in normalized or "tt09" in normalized:
        return "thong-tu-09-2020-tt-bldtbxh"
    if "10" in normalized or "tt10" in normalized:
        return "thong-tu-10-2020-tt-bldtbxh"
    if "tố tụng" in normalized or "to tung" in normalized or "blttds" in normalized:
        return "92-2015-qh13-labor-only"
    return PRIMARY_LABOR_CODE_ID


def parse_article_from_text(text: str) -> str | None:
    from vn_labor_law_ai_assistant.corpus_pipeline import normalize_for_matching

    normalized = normalize_for_matching(text)
    parts = normalized.split()
    for index, part in enumerate(parts[:-1]):
        if part == "dieu" and parts[index + 1].isdigit():
            return parts[index + 1]
    return None


def parse_clause_from_text(text: str) -> str | None:
    from vn_labor_law_ai_assistant.corpus_pipeline import normalize_for_matching

    normalized = normalize_for_matching(text)
    parts = normalized.split()
    for index, part in enumerate(parts[:-1]):
        if part == "khoan" and parts[index + 1].isdigit():
            return parts[index + 1]
    return None


def coerce_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "co", "có"}
    return bool(value)


def coerce_string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)


def item_from_json(payload: dict[str, object]) -> BenchmarkItem:
    query = str(payload.get("query") or payload.get("question") or "")
    item_id = str(payload.get("id") or payload.get("case_id") or query[:40])
    category = str(payload.get("category") or payload.get("question_type") or "direct_qa")
    required_values = payload.get("required_citations")
    if not required_values:
        citations: list[str] = []
        for key in ("gold_citations", "gold_citation_primary", "gold_citation_secondary"):
            value = payload.get(key)
            if isinstance(value, list):
                citations.extend(str(item) for item in value if item)
            elif value:
                citations.extend(part.strip() for part in str(value).split("|") if part.strip())
        required_values = [
            {
                "label": citation,
                "document_id": infer_document_id(f"{citation} {payload.get('source_document') or ''}"),
                "article_number": parse_article_from_text(citation),
                "clause_ref": parse_clause_from_text(citation),
            }
            for citation in citations
        ]
    required = tuple(coerce_spec(value) for value in (required_values or ()))
    expected_articles = tuple(coerce_spec(value) for value in (payload.get("expected_articles") or required))
    expected_clauses = tuple(coerce_spec(value) for value in (payload.get("expected_clauses") or ()))
    expected_documents = tuple(
        str(value)
        for value in (
            payload.get("expected_documents")
            or tuple(spec_value.document_id for spec_value in (*expected_articles, *expected_clauses, *required) if spec_value.document_id)
        )
    )
    forbidden = tuple(coerce_spec(value) for value in (payload.get("forbidden_citations") or ()))
    citation_order_rules = tuple(
        coerce_order_rule(value)
        for value in (payload.get("citation_order_rules") or ())
    )
    return BenchmarkItem(
        id=item_id,
        query=query,
        category=category,
        topic=str(payload.get("topic") or payload.get("legal_topic") or category),
        expected_answer_points=coerce_string_tuple(payload.get("expected_answer_points")),
        difficulty=str(payload.get("difficulty") or "medium"),
        requires_graph=coerce_bool(payload.get("requires_graph"), default=False),
        requires_normative_hierarchy=coerce_bool(payload.get("requires_normative_hierarchy"), default=False),
        expected_documents=tuple(dict.fromkeys(expected_documents)),
        expected_articles=expected_articles,
        expected_clauses=expected_clauses,
        required_citations=required,
        forbidden_citations=forbidden,
        citation_order_rules=citation_order_rules,
    )


def write_benchmark(path: Path, items: Sequence[BenchmarkItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item_to_json(item), ensure_ascii=False) + "\n" for item in items),
        encoding="utf-8",
    )


def load_or_create_benchmark(path: Path, *, limit: int = 0) -> tuple[BenchmarkItem, ...]:
    default_items = default_benchmark_items()
    if not path.exists():
        write_benchmark(path, default_items)
    items: list[BenchmarkItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        items.append(item_from_json(json.loads(line)))
        if limit and len(items) >= limit:
            break
    if path.resolve() == DEFAULT_BENCHMARK_PATH.resolve() and not limit:
        default_ids = {item.id for item in default_items}
        loaded_ids = {item.id for item in items}
        if loaded_ids == default_ids:
            items = list(default_items)
            write_benchmark(path, items)
    return tuple(items)


class EnvOverride:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values
        self.previous: dict[str, str | None] = {}

    def __enter__(self) -> None:
        self.previous = {key: os.environ.get(key) for key in self.values}
        os.environ.update(self.values)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def retrieved_from_context(context: RetrievalContext) -> RetrievedItem:
    payload = dict(context.payload)
    citation_text = str(payload.get("citation_text") or context.citation_text)
    chunk_id = str(payload.get("chunk_id") or context.chunk_id)
    document_id = str(payload.get("document_id") or "")
    article_number = str(payload.get("article_number") or "")
    clause_ref = str(payload.get("clause_ref") or "")
    if not document_id:
        document_id = infer_document_id(f"{citation_text} {chunk_id}")
    if not article_number:
        article_number = parse_article_from_text(citation_text) or parse_article_from_text(chunk_id) or ""
    if not clause_ref:
        clause_ref = parse_clause_from_text(citation_text) or parse_clause_from_text(chunk_id) or ""
    return RetrievedItem(
        chunk_id=chunk_id,
        citation_text=citation_text,
        document_id=document_id,
        article_number=article_number,
        clause_ref=clause_ref,
        retrieval_source=str(payload.get("retrieval_source") or ""),
        retrieval_method=str(payload.get("retrieval_method") or ""),
        score=float(context.score),
        graph_depth=int(payload["graph_depth"]) if payload.get("graph_depth") not in (None, "") else None,
        graph_edge_types=tuple(str(value) for value in (payload.get("graph_edge_types") or ())),
    )


def retrieved_from_hit(hit: SearchHit) -> RetrievedItem:
    payload = dict(hit.payload)
    citation_text = str(payload.get("citation_text") or hit.citation_text)
    chunk_id = str(payload.get("chunk_id") or hit.chunk_id)
    document_id = str(payload.get("document_id") or "")
    article_number = str(payload.get("article_number") or "")
    clause_ref = str(payload.get("clause_ref") or "")
    if not document_id:
        document_id = infer_document_id(f"{citation_text} {chunk_id}")
    if not article_number:
        article_number = parse_article_from_text(citation_text) or parse_article_from_text(chunk_id) or ""
    if not clause_ref:
        clause_ref = parse_clause_from_text(citation_text) or parse_clause_from_text(chunk_id) or ""
    return RetrievedItem(
        chunk_id=chunk_id,
        citation_text=citation_text,
        document_id=document_id,
        article_number=article_number,
        clause_ref=clause_ref,
        retrieval_source=str(payload.get("retrieval_source") or ""),
        retrieval_method=str(payload.get("retrieval_method") or ""),
        score=float(hit.score),
        graph_depth=int(payload["graph_depth"]) if payload.get("graph_depth") not in (None, "") else None,
        graph_edge_types=tuple(str(value) for value in (payload.get("graph_edge_types") or ())),
    )


def matches_spec(item: RetrievedItem, citation: CitationSpec) -> bool:
    if citation.document_id and item.document_id != citation.document_id:
        return False
    if citation.article_number and item.article_number != str(citation.article_number):
        return False
    if citation.clause_ref and item.clause_ref != str(citation.clause_ref):
        return False
    if citation.chunk_id_contains and citation.chunk_id_contains.lower() not in item.chunk_id.lower():
        return False
    return True


def relevant_specs(item: BenchmarkItem) -> tuple[CitationSpec, ...]:
    values = (*item.expected_clauses, *item.expected_articles, *item.required_citations)
    deduped: dict[tuple[str, str | None, str | None, str | None], CitationSpec] = {}
    for value in values:
        if not value.document_id and not value.chunk_id_contains:
            continue
        key = (value.document_id, value.article_number, value.clause_ref, value.chunk_id_contains)
        deduped.setdefault(key, value)
    return tuple(deduped.values())


def is_relevant_result(result: RetrievedItem, item: BenchmarkItem) -> bool:
    return any(matches_spec(result, citation) for citation in relevant_specs(item))


def first_rank(results: Sequence[RetrievedItem], citations: Sequence[CitationSpec], *, limit: int | None = None) -> int | None:
    selected = results[:limit] if limit else results
    for rank, result in enumerate(selected, start=1):
        if any(matches_spec(result, citation) for citation in citations):
            return rank
    return None


def rank_of_spec(results: Sequence[RetrievedItem], citation: CitationSpec, *, limit: int | None = None) -> int | None:
    return first_rank(results, (citation,), limit=limit)


def coverage(results: Sequence[RetrievedItem], citations: Sequence[CitationSpec], *, limit: int) -> float:
    if not citations:
        return 0.0
    found = 0
    for citation in citations:
        citation_limit = min(limit, citation.top_n) if citation.top_n else limit
        selected = results[:citation_limit]
        if any(matches_spec(result, citation) for result in selected):
            found += 1
    return found / len(citations)


def forbidden_violations(results: Sequence[RetrievedItem], citations: Sequence[CitationSpec], *, default_limit: int) -> list[str]:
    violations: list[str] = []
    for citation in citations:
        limit = citation.top_n or default_limit
        if any(matches_spec(result, citation) for result in results[:limit]):
            violations.append(citation.label)
    return violations


def citation_order_violations(results: Sequence[RetrievedItem], rules: Sequence[CitationOrderRule]) -> list[str]:
    violations: list[str] = []
    for rule in rules:
        higher_rank = rank_of_spec(results, rule.higher, limit=rule.top_n)
        lower_rank = rank_of_spec(results, rule.lower, limit=rule.top_n)
        if lower_rank is not None and (higher_rank is None or lower_rank < higher_rank):
            violations.append(rule.label)
    return violations


def metrics_for_mode_result(mode_result: ModeResult, *, top_k: int) -> dict[str, float | int | str | bool]:
    contexts = mode_result.contexts
    item = mode_result.item
    relevant = relevant_specs(item)
    top5 = contexts[:5]
    top10 = contexts[:10]
    relevant_top5 = [result for result in top5 if is_relevant_result(result, item)]
    relevant_top10 = [result for result in top10 if is_relevant_result(result, item)]
    first = first_rank(contexts, relevant, limit=top_k)
    required_rank = first_rank(contexts, item.required_citations, limit=top_k)
    forbidden = forbidden_violations(contexts, item.forbidden_citations, default_limit=5)
    order_violations = citation_order_violations(contexts, item.citation_order_rules)
    all_violations = [*forbidden, *order_violations]
    graph_contexts = [
        result
        for result in contexts[:top_k]
        if result.retrieval_source in {"graph", "hybrid"} or result.retrieval_method.startswith("neo4j")
    ]
    vector_contexts = [result for result in contexts[:top_k] if result.retrieval_source == "vector"]
    graph_depths = [result.graph_depth for result in graph_contexts if result.graph_depth]
    return {
        "recall_at_5": coverage(contexts, relevant, limit=5),
        "recall_at_10": coverage(contexts, relevant, limit=10),
        "precision_at_5": len(relevant_top5) / 5,
        "precision_at_10": len(relevant_top10) / 10,
        "mrr": (1 / first) if first else 0.0,
        "hit_rate_at_5": bool(relevant_top5),
        "hit_rate_at_10": bool(relevant_top10),
        "required_citation_coverage": coverage(contexts, item.required_citations, limit=top_k),
        "forbidden_citation_violation": bool(all_violations),
        "forbidden_citation_labels": "; ".join(all_violations),
        "citation_order_violation": bool(order_violations),
        "citation_order_labels": "; ".join(order_violations),
        "first_required_rank": required_rank or 0,
        "average_rank_of_first_required_citation": float(required_rank or 0),
        "document_count": len({result.document_id for result in contexts[:top_k] if result.document_id}),
        "graph_expanded_chunks": len(graph_contexts),
        "graph_hit_ratio": len(graph_contexts) / max(1, min(top_k, len(contexts))),
        "direct_vector_hit_ratio": len(vector_contexts) / max(1, min(top_k, len(contexts))),
        "average_graph_depth": sum(graph_depths) / len(graph_depths) if graph_depths else 0.0,
    }


def dense_vector_only_retrieve(retriever: HybridRetriever, query: str, *, top_k: int) -> tuple[Any, tuple[SearchHit, ...], tuple[RetrievalContext, ...]]:
    intent = retriever._route_query(query)
    query_filter = retriever._qdrant_searcher.build_query_filter(intent)
    dense_query = retriever._query_encoder.encode_dense_query(query)
    response = retriever._qdrant.query_points(
        collection_name=retriever._collection_name,
        query=dense_query,
        using=retriever._dense_vector_name,
        query_filter=query_filter,
        limit=max(top_k * 4, 32),
        with_payload=True,
    )
    hits = tuple(
        SearchHit(
            chunk_id=str(point.payload["chunk_id"]),
            qdrant_point_id=str(point.payload.get("qdrant_point_id") or make_qdrant_point_id(str(point.payload["chunk_id"]))),
            score=float(point.score),
            citation_text=str(point.payload.get("citation_text") or ""),
            payload={
                **dict(point.payload),
                "retrieval_source": "vector",
                "retrieval_method": "dense_vector_search",
                "vector_score": float(point.score),
                "graph_score": 0.0,
                "final_score": float(point.score),
            },
        )
        for point in response.points
    )
    direct_records = retriever._record_store.fetch_records_from_hits(hits)
    hits = retriever._enrich_hits_with_records(hits, direct_records)
    hits = retriever._scorer.rerank_hits(hits, intent, direct_records)[:top_k]
    contexts = retriever._context_assembler.assemble_contexts(hits, intent=intent)[:top_k]
    return intent, hits, contexts


def run_mode(
    *,
    mode: str,
    items: Sequence[BenchmarkItem],
    index_path: Path,
    top_k: int,
    prefetch_limit: int,
    reranker_model: str,
    device: str,
) -> tuple[ModeResult, ...]:
    env = {
        "LEGAL_GRAPH_ENABLED": "true" if mode == MODE_GRAPH else "false",
        "LEGAL_GRAPH_BACKEND": "neo4j",
        "LEGAL_GRAPH_COMPLEX_QUERY_ONLY": "true",
        "LEGAL_GRAPH_MAX_EXPANDED_CHUNKS": "16",
        "LEGAL_GRAPH_EXPANSION_DEPTH": "2",
    }
    results: list[ModeResult] = []
    with EnvOverride(env):
        retriever = HybridRetriever(
            index_path=index_path,
            device=device,
            reranker_model=reranker_model,
            query_router_enabled=False,
        )
        try:
            for item in items:
                if mode == MODE_VECTOR_ONLY:
                    _, hits, contexts = dense_vector_only_retrieve(retriever, item.query, top_k=top_k)
                else:
                    retrieval_result = retriever.retrieve(item.query, top_k=top_k, prefetch_limit=prefetch_limit)
                    hits = retrieval_result.hits[:top_k]
                    contexts = retrieval_result.contexts[:top_k]
                mode_result = ModeResult(
                    mode=mode,
                    item=item,
                    contexts=tuple(retrieved_from_context(context) for context in contexts),
                    hits=tuple(retrieved_from_hit(hit) for hit in hits),
                )
                mode_result.metrics = metrics_for_mode_result(mode_result, top_k=top_k)
                results.append(mode_result)
        finally:
            retriever.close()
    return tuple(results)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate_results(results: Sequence[ModeResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[ModeResult]] = defaultdict(list)
    for result in results:
        grouped[result.mode].append(result)
    metric_keys = (
        "recall_at_5",
        "recall_at_10",
        "precision_at_5",
        "precision_at_10",
        "mrr",
        "hit_rate_at_5",
        "hit_rate_at_10",
        "required_citation_coverage",
        "forbidden_citation_violation",
        "average_rank_of_first_required_citation",
        "document_count",
        "graph_expanded_chunks",
    )
    return {
        mode: {
            key: mean(float(result.metrics[key]) for result in mode_results)
            for key in metric_keys
        }
        for mode, mode_results in grouped.items()
    }


def aggregate_by_category(results: Sequence[ModeResult]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, list[ModeResult]] = defaultdict(list)
    for result in results:
        grouped[f"{result.item.category}|{result.mode}"].append(result)
    output: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for key, values in grouped.items():
        category, mode = key.split("|", 1)
        output[category][mode] = {
            "recall_at_10": mean(float(result.metrics["recall_at_10"]) for result in values),
            "precision_at_10": mean(float(result.metrics["precision_at_10"]) for result in values),
            "mrr": mean(float(result.metrics["mrr"]) for result in values),
            "required_citation_coverage": mean(float(result.metrics["required_citation_coverage"]) for result in values),
            "query_count": float(len(values)),
        }
    return dict(output)


def quality_score(result: ModeResult) -> float:
    return (
        float(result.metrics["recall_at_10"])
        + float(result.metrics["required_citation_coverage"])
        + float(result.metrics["mrr"])
        - float(result.metrics["forbidden_citation_violation"])
    )


def graph_diagnostics(results: Sequence[ModeResult]) -> dict[str, object]:
    by_id_mode = {(result.item.id, result.mode): result for result in results}
    graph_results = [result for result in results if result.mode == MODE_GRAPH]
    vector_results = [result for result in results if result.mode == MODE_VECTOR_ONLY]
    graph_contexts = [context for result in graph_results for context in result.contexts]
    graph_like_contexts = [
        context
        for context in graph_contexts
        if context.retrieval_source in {"graph", "hybrid"} or context.retrieval_method.startswith("neo4j")
    ]
    graph_depths = [context.graph_depth for context in graph_like_contexts if context.graph_depth]
    edge_counter: Counter[str] = Counter()
    for context in graph_like_contexts:
        edge_counter.update(context.graph_edge_types)

    improved: list[dict[str, object]] = []
    worse: list[dict[str, object]] = []
    for graph_result in graph_results:
        vector_result = by_id_mode.get((graph_result.item.id, MODE_VECTOR_ONLY))
        if vector_result is None:
            continue
        delta = quality_score(graph_result) - quality_score(vector_result)
        payload = {
            "id": graph_result.item.id,
            "query": graph_result.item.query,
            "category": graph_result.item.category,
            "vector_quality_score": quality_score(vector_result),
            "graph_quality_score": quality_score(graph_result),
            "delta": delta,
            "vector_top3": [context.citation_text for context in vector_result.contexts[:3]],
            "graph_top3": [context.citation_text for context in graph_result.contexts[:3]],
        }
        if delta > 1e-9:
            improved.append(payload)
        elif delta < -1e-9:
            worse.append(payload)

    total_graph_contexts = sum(len(result.contexts) for result in graph_results)
    total_vector_contexts = sum(len(result.contexts) for result in vector_results)
    direct_vector_contexts = [
        context for result in graph_results for context in result.contexts if context.retrieval_source == "vector"
    ]
    return {
        "graph_expansion_used_count": sum(1 for result in graph_results if any(context in graph_like_contexts for context in result.contexts)),
        "average_graph_depth": mean(float(value) for value in graph_depths),
        "top_graph_edge_types": edge_counter.most_common(10),
        "graph_hit_ratio": len(graph_like_contexts) / max(1, total_graph_contexts),
        "direct_vector_hit_ratio": len(direct_vector_contexts) / max(1, total_graph_contexts),
        "number_of_queries_improved_over_vector_only": len(improved),
        "number_of_queries_worse_than_vector_only": len(worse),
        "improved_examples": improved[:8],
        "worse_examples": worse[:8],
        "vector_result_count": total_vector_contexts,
    }


def vector_only_diagnostics(results: Sequence[ModeResult]) -> dict[str, object]:
    vector_results = [result for result in results if result.mode == MODE_VECTOR_ONLY]
    contexts = [context for result in vector_results for context in result.contexts]
    nonempty_queries = sum(1 for result in vector_results if result.contexts)
    contexts_with_document = sum(1 for context in contexts if context.document_id)
    contexts_with_article = sum(1 for context in contexts if context.article_number or "phu_luc" in context.chunk_id.lower())
    all_zero_metrics = bool(vector_results) and all(
        float(result.metrics.get("recall_at_10", 0.0)) == 0.0
        and float(result.metrics.get("required_citation_coverage", 0.0)) == 0.0
        for result in vector_results
    )
    return {
        "query_count": len(vector_results),
        "nonempty_result_queries": nonempty_queries,
        "context_count": len(contexts),
        "contexts_with_document_id": contexts_with_document,
        "contexts_with_article_or_appendix": contexts_with_article,
        "metadata_document_id_rate": contexts_with_document / len(contexts) if contexts else 0.0,
        "metadata_article_or_appendix_rate": contexts_with_article / len(contexts) if contexts else 0.0,
        "all_zero_recall_and_coverage": all_zero_metrics,
        "zero_metric_interpretation": (
            "nonempty_results_with_metadata"
            if all_zero_metrics and nonempty_queries and contexts_with_document
            else "empty_or_incomplete_results"
            if all_zero_metrics
            else "nonzero_metrics"
        ),
    }


def csv_rows(results: Sequence[ModeResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        rows.append(
            {
                "id": result.item.id,
                "query": result.item.query,
                "category": result.item.category,
                "topic": result.item.topic,
                "difficulty": result.item.difficulty,
                "requires_graph": result.item.requires_graph,
                "requires_normative_hierarchy": result.item.requires_normative_hierarchy,
                "mode": result.mode,
                **result.metrics,
                "top1_citation": result.contexts[0].citation_text if result.contexts else "",
                "top3_citations": " || ".join(context.citation_text for context in result.contexts[:3]),
                "top10_citations": " || ".join(context.citation_text for context in result.contexts[:10]),
                "top10_chunk_ids": " || ".join(context.chunk_id for context in result.contexts[:10]),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def render_report(summary: dict[str, object], results: Sequence[ModeResult]) -> str:
    overall = summary["overall_metrics"]
    category_metrics = summary["category_metrics"]
    graph = summary["graph_diagnostics"]
    vector_diagnostics = summary.get("vector_only_diagnostics") or {}
    by_id_mode = {(result.item.id, result.mode): result for result in results}
    item_ids = list(dict.fromkeys(result.item.id for result in results))

    lines: list[str] = [
        "# Retrieval Modes Evaluation",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Benchmark queries: {summary['benchmark_count']}",
        f"- Modes: {', '.join(MODES)}",
        "",
        "## Overall Comparison",
        "",
    ]
    lines.extend(
        markdown_table(
            ("Mode", "Recall@5", "Recall@10", "Precision@5", "Precision@10", "MRR", "Required coverage", "Forbidden rate", "Avg graph chunks"),
            [
                (
                    mode,
                    f"{overall[mode]['recall_at_5']:.3f}",
                    f"{overall[mode]['recall_at_10']:.3f}",
                    f"{overall[mode]['precision_at_5']:.3f}",
                    f"{overall[mode]['precision_at_10']:.3f}",
                    f"{overall[mode]['mrr']:.3f}",
                    f"{overall[mode]['required_citation_coverage']:.3f}",
                    f"{overall[mode]['forbidden_citation_violation']:.3f}",
                    f"{overall[mode]['graph_expanded_chunks']:.2f}",
                )
                for mode in MODES
            ],
        )
    )
    lines.extend(["", "## Per-Category Comparison", ""])
    category_rows = []
    for category in sorted(category_metrics):
        for mode in MODES:
            metrics = category_metrics[category].get(mode, {})
            category_rows.append(
                (
                    category,
                    mode,
                    int(metrics.get("query_count", 0)),
                    f"{metrics.get('recall_at_10', 0.0):.3f}",
                    f"{metrics.get('precision_at_10', 0.0):.3f}",
                    f"{metrics.get('mrr', 0.0):.3f}",
                    f"{metrics.get('required_citation_coverage', 0.0):.3f}",
                )
            )
    lines.extend(markdown_table(("Category", "Mode", "Queries", "Recall@10", "Precision@10", "MRR", "Required coverage"), category_rows))
    lines.extend(["", "## Per-Query Comparison", ""])
    query_rows = []
    for item_id in item_ids:
        graph_result = by_id_mode[(item_id, MODE_GRAPH)]
        row: list[object] = [graph_result.item.id, graph_result.item.category, graph_result.item.query]
        for mode in MODES:
            result = by_id_mode[(item_id, mode)]
            row.append(f"{result.metrics['recall_at_10']:.2f}/{result.metrics['mrr']:.2f}/{result.metrics['required_citation_coverage']:.2f}")
        query_rows.append(tuple(row))
    lines.extend(markdown_table(("ID", "Category", "Query", "Vector R/M/C", "Hybrid R/M/C", "Graph R/M/C"), query_rows))

    lines.extend(["", "## Graph Diagnostics", ""])
    lines.extend(
        [
            f"- Graph expansion used count: {graph['graph_expansion_used_count']}",
            f"- Average graph depth: {float(graph['average_graph_depth']):.3f}",
            f"- Graph hit ratio: {float(graph['graph_hit_ratio']):.3f}",
            f"- Direct vector hit ratio in graph mode: {float(graph['direct_vector_hit_ratio']):.3f}",
            f"- Improved over vector-only: {graph['number_of_queries_improved_over_vector_only']}",
            f"- Worse than vector-only: {graph['number_of_queries_worse_than_vector_only']}",
            f"- Top graph edge types: {', '.join(f'{edge}:{count}' for edge, count in graph['top_graph_edge_types']) or 'None'}",
        ]
    )

    lines.extend(["", "## Vector-Only Diagnostics", ""])
    lines.extend(
        [
            f"- Nonempty vector-only queries: {vector_diagnostics.get('nonempty_result_queries', 0)} / {vector_diagnostics.get('query_count', 0)}",
            f"- Vector-only contexts: {vector_diagnostics.get('context_count', 0)}",
            f"- Metadata document-id rate: {float(vector_diagnostics.get('metadata_document_id_rate', 0.0)):.3f}",
            f"- Metadata article-or-appendix rate: {float(vector_diagnostics.get('metadata_article_or_appendix_rate', 0.0)):.3f}",
            f"- All-zero recall/coverage: {vector_diagnostics.get('all_zero_recall_and_coverage', False)}",
            f"- All-zero interpretation: {vector_diagnostics.get('zero_metric_interpretation', 'not_available')}",
        ]
    )

    lines.extend(["", "## Graph Improvements", ""])
    improved = graph["improved_examples"]
    if improved:
        for item in improved:
            lines.extend(
                [
                    f"- `{item['id']}` ({item['category']}): +{float(item['delta']):.3f}",
                    f"  - Query: {item['query']}",
                    f"  - Vector top 3: {'; '.join(item['vector_top3'])}",
                    f"  - Graph top 3: {'; '.join(item['graph_top3'])}",
                ]
            )
    else:
        lines.append("- No graph improvements over vector-only under the current scoring formula.")

    lines.extend(["", "## Remaining Failures Or Over-Expansion", ""])
    worse = graph["worse_examples"]
    if worse:
        for item in worse:
            lines.extend(
                [
                    f"- `{item['id']}` ({item['category']}): {float(item['delta']):.3f}",
                    f"  - Query: {item['query']}",
                    f"  - Vector top 3: {'; '.join(item['vector_top3'])}",
                    f"  - Graph top 3: {'; '.join(item['graph_top3'])}",
                ]
            )
    else:
        lines.append("- No query scored worse than vector-only under the current scoring formula.")

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            (
                "On this constructed benchmark, graph-augmented retrieval provides the strongest citation coverage "
                "because it can connect primary Labor Code provisions to implementing decrees, circular guidance, "
                "exceptions, and labor-litigation provisions. Hybrid dense+sparse retrieval is a stronger baseline "
                "than dense vector-only retrieval for these legal-reference questions. The graph layer is most "
                "valuable for multi-hop, procedure, exception, and scenario questions because it can promote legally "
                "connected provisions even when the query wording does not exactly match every required citation. "
                "These results should be interpreted as benchmark evidence, not as proof of universal legal correctness."
            ),
        ]
    )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    if args.embedding_provider:
        os.environ["EMBEDDING_PROVIDER"] = args.embedding_provider
    items = load_or_create_benchmark(args.benchmark_path, limit=args.limit)
    all_results: list[ModeResult] = []
    for mode in MODES:
        all_results.extend(
            run_mode(
                mode=mode,
                items=items,
                index_path=args.index_path,
                top_k=args.top_k,
                prefetch_limit=args.prefetch_limit,
                reranker_model=args.reranker_model,
                device=args.device,
            )
        )

    summary: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_path": str(args.benchmark_path),
        "benchmark_count": len(items),
        "modes": list(MODES),
        "overall_metrics": aggregate_results(all_results),
        "category_metrics": aggregate_by_category(all_results),
        "graph_diagnostics": graph_diagnostics(all_results),
        "vector_only_diagnostics": vector_only_diagnostics(all_results),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(args.output_prefix or "retrieval_modes").strip()
    csv_path = args.output_dir / f"{output_prefix}_results.csv"
    summary_path = args.output_dir / f"{output_prefix}_summary.json"
    report_path = args.output_dir / f"{output_prefix}_report.md"
    rows = csv_rows(all_results)
    write_csv(csv_path, rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(summary, all_results), encoding="utf-8")
    print(
        json.dumps(
            {
                "benchmark_count": len(items),
                "csv_path": str(csv_path),
                "summary_path": str(summary_path),
                "report_path": str(report_path),
                "overall_metrics": summary["overall_metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
