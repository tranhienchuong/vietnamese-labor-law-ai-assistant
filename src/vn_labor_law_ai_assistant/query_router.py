from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from typing import Any, Callable, Literal, Mapping, Sequence, get_args

from pydantic import BaseModel, ConfigDict, Field

from .corpus_pipeline import normalize_for_matching
from .heuristic_router import QueryIntent


ActorLabel = Literal[
    "nguoi_lao_dong",
    "nguoi_su_dung_lao_dong",
    "lao_dong_nu",
    "lao_dong_chua_thanh_nien",
    "nguoi_cao_tuoi",
    "nguoi_lao_dong_nuoc_ngoai",
    "to_chuc_dai_dien_nguoi_lao_dong",
]

TopicLabel = Literal[
    "cham_dut_hop_dong_lao_dong",
    "don_phuong_cham_dut",
    "bao_truoc",
    "tro_cap",
    "ky_luat_sa_thai",
    "thay_doi_co_cau_kinh_te",
    "tam_hoan_hop_dong",
    "hop_dong_lao_dong",
    "dao_tao_nghe",
    "bao_ve_thai_san",
    "tien_luong",
    "thu_viec",
    "bao_hiem",
]

IssueLabel = Literal[
    "can_cu_cham_dut",
    "quyen_don_phuong_cham_dut",
    "thoi_han_bao_truoc",
    "tro_cap_thoi_viec",
    "tro_cap_mat_viec",
    "nghia_vu_khi_cham_dut",
    "trai_phap_luat",
    "boi_thuong",
    "sa_thai",
    "noi_quy_lao_dong",
    "thong_bao_cham_dut",
    "tam_hoan_hop_dong",
    "dao_tao",
    "thay_doi_co_cau_kinh_te",
    "bao_ve_thai_san",
    "giao_ket_hop_dong",
    "sua_doi_bo_sung_hop_dong",
    "thong_tin_giao_ket",
    "loai_hop_dong",
    "dieu_chuyen_cong_viec",
    "doi_thoai_tai_noi_lam_viec",
    "xu_ly_ky_luat_lao_dong",
    "tien_luong",
    "thu_viec",
    "bao_hiem_xa_hoi",
]

DocumentLabel = Literal[
    "45-2019-qh14",
    "nghi-dinh-145-2020-nd-cp",
]

QueryTypeLabel = Literal[
    "yes_no",
    "time_limit",
    "money_percentage",
    "procedure",
    "remedy",
    "definition",
    "classification",
    "enumeration",
    "missing_fact",
]

VALID_ACTORS = get_args(ActorLabel)
VALID_TOPICS = get_args(TopicLabel)
VALID_ISSUES = get_args(IssueLabel)
VALID_DOCUMENTS = get_args(DocumentLabel)
VALID_QUERY_TYPES = get_args(QueryTypeLabel)

LABEL_ALIASES = {
    "cham_dut_hop_dong": "cham_dut_hop_dong_lao_dong",
    "cham_dut_hdld": "cham_dut_hop_dong_lao_dong",
    "thai_san": "bao_ve_thai_san",
    "ky_luat": "ky_luat_sa_thai",
    "dao_tao": "dao_tao_nghe",
    "bhxh": "bao_hiem_xa_hoi",
    "45_2019_qh14": "45-2019-qh14",
    "nghi_dinh_145_2020_nd_cp": "nghi-dinh-145-2020-nd-cp",
}

ACTOR_DESCRIPTIONS = {
    "nguoi_lao_dong": "employee or worker asking about their own rights or duties",
    "nguoi_su_dung_lao_dong": "employer, company, manager, or HR asking about employer rights or duties",
    "lao_dong_nu": "female worker, pregnant worker, maternity, or worker raising a child under 12 months",
    "lao_dong_chua_thanh_nien": "minor or underage worker",
    "nguoi_cao_tuoi": "older worker or retirement-age worker",
    "nguoi_lao_dong_nuoc_ngoai": "foreign worker in Vietnam",
    "to_chuc_dai_dien_nguoi_lao_dong": "employee representative organization or trade union context",
}

TOPIC_DESCRIPTIONS = {
    "cham_dut_hop_dong_lao_dong": "termination or ending of an employment contract",
    "don_phuong_cham_dut": "unilateral termination, resigning, quitting, or one party ending the contract",
    "bao_truoc": "advance notice period or no-notice termination",
    "tro_cap": "severance allowance or job-loss allowance",
    "ky_luat_sa_thai": "labor discipline, dismissal, or firing as discipline",
    "thay_doi_co_cau_kinh_te": "structural change, technology change, merger, split, or economic reason",
    "tam_hoan_hop_dong": "suspension or temporary pause of employment contract performance",
    "hop_dong_lao_dong": "employment contract formation, content, type, amendment, or transfer of work",
    "dao_tao_nghe": "vocational training or training cost agreement",
    "bao_ve_thai_san": "maternity protection, pregnancy, childbirth leave, or child under 12 months",
    "tien_luong": "salary, wage payment, late salary, overtime pay, bonus, or deduction",
    "thu_viec": "probation or probationary employment",
    "bao_hiem": "social insurance or unemployment insurance context",
}

ISSUE_DESCRIPTIONS = {
    "can_cu_cham_dut": "legal grounds or permitted cases for contract termination",
    "quyen_don_phuong_cham_dut": "right to unilaterally terminate or resign",
    "thoi_han_bao_truoc": "how many days of notice are required",
    "tro_cap_thoi_viec": "severance allowance calculation or eligibility",
    "tro_cap_mat_viec": "job-loss allowance calculation or eligibility",
    "nghia_vu_khi_cham_dut": "final payment, returning documents, social insurance confirmation after termination",
    "trai_phap_luat": "illegal or unlawful termination",
    "boi_thuong": "compensation, damages, reimbursement, or indemnity",
    "sa_thai": "dismissal as a disciplinary measure",
    "noi_quy_lao_dong": "internal labor rules",
    "thong_bao_cham_dut": "written notice or notification of termination",
    "tam_hoan_hop_dong": "temporary suspension of employment contract",
    "dao_tao": "training contract or training cost reimbursement",
    "thay_doi_co_cau_kinh_te": "workforce changes due to structure, technology, or economic reason",
    "bao_ve_thai_san": "maternity protection issue",
    "giao_ket_hop_dong": "entering into an employment contract",
    "sua_doi_bo_sung_hop_dong": "amending or supplementing an employment contract",
    "thong_tin_giao_ket": "information duties before entering into a contract",
    "loai_hop_dong": "fixed-term, indefinite-term, or other contract type",
    "dieu_chuyen_cong_viec": "temporarily assigning different work than the contract",
    "doi_thoai_tai_noi_lam_viec": "workplace dialogue",
    "xu_ly_ky_luat_lao_dong": "disciplinary process, limitation period, meeting, or suspension",
    "tien_luong": "salary, wage payment, late payment, or deduction issue",
    "thu_viec": "probation issue",
    "bao_hiem_xa_hoi": "social insurance or unemployment insurance issue",
}

DOCUMENT_DESCRIPTIONS = {
    "45-2019-qh14": "Vietnam Labor Code 2019",
    "nghi-dinh-145-2020-nd-cp": "Decree 145/2020/ND-CP implementation guidance",
}

QUERY_TYPE_DESCRIPTIONS = {
    "yes_no": "asks whether something is allowed, required, lawful, or unlawful",
    "time_limit": "asks about deadlines, periods, number of days, or duration",
    "money_percentage": "asks about money, salary, allowance, compensation, percentage, or calculation",
    "procedure": "asks about process, paperwork, steps, meeting, or authority",
    "remedy": "asks what to claim, how to fix a violation, reinstatement, or compensation",
    "definition": "asks what a legal term means",
    "classification": "asks which legal category or case applies",
    "enumeration": "asks to list all cases or conditions",
    "missing_fact": "contains conditional facts that may require more information",
}

DEFAULT_ROUTER_PROVIDER = "groq"
DEFAULT_ROUTER_MODEL = "llama-3.1-8b-instant"
DEFAULT_ROUTER_MODEL_ENV = "QUERY_ROUTER_MODEL"
DEFAULT_ROUTER_PROVIDER_ENV = "QUERY_ROUTER_PROVIDER"
JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


class QueryMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: ActorLabel | None = Field(
        description=(
            "Primary legal actor in the user's question. Use null if there is no clear "
            "single primary actor."
        )
    )
    actors: list[ActorLabel] = Field(
        description=(
            "All legal actors that matter for retrieval. Include the primary actor too "
            "when known. Use [] if unclear."
        )
    )
    topics: list[TopicLabel] = Field(
        description="Main legal topics. Use only valid topic labels. Use [] if none fit."
    )
    issues: list[IssueLabel] = Field(
        description="More specific legal issues. Use only valid issue labels. Use [] if none fit."
    )
    document_ids: list[DocumentLabel] = Field(
        description="Explicitly requested legal documents only. Use [] if no document is mentioned."
    )
    query_types: list[QueryTypeLabel] = Field(
        description="Question shape labels useful for retrieval and reranking. Use [] if unclear."
    )
    article_numbers: list[str] = Field(
        description="Explicit article numbers mentioned by the user, e.g. ['35']. Use [] if none."
    )
    clause_refs: list[str] = Field(
        description="Explicit clause numbers mentioned by the user, e.g. ['2']. Use [] if none."
    )
    point_refs: list[str] = Field(
        description="Explicit point references mentioned by the user, e.g. ['a'] or ['b.1']. Use [] if none."
    )


def _normalize_text(value: str) -> str:
    value = value.replace("đ", "d").replace("Đ", "D")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _coerce_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return list(value)
    return []


def _normalize_label(value: object) -> str:
    normalized = _normalize_text(str(value or ""))
    normalized = normalized.replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"[^a-z0-9_.]+", "_", normalized).strip("_")
    return LABEL_ALIASES.get(normalized, normalized)


def _clean_labels(value: object, valid_labels: Sequence[str]) -> list[str]:
    valid = set(valid_labels)
    labels = [
        normalized
        for item in _coerce_list(value)
        if (normalized := _normalize_label(item)) in valid
    ]
    return _dedupe_preserve_order(labels)


def _clean_refs(value: object, pattern: re.Pattern[str]) -> list[str]:
    refs: list[str] = []
    for item in _coerce_list(value):
        normalized = _normalize_text(str(item or ""))
        if match := pattern.search(normalized):
            refs.append(match.group("value").lower())
            continue
        if pattern.fullmatch(normalized):
            refs.append(normalized.lower())
    return _dedupe_preserve_order(refs)


def sanitize_query_metadata_payload(payload: Mapping[str, object]) -> dict[str, object]:
    actor = _clean_labels(payload.get("actor"), VALID_ACTORS)
    actors = _clean_labels(payload.get("actors"), VALID_ACTORS)
    if actor:
        actors = _dedupe_preserve_order([actor[0], *actors])

    return {
        "actor": actor[0] if actor else (actors[0] if actors else None),
        "actors": actors,
        "topics": _clean_labels(payload.get("topics"), VALID_TOPICS),
        "issues": _clean_labels(payload.get("issues"), VALID_ISSUES),
        "document_ids": _clean_labels(payload.get("document_ids"), VALID_DOCUMENTS),
        "query_types": _clean_labels(payload.get("query_types"), VALID_QUERY_TYPES),
        "article_numbers": _clean_refs(
            payload.get("article_numbers"),
            re.compile(r"(?:dieu\s*)?(?P<value>\d+[a-z]?)"),
        ),
        "clause_refs": _clean_refs(
            payload.get("clause_refs"),
            re.compile(r"(?:khoan\s*)?(?P<value>\d+)"),
        ),
        "point_refs": _clean_refs(
            payload.get("point_refs"),
            re.compile(r"(?:diem\s*)?(?P<value>[a-z](?:\.\d+)?)"),
        ),
    }


def parse_query_metadata(content: str) -> QueryMetadata:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        raw_payload = json.loads(text)
    except json.JSONDecodeError:
        match = JSON_OBJECT_RE.search(text)
        if match is None:
            raise
        raw_payload = json.loads(match.group(0))

    if not isinstance(raw_payload, Mapping):
        raise ValueError("Query router response must be a JSON object.")

    payload = sanitize_query_metadata_payload(raw_payload)
    return QueryMetadata.model_validate(payload)


def query_metadata_json_schema() -> dict[str, Any]:
    schema = QueryMetadata.model_json_schema()
    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        schema["required"] = list(properties)
    schema["additionalProperties"] = False
    return schema


def _format_allowed_values(descriptions: Mapping[str, str]) -> str:
    return "\n".join(f"- {label}: {description}" for label, description in descriptions.items())


def build_query_router_messages(query: str) -> list[dict[str, str]]:
    system_prompt = f"""You are a query router for a Vietnamese labor-law RAG system.

Return exactly one JSON object matching the provided schema. Do not explain.

Routing rules:
- Classify the user's legal intent, not isolated keywords.
- Do not mark maternity just because the words appear in a side example. Mark it only when the legal question is about pregnancy, childbirth leave, maternity protection, or a worker raising a child under 12 months.
- Do not invent labels. Use only the valid labels below.
- For explicit legal references, copy only numbers/letters the user mentioned. Do not infer article numbers.
- Article numbers come after "Dieu"; clause refs come after "khoan"; point refs come after "diem". Never put a clause number into article_numbers.
- If uncertain, prefer null or [] rather than a noisy tag.
- Prefer the most specific actor over a generic actor. For example, use lao_dong_nu for pregnant or maternity questions.
- Use canonical tag bao_ve_thai_san for maternity; do not output thai_san.

Few-shot routing examples:
Input: "Ba bau tu y nghi ngang thi co den tien khong?"
Output: {{"actor":"lao_dong_nu","actors":["lao_dong_nu","nguoi_lao_dong"],"topics":["bao_ve_thai_san","cham_dut_hop_dong_lao_dong","don_phuong_cham_dut"],"issues":["quyen_don_phuong_cham_dut","boi_thuong"],"document_ids":[],"query_types":["yes_no","remedy"],"article_numbers":[],"clause_refs":[],"point_refs":[]}}

Input: "Cong ty cam nhan vien noi chuyen ve phu nu mang thai co dung luat khong?"
Output: {{"actor":"nguoi_lao_dong","actors":["nguoi_lao_dong","nguoi_su_dung_lao_dong"],"topics":[],"issues":[],"document_ids":[],"query_types":["yes_no"],"article_numbers":[],"clause_refs":[],"point_refs":[]}}

Input: "Cong ty no luong 2 thang toi nghi luon duoc khong?"
Output: {{"actor":"nguoi_lao_dong","actors":["nguoi_lao_dong","nguoi_su_dung_lao_dong"],"topics":["tien_luong","cham_dut_hop_dong_lao_dong","don_phuong_cham_dut"],"issues":["tien_luong","quyen_don_phuong_cham_dut"],"document_ids":[],"query_types":["yes_no"],"article_numbers":[],"clause_refs":[],"point_refs":[]}}

Input: "Dieu 35 khoan 2 quy dinh gi?"
Output: {{"actor":null,"actors":[],"topics":[],"issues":[],"document_ids":[],"query_types":["definition"],"article_numbers":["35"],"clause_refs":["2"],"point_refs":[]}}

Valid actors:
{_format_allowed_values(ACTOR_DESCRIPTIONS)}

Valid topics:
{_format_allowed_values(TOPIC_DESCRIPTIONS)}

Valid issues:
{_format_allowed_values(ISSUE_DESCRIPTIONS)}

Valid documents:
{_format_allowed_values(DOCUMENT_DESCRIPTIONS)}

Valid query types:
{_format_allowed_values(QUERY_TYPE_DESCRIPTIONS)}
"""
    user_prompt = f'Classify this user question for retrieval:\n"{query}"'
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


CompletionFn = Callable[..., Any]


def analyze_query_smart(
    query: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    completion_fn: CompletionFn | None = None,
) -> QueryMetadata:
    if completion_fn is None:
        from .llm import chat_completion

        completion_fn = chat_completion

    response = completion_fn(
        provider=provider or os.getenv(DEFAULT_ROUTER_PROVIDER_ENV, DEFAULT_ROUTER_PROVIDER),
        model=model if model is not None else os.getenv(DEFAULT_ROUTER_MODEL_ENV, DEFAULT_ROUTER_MODEL),
        messages=build_query_router_messages(query),
        temperature=0,
        json_schema=query_metadata_json_schema(),
        json_schema_name="query_metadata",
    )
    content = str(getattr(response, "content", response) or "")
    return parse_query_metadata(content)


def query_intent_from_metadata(query: str, metadata: QueryMetadata) -> QueryIntent:
    normalized_query = normalize_for_matching(f" {query} ")
    actors = tuple(value for value in (metadata.actor, *metadata.actors) if value)
    return QueryIntent(
        raw_query=query,
        normalized_query=normalized_query,
        actor_filters=tuple(_dedupe_preserve_order(actors)),
        topic_filters=tuple(_dedupe_preserve_order(tuple(metadata.topics))),
        issue_filters=tuple(_dedupe_preserve_order(tuple(metadata.issues))),
        document_filters=tuple(_dedupe_preserve_order(tuple(metadata.document_ids))),
        article_numbers=tuple(_dedupe_preserve_order(tuple(metadata.article_numbers))),
        inferred_article_numbers=(),
        clause_refs=tuple(_dedupe_preserve_order(tuple(metadata.clause_refs))),
        point_refs=tuple(_dedupe_preserve_order(tuple(metadata.point_refs))),
        query_expansions=(),
        query_types=tuple(_dedupe_preserve_order(tuple(metadata.query_types))),
    )


def route_query_with_llm(
    query: str,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> QueryIntent:
    metadata = analyze_query_smart(query, provider=provider, model=model)
    return query_intent_from_metadata(query, metadata)


def metadata_to_retrieval_filters(metadata: QueryMetadata) -> dict[str, list[str]]:
    filters: dict[str, list[str]] = {}
    actors = _dedupe_preserve_order(
        [value for value in [metadata.actor, *metadata.actors] if value]
    )
    if actors:
        filters["actor"] = actors
    if metadata.topics:
        filters["topic"] = list(metadata.topics)
    if metadata.issues:
        filters["issue_type"] = list(metadata.issues)
    if metadata.document_ids:
        filters["document_id"] = list(metadata.document_ids)
    if metadata.article_numbers:
        filters["article_number"] = list(metadata.article_numbers)
    if metadata.clause_refs:
        filters["clause_ref"] = list(metadata.clause_refs)
    if metadata.point_refs:
        filters["point_ref"] = list(metadata.point_refs)
    return filters


def main() -> None:
    parser = argparse.ArgumentParser(description="Route a Vietnamese labor-law query with an LLM.")
    parser.add_argument("query", help="Question to classify.")
    parser.add_argument("--provider", default=None, help="LLM provider. Defaults to QUERY_ROUTER_PROVIDER or groq.")
    parser.add_argument("--model", default=None, help="Router model. Defaults to QUERY_ROUTER_MODEL.")
    args = parser.parse_args()

    metadata = analyze_query_smart(args.query, provider=args.provider, model=args.model)
    print(json.dumps(metadata.model_dump(), ensure_ascii=False, indent=2))
    print(json.dumps(metadata_to_retrieval_filters(metadata), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "ACTOR_DESCRIPTIONS",
    "DEFAULT_ROUTER_MODEL",
    "DEFAULT_ROUTER_PROVIDER",
    "DOCUMENT_DESCRIPTIONS",
    "ISSUE_DESCRIPTIONS",
    "QUERY_TYPE_DESCRIPTIONS",
    "TOPIC_DESCRIPTIONS",
    "VALID_ACTORS",
    "VALID_DOCUMENTS",
    "VALID_ISSUES",
    "VALID_QUERY_TYPES",
    "VALID_TOPICS",
    "QueryMetadata",
    "analyze_query_smart",
    "build_query_router_messages",
    "metadata_to_retrieval_filters",
    "parse_query_metadata",
    "query_intent_from_metadata",
    "query_metadata_json_schema",
    "route_query_with_llm",
    "sanitize_query_metadata_payload",
]
