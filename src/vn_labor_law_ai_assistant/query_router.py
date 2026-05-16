from __future__ import annotations

import argparse
import json
import re
import unicodedata
from typing import Any, Callable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .core.config import load_settings
from .corpus_pipeline import normalize_for_matching
from .heuristic_router import (
    QueryIntent,
    collect_keyword_matches,
    collect_mapped_article_expansions,
    collect_rule_based_routing,
    dedupe_preserve_order,
)
from .rule_loader import DEFAULT_RULE_CONFIG


RULE_CONFIG = DEFAULT_RULE_CONFIG


ActorLabel = str
TopicLabel = str
IssueLabel = str
DocumentLabel = str
QueryTypeLabel = str

VALID_ACTORS = RULE_CONFIG.VALID_ACTORS
VALID_TOPICS = RULE_CONFIG.VALID_TOPICS
VALID_ISSUES = RULE_CONFIG.VALID_ISSUES
VALID_DOCUMENTS = RULE_CONFIG.VALID_DOCUMENTS
VALID_QUERY_TYPES = RULE_CONFIG.VALID_QUERY_TYPES

LABEL_ALIASES = RULE_CONFIG.LABEL_ALIASES
ACTOR_DESCRIPTIONS = RULE_CONFIG.ACTOR_DESCRIPTIONS
TOPIC_DESCRIPTIONS = RULE_CONFIG.TOPIC_DESCRIPTIONS
ISSUE_DESCRIPTIONS = RULE_CONFIG.ISSUE_DESCRIPTIONS
DOCUMENT_DESCRIPTIONS = RULE_CONFIG.DOCUMENT_DESCRIPTIONS
QUERY_TYPE_DESCRIPTIONS = RULE_CONFIG.QUERY_TYPE_DESCRIPTIONS

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
        description="Explicit article numbers mentioned by the user. Use [] if none."
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


def _format_few_shot_examples(examples: Sequence[Mapping[str, object]]) -> str:
    parts: list[str] = []
    for example in examples:
        parts.append(
            "Input: "
            f"\"{example.get('input', '')}\"\n"
            "Output: "
            f"{json.dumps(example.get('output', {}), ensure_ascii=False, separators=(',', ':'))}"
        )
    return "\n\n".join(parts)


def build_query_router_messages(query: str) -> list[dict[str, str]]:
    prompt_config = RULE_CONFIG.ROUTER_PROMPT
    routing_rules = "\n".join(f"- {rule}" for rule in prompt_config.get("routing_rules", ()))
    few_shot_examples = _format_few_shot_examples(prompt_config.get("few_shot_examples", ()))
    system_prompt = f"""You are a query router for a Vietnamese labor-law RAG system.

Return exactly one JSON object matching the provided schema. Do not explain.

Routing rules:
{routing_rules}

Few-shot routing examples:
{few_shot_examples}

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

    settings = load_settings()
    provider_name = (
        settings.query_router_provider
        if settings.field_was_configured("query_router_provider")
        else DEFAULT_ROUTER_PROVIDER
    )
    model_name = (
        settings.query_router_model
        if settings.field_was_configured("query_router_model")
        else DEFAULT_ROUTER_MODEL
    )
    response = completion_fn(
        provider=provider or provider_name,
        model=model if model is not None else model_name,
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
    document_filters = dedupe_preserve_order(
        (
            *metadata.document_ids,
            *collect_keyword_matches(normalized_query, RULE_CONFIG.DOCUMENT_KEYWORDS),
        )
    )
    query_types = dedupe_preserve_order(
        (
            *metadata.query_types,
            *collect_keyword_matches(normalized_query, RULE_CONFIG.QUERY_TYPE_KEYWORDS),
        )
    )
    rule_routing = collect_rule_based_routing(
        normalized_query,
        RULE_CONFIG,
        query_types=query_types,
        document_filters=document_filters,
    )
    topic_filters = dedupe_preserve_order((*metadata.topics, *rule_routing.topics))
    issue_filters = dedupe_preserve_order((*metadata.issues, *rule_routing.issues))
    mapped_articles, mapped_expansions = collect_mapped_article_expansions(
        topic_filters=topic_filters,
        issue_filters=issue_filters,
        rule_config=RULE_CONFIG,
    )
    article_numbers = dedupe_preserve_order(tuple(metadata.article_numbers))
    return QueryIntent(
        raw_query=query,
        normalized_query=normalized_query,
        actor_filters=dedupe_preserve_order(actors),
        topic_filters=topic_filters,
        issue_filters=issue_filters,
        document_filters=document_filters,
        article_numbers=article_numbers,
        inferred_article_numbers=dedupe_preserve_order(
            (*rule_routing.inferred_articles, *mapped_articles)
        ),
        force_reference_article_numbers=dedupe_preserve_order(
            (*article_numbers, *rule_routing.force_reference_articles)
        ),
        forced_references=rule_routing.forced_references,
        matched_direct_reference_rules=rule_routing.direct_reference_rule_names,
        clause_refs=dedupe_preserve_order(tuple(metadata.clause_refs)),
        point_refs=dedupe_preserve_order(tuple(metadata.point_refs)),
        query_expansions=dedupe_preserve_order((*rule_routing.expansions, *mapped_expansions)),
        query_types=query_types,
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
