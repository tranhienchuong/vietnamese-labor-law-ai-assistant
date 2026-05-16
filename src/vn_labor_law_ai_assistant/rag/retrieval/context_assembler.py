from __future__ import annotations

from typing import Any, Sequence

from ...heuristic_router import (
    QueryIntent,
    dedupe_preserve_order,
    infer_employee_notice_period_reference,
    query_asks_for_enumeration,
)
from .constants import RULE_CONFIG
from .models import RetrievedRecord, RetrievalContext, SearchHit
from .record_store import RecordStore
from .utils import (
    build_expanded_context_text,
    context_article_key,
    context_looks_like_enumeration_parent,
    extend_context_with_records,
)


class ContextAssembler:
    def __init__(
        self,
        *,
        record_store: RecordStore,
        rule_config: Any = RULE_CONFIG,
    ) -> None:
        self.record_store = record_store
        self.rule_config = rule_config

    def query_needs_article_siblings(self, intent: QueryIntent) -> bool:
        if query_asks_for_enumeration(intent):
            return True
        if intent.clause_refs or intent.point_refs:
            return True
        query_context = self.rule_config.QUERY_CONTEXT
        if set(intent.query_types).intersection(query_context.get("article_sibling_query_types", ())):
            return True
        return bool(
            set(intent.issue_filters).intersection(query_context.get("article_sibling_issues", ()))
        )

    def add_article_sibling_contexts(
        self,
        contexts: Sequence[RetrievalContext],
        *,
        intent: QueryIntent,
        direct_records: dict[str, RetrievedRecord],
    ) -> tuple[RetrievalContext, ...]:
        if not self.query_needs_article_siblings(intent):
            return tuple(contexts)

        seen_chunk_ids = {context.chunk_id for context in contexts}
        seen_chunk_ids.update(
            chunk_id
            for context in contexts
            for chunk_id in context.matched_chunk_ids
        )
        context_article_levels = {
            context_article_key(context): str(context.payload.get("level") or "")
            for context in contexts
            if context_article_key(context) is not None
        }
        records_by_article: dict[tuple[str, str], RetrievedRecord] = {}
        for record in direct_records.values():
            document_id = str(record.payload.get("document_id") or "")
            article_number = str(record.payload.get("article_number") or "")
            if not document_id or not article_number:
                continue
            key = (document_id, article_number)
            article_context_needs_children = any(
                context_article_key(context) == key
                and context_looks_like_enumeration_parent(context)
                for context in contexts
            )
            if (
                context_article_levels.get(key) == "article"
                and not query_asks_for_enumeration(intent)
                and not article_context_needs_children
            ):
                continue
            records_by_article.setdefault(key, record)

        if not records_by_article:
            return tuple(contexts)

        expanded_contexts: list[RetrievalContext] = []
        added_sibling_ids: set[str] = set()
        for context in contexts:
            key = context_article_key(context)
            if key not in records_by_article:
                expanded_contexts.append(context)
                continue
            document_id, article_number = key
            expanded_context = context
            merge_enumeration_records = (
                query_asks_for_enumeration(intent)
                or context_looks_like_enumeration_parent(context)
            )
            notice_clause_refs, notice_point_refs = infer_employee_notice_period_reference(intent)
            notice_article = self.rule_config.QUERY_CONTEXT.get(
                "employee_notice_period_reference",
                {},
            ).get("article")
            if article_number == notice_article and notice_clause_refs:
                notice_limit = (
                    4 if notice_point_refs else self.rule_config.MAX_ENUMERATION_CONTEXT_RECORDS
                )
                siblings = self.record_store.fetch_records_by_reference(
                    document_ids=(document_id,),
                    article_numbers=(article_number,),
                    clause_refs=notice_clause_refs,
                    point_refs=notice_point_refs,
                    exclude_chunk_ids=tuple(
                        (seen_chunk_ids if notice_point_refs else set()) | added_sibling_ids
                    ),
                    limit=notice_limit,
                )
                force_include_siblings = True
                if not notice_point_refs:
                    expanded_context = extend_context_with_records(
                        context,
                        siblings,
                        force_include=True,
                        replace_text=True,
                    )
                    added_sibling_ids.update(sibling.chunk_id for sibling in siblings)
                    expanded_contexts.append(expanded_context)
                    continue
            elif merge_enumeration_records:
                context_level = str(context.payload.get("level") or "")
                clause_ref = str(context.payload.get("clause_ref") or "")
                if context_level == "clause" and context_looks_like_enumeration_parent(context) and clause_ref:
                    siblings = self.record_store.fetch_records_by_reference(
                        document_ids=(document_id,),
                        article_numbers=(article_number,),
                        clause_refs=(clause_ref,),
                        exclude_chunk_ids=tuple(added_sibling_ids),
                        limit=self.rule_config.MAX_ENUMERATION_CONTEXT_RECORDS,
                    )
                else:
                    siblings = self.record_store.fetch_article_siblings(
                        document_id=document_id,
                        article_number=article_number,
                        exclude_chunk_ids=tuple(added_sibling_ids),
                        limit=self.rule_config.MAX_ENUMERATION_CONTEXT_RECORDS,
                    )
                expanded_context = extend_context_with_records(
                    context,
                    siblings,
                    force_include=True,
                    replace_text=True,
                )
                added_sibling_ids.update(sibling.chunk_id for sibling in siblings)
                expanded_contexts.append(expanded_context)
                continue
            else:
                siblings = self.record_store.fetch_article_siblings(
                    document_id=document_id,
                    article_number=article_number,
                    exclude_chunk_ids=tuple(seen_chunk_ids | added_sibling_ids),
                    limit=3,
                )
                force_include_siblings = False
            expanded_contexts.append(expanded_context)
            for rank, sibling in enumerate(siblings, start=1):
                added_sibling_ids.add(sibling.chunk_id)
                payload = dict(sibling.payload)
                if force_include_siblings:
                    payload["retrieval_force_include"] = True
                expanded_contexts.append(
                    RetrievalContext(
                        chunk_id=sibling.chunk_id,
                        citation_text=sibling.citation_text,
                        text=sibling.text,
                        payload=payload,
                        score=max(context.score - (rank * 1e-3), 0.0),
                        matched_chunk_ids=(sibling.chunk_id,),
                        matched_citations=(sibling.citation_text,),
                    )
                )

        return tuple(expanded_contexts)

    def assemble_contexts(
        self,
        hits: Sequence[SearchHit],
        intent: QueryIntent | None = None,
    ) -> tuple[RetrievalContext, ...]:
        direct_records = self.record_store.fetch_records_from_hits(hits)
        parent_ids = [
            record.parent_chunk_id
            for record in direct_records.values()
            if record.parent_chunk_id and record.parent_chunk_id not in direct_records
        ]
        parent_records = self.record_store.fetch_records(parent_ids)
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
                    "matched_records": [],
                },
            )
            group["score"] = max(float(group["score"]), hit.score)
            group["rank"] = min(int(group["rank"]), rank)
            group["matched_chunk_ids"].append(hit.chunk_id)
            group["matched_citations"].append(hit.citation_text)
            group["matched_records"].append(matched_record)

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
            matched_records = tuple(
                record
                for record in item["matched_records"]
                if isinstance(record, RetrievedRecord)
            )
            contexts.append(
                RetrievalContext(
                    chunk_id=record.chunk_id,
                    citation_text=record.citation_text,
                    text=build_expanded_context_text(record, matched_records),
                    payload=record.payload,
                    score=float(item["score"]),
                    matched_chunk_ids=matched_chunk_ids,
                    matched_citations=matched_citations,
                )
            )
        if intent is None:
            return tuple(contexts)
        return self.add_article_sibling_contexts(
            contexts,
            intent=intent,
            direct_records=direct_records,
        )


__all__ = ["ContextAssembler"]

