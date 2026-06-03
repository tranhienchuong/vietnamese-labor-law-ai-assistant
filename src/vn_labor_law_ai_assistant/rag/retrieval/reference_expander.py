from __future__ import annotations

from typing import Any, Sequence

from ...heuristic_router import LegalReference, QueryIntent
from .constants import FORCED_REFERENCE_SCORE_MARGIN, RULE_CONFIG
from .models import RetrievedRecord, SearchHit
from .record_store import RecordStore
from .utils import dedupe_records_by_chunk_id


class ReferenceExpander:
    def __init__(
        self,
        *,
        record_store: RecordStore,
        rule_config: Any = RULE_CONFIG,
    ) -> None:
        self.record_store = record_store
        self.rule_config = rule_config

    @staticmethod
    def record_to_search_hit(record: RetrievedRecord, score: float) -> SearchHit:
        return SearchHit(
            chunk_id=record.chunk_id,
            qdrant_point_id=str(record.payload.get("qdrant_point_id") or record.chunk_id),
            score=score,
            citation_text=record.citation_text,
            payload=record.payload,
        )

    @staticmethod
    def reference_label(reference: LegalReference) -> str:
        return ":".join(
            part
            for part in (
                reference.document_id,
                f"article={reference.article}" if reference.article else "",
                f"clause={reference.clause}" if reference.clause else "",
                f"point={reference.point}" if reference.point else "",
            )
            if part
        )

    @staticmethod
    def payload_matches_reference(payload: dict[str, object], reference: LegalReference) -> bool:
        if reference.document_id and str(payload.get("document_id") or "") != reference.document_id:
            return False
        if reference.article and str(payload.get("article_number") or "") != reference.article:
            return False
        if reference.clause and str(payload.get("clause_ref") or "") != reference.clause:
            return False
        if reference.point:
            payload_point_refs = {str(value) for value in payload.get("point_refs") or []}
            payload_point_ref = str(payload.get("point_ref") or "")
            if payload_point_ref != reference.point and reference.point not in payload_point_refs:
                return False
        return bool(reference.article)

    def hit_matches_forced_reference(self, hit: SearchHit, intent: QueryIntent) -> bool:
        return any(
            self.payload_matches_reference(hit.payload, reference)
            for reference in intent.forced_references
        )

    def forced_reference_records(
        self,
        intent: QueryIntent,
        *,
        limit: int,
    ) -> tuple[tuple[LegalReference, RetrievedRecord], ...]:
        if not intent.forced_references:
            return ()

        records: list[tuple[LegalReference, RetrievedRecord]] = []
        excluded_chunk_ids: set[str] = set()
        per_reference_limit = max(1, int(limit) // max(1, len(intent.forced_references)))
        for reference in intent.forced_references:
            if not reference.article:
                continue
            document_ids = (reference.document_id,) if reference.document_id else intent.document_filters
            clause_refs = (reference.clause,) if reference.clause else ()
            point_refs = (reference.point,) if reference.point else ()
            fetch_limit = 1 if clause_refs or point_refs else max(1, min(4, per_reference_limit))
            if (
                reference.document_id == "92-2015-qh13-labor-only"
                or (reference.document_id == "45-2019-qh14" and reference.article == "188")
            ):
                fetch_limit = 1
            fetched_records = self.record_store.fetch_records_by_reference(
                document_ids=document_ids,
                article_numbers=(reference.article,),
                clause_refs=clause_refs,
                point_refs=point_refs,
                exclude_chunk_ids=tuple(excluded_chunk_ids),
                limit=fetch_limit,
            )
            for record in fetched_records:
                records.append((reference, record))
                excluded_chunk_ids.add(record.chunk_id)
        return tuple(records)

    def append_forced_reference_hits(
        self,
        hits: Sequence[SearchHit],
        intent: QueryIntent,
        *,
        limit: int,
    ) -> tuple[SearchHit, ...]:
        forced_records = self.forced_reference_records(intent, limit=limit)
        if not forced_records:
            return tuple(hits)

        max_existing_score = max((hit.score for hit in hits), default=0.0)
        forced_hits: list[SearchHit] = []
        seen_forced_chunk_ids: set[str] = set()
        for rank, (reference, record) in enumerate(forced_records):
            if record.chunk_id in seen_forced_chunk_ids:
                continue
            seen_forced_chunk_ids.add(record.chunk_id)
            payload = dict(record.payload)
            payload["retrieval_forced_reference"] = True
            payload["retrieval_forced_reference_label"] = self.reference_label(reference)
            forced_hits.append(
                SearchHit(
                    chunk_id=record.chunk_id,
                    qdrant_point_id=str(record.payload.get("qdrant_point_id") or record.chunk_id),
                    score=max_existing_score + FORCED_REFERENCE_SCORE_MARGIN - (rank * 1e-4),
                    citation_text=record.citation_text,
                    payload=payload,
                )
            )

        remaining_hits = tuple(hit for hit in hits if hit.chunk_id not in seen_forced_chunk_ids)
        return tuple((*forced_hits, *remaining_hits))

    def append_reference_fallback_hits(
        self,
        hits: Sequence[SearchHit],
        intent: QueryIntent,
        *,
        limit: int,
    ) -> tuple[SearchHit, ...]:
        fallback_kind = "explicit" if intent.article_numbers else "high_confidence"
        article_candidates = intent.article_numbers
        if not article_candidates:
            article_candidates = tuple(
                article
                for article in intent.force_reference_article_numbers
                if article not in intent.article_numbers
            )
            if len(article_candidates) > 4:
                return tuple(hits)

        if not article_candidates:
            return tuple(hits)

        existing_records = self.record_store.fetch_records_from_hits(hits)
        existing_articles = {
            str(record.payload.get("article_number") or "")
            for record in existing_records.values()
        }
        missing_articles = tuple(
            article for article in article_candidates if article not in existing_articles
        )
        if fallback_kind == "explicit":
            article_numbers_to_fetch = missing_articles or article_candidates
            if not missing_articles and not intent.clause_refs and not intent.point_refs:
                return tuple(hits)
        else:
            article_numbers_to_fetch = missing_articles
            if not article_numbers_to_fetch or intent.clause_refs or intent.point_refs:
                return tuple(hits)

        if article_numbers_to_fetch and not intent.clause_refs and not intent.point_refs:
            fallback_records_list: list[RetrievedRecord] = []
            excluded_chunk_ids: set[str] = set()
            query_context = self.rule_config.QUERY_CONTEXT
            primary_article_limit = max(
                limit,
                8
                if (
                    set(intent.query_types).intersection(
                        query_context.get("primary_article_limit_query_types", ())
                    )
                    or set(intent.issue_filters).intersection(
                        query_context.get("primary_article_limit_issues", ())
                    )
                )
                else 4,
            )
            secondary_article_limit = max(2, min(4, limit))
            for index, article_number in enumerate(article_numbers_to_fetch):
                fetched_records = self.record_store.fetch_records_by_reference(
                    document_ids=intent.document_filters,
                    article_numbers=(article_number,),
                    exclude_chunk_ids=tuple(excluded_chunk_ids),
                    limit=primary_article_limit if index == 0 else secondary_article_limit,
                )
                fallback_records_list.extend(fetched_records)
                excluded_chunk_ids.update(record.chunk_id for record in fetched_records)
            fallback_records = tuple(fallback_records_list)
        else:
            fallback_records = self.record_store.fetch_records_by_reference(
                document_ids=intent.document_filters,
                article_numbers=article_numbers_to_fetch,
                clause_refs=intent.clause_refs,
                point_refs=intent.point_refs,
                limit=limit,
            )
        if not fallback_records:
            return tuple(hits)

        max_existing_score = max((hit.score for hit in hits), default=0.0)
        fallback_score = (
            max_existing_score + 1e-3
            if fallback_kind == "explicit"
            else max(max_existing_score - 1e-3, 0.0)
        )
        fallback_hits_by_chunk_id = {
            record.chunk_id: self.record_to_search_hit(record, fallback_score - (rank * 1e-4))
            for rank, record in enumerate(dedupe_records_by_chunk_id(fallback_records))
        }
        promoted_hits: list[SearchHit] = []
        used_fallback_chunk_ids: set[str] = set()
        for hit in hits:
            fallback_hit = fallback_hits_by_chunk_id.get(hit.chunk_id)
            if fallback_hit is None:
                promoted_hits.append(hit)
                continue
            promoted_hits.append(fallback_hit if fallback_kind == "explicit" else hit)
            used_fallback_chunk_ids.add(hit.chunk_id)
        promoted_hits.extend(
            fallback_hit
            for chunk_id, fallback_hit in fallback_hits_by_chunk_id.items()
            if chunk_id not in used_fallback_chunk_ids
        )
        return tuple(promoted_hits)

    def pin_forced_reference_hits(
        self,
        hits: Sequence[SearchHit],
        intent: QueryIntent,
    ) -> tuple[SearchHit, ...]:
        if not intent.forced_references:
            return tuple(hits)

        pinned: list[SearchHit] = []
        remaining: list[SearchHit] = []
        seen_chunk_ids: set[str] = set()
        for hit in hits:
            if hit.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(hit.chunk_id)
            if bool(hit.payload.get("retrieval_forced_reference")) or self.hit_matches_forced_reference(hit, intent):
                pinned.append(hit)
            else:
                remaining.append(hit)
        return tuple((*pinned, *remaining))


__all__ = ["ReferenceExpander"]

