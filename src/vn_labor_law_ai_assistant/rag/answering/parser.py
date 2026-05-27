from __future__ import annotations

import json
import re
from typing import Sequence

from ...retriever import RetrievalContext, dedupe_preserve_order
from .citation_guard import (
    normalize_quote_surface,
    sanitize_evidence_quotes,
    sanitize_legal_basis,
)
from .overrides import contextual_answer_override
from .schema import EvidenceQuote, ParsedAnswer

def extract_json_candidate(raw_content: str) -> str:
    cleaned_content = raw_content.strip()

    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_content, re.IGNORECASE)
    if fenced_match:
        cleaned_content = fenced_match.group(1).strip()

    if (
        (cleaned_content.startswith("{") and cleaned_content.endswith("}"))
        or (cleaned_content.startswith("[") and cleaned_content.endswith("]"))
    ):
        return cleaned_content

    container_match = re.search(r"(\{.*\}|\[.*\])", cleaned_content, re.DOTALL)
    if container_match:
        return container_match.group(1).strip()

    return cleaned_content


def parse_answer_payload(
    raw_content: str,
    contexts: Sequence[RetrievalContext],
    *,
    question: str = "",
) -> ParsedAnswer:
    cleaned_content = extract_json_candidate(raw_content)

    try:
        raw_payload = json.loads(cleaned_content)
        if isinstance(raw_payload, list) and raw_payload and isinstance(raw_payload[0], dict):
            payload = raw_payload[0]
        elif isinstance(raw_payload, dict):
            payload = raw_payload
        else:
            raise ValueError("JSON is valid but not a dictionary or list of dictionaries.")
    except (json.JSONDecodeError, ValueError):
        payload = {
            "answer": raw_content.strip(),
            "legal_basis": [],
            "insufficient_context": True,
            "notes": "Mo hinh khong tra ve JSON hop le; khong the xac nhan can cu phap ly.",
        }

    insufficient_context = bool(payload.get("insufficient_context"))
    legal_basis = ()
    evidence_quotes: tuple[EvidenceQuote, ...] = ()
    if not insufficient_context:
        legal_basis = sanitize_legal_basis(payload.get("legal_basis"), contexts)
        evidence_quotes = sanitize_evidence_quotes(payload.get("evidence_quotes"), contexts)
        if not legal_basis:
            existing_notes = str(payload.get("notes") or "").strip()
            payload["notes"] = " ".join(
                part
                for part in [
                    existing_notes,
                    "Khong xac nhan duoc co so phap ly vi legal_basis khong khop ALLOWED_CITATIONS.",
                ]
                if part
            ).strip()

            if not str(payload.get("answer") or "").strip():
                insufficient_context = True
                payload["answer"] = (
                    "Chua du can cu de ket luan mot cach chac chan tu context hien tai."
                )
        elif not evidence_quotes:
            existing_notes = str(payload.get("notes") or "").strip()
            payload["notes"] = " ".join(
                part
                for part in [
                    existing_notes,
                    "Khong xac nhan duoc evidence_quotes truc tiep trong CONTEXT.",
                ]
                if part
            ).strip()

    if insufficient_context and not str(payload.get("answer") or "").strip():
        payload["answer"] = "Chua du can cu de ket luan mot cach chac chan tu context hien tai."

    override = contextual_answer_override(question, contexts) if question else None
    if override is not None:
        insufficient_context = False
        if normalize_quote_surface(override.evidence_quote.citation) in normalize_quote_surface(override.answer):
            payload["answer"] = override.answer
        else:
            payload["answer"] = f"{override.answer} ({override.evidence_quote.citation})"
        legal_basis = dedupe_preserve_order(
            (override.evidence_quote.citation, *legal_basis)
        )
        evidence_quotes = (override.evidence_quote,) + tuple(
            evidence_quote
            for evidence_quote in evidence_quotes
            if evidence_quote.citation != override.evidence_quote.citation
            or normalize_quote_surface(evidence_quote.quote)
            != normalize_quote_surface(override.evidence_quote.quote)
        )
        existing_notes = str(payload.get("notes") or "").strip()
        payload["notes"] = " ".join(
            part for part in (existing_notes, override.notes) if part
        ).strip()

    return ParsedAnswer(
        answer=str(payload.get("answer") or "").strip(),
        legal_basis=legal_basis,
        evidence_quotes=evidence_quotes,
        insufficient_context=insufficient_context,
        notes=str(payload.get("notes") or "").strip(),
        raw_content=raw_content,
    )


__all__ = [
    "extract_json_candidate",
    "parse_answer_payload",
]
