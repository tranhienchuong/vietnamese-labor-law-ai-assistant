from __future__ import annotations

from dataclasses import dataclass

ANSWER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "legal_basis": {
            "type": "array",
            "items": {"type": "string"},
        },
        "evidence_quotes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "citation": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["citation", "quote"],
                "additionalProperties": False,
            },
        },
        "insufficient_context": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": ["answer", "legal_basis", "evidence_quotes", "insufficient_context", "notes"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class EvidenceQuote:
    citation: str
    quote: str


@dataclass(frozen=True)
class ParsedAnswer:
    answer: str
    legal_basis: tuple[str, ...]
    evidence_quotes: tuple[EvidenceQuote, ...]
    insufficient_context: bool
    notes: str
    raw_content: str


__all__ = [
    "ANSWER_JSON_SCHEMA",
    "EvidenceQuote",
    "ParsedAnswer",
]
