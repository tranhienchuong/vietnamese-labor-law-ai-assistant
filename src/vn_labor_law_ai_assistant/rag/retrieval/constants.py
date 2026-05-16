from __future__ import annotations

import re

from ...rule_loader import DEFAULT_RULE_CONFIG


RULE_CONFIG = DEFAULT_RULE_CONFIG

DEFAULT_MAX_CONTEXT_CHARS = 8000
DEFAULT_MAX_CONTEXT_TOKENS = 1400
TOKEN_ESTIMATE_RE = re.compile(r"\S+")
DEFAULT_RERANKER_TOP_N = 24
RECORD_SOURCE_SQLITE = "sqlite"
RECORD_SOURCE_QDRANT_PAYLOAD = "qdrant_payload"
SUPPORTED_RECORD_SOURCES = {RECORD_SOURCE_SQLITE, RECORD_SOURCE_QDRANT_PAYLOAD}
RRF_K = 60.0
FORCED_REFERENCE_SCORE_MARGIN = 10.0
TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
FALSE_ENV_VALUES = frozenset({"0", "false", "no", "off"})
POINT_REF_ALPHABET = (
    "a",
    "b",
    "c",
    "d",
    "\u0111",
    "e",
    "g",
    "h",
    "i",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "x",
    "y",
)
POINT_REF_ORDER = {value: index for index, value in enumerate(POINT_REF_ALPHABET)}

__all__ = [
    "DEFAULT_MAX_CONTEXT_CHARS",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "DEFAULT_RERANKER_TOP_N",
    "FALSE_ENV_VALUES",
    "FORCED_REFERENCE_SCORE_MARGIN",
    "POINT_REF_ALPHABET",
    "POINT_REF_ORDER",
    "RECORD_SOURCE_QDRANT_PAYLOAD",
    "RECORD_SOURCE_SQLITE",
    "RRF_K",
    "RULE_CONFIG",
    "SUPPORTED_RECORD_SOURCES",
    "TOKEN_ESTIMATE_RE",
    "TRUE_ENV_VALUES",
]

