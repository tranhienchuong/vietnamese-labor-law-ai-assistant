from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata


SUPPORTED_CITATION_MATCH_MODES = frozenset({"literal", "normalized", "containment"})
DEFAULT_CITATION_MATCH_MODE = "containment"

ARTICLE_RE = re.compile(r"\bdieu\s+(?P<value>\d+[a-z]?)\b")
CLAUSE_GROUP_RE = re.compile(
    r"\b(?:cac\s+)?khoan\s+(?P<values>\d+(?:\s*(?:,|va)\s*\d+)*)"
)
POINT_GROUP_RE = re.compile(
    r"\b(?:cac\s+)?diem\s+(?P<values>[a-z](?:\.\d+)?(?:\s*(?:,|va)\s*[a-z](?:\.\d+)?)*)"
)
VALUE_SPLIT_RE = re.compile(r"\s*(?:,|va)\s*")
LAW_CODE_2019_RE = re.compile(r"\b(?:45\s*/\s*2019\s*/\s*qh\s*14|bo luat lao dong 2019)\b")
DECREE_145_RE = re.compile(r"\b(?:145\s*/\s*2020\s*/\s*nd\s*cp|nghi dinh 145)\b")


@dataclass(frozen=True)
class CitationRef:
    document_id: str | None
    law_id: str | None
    article: str | None
    clauses: frozenset[str] = field(default_factory=frozenset)
    points: frozenset[str] = field(default_factory=frozenset)
    raw: str = ""


def normalize_vietnamese_citation(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.replace("Ä‘", "d").replace("Ä", "d")
    normalized = normalized.replace("đ", "d").replace("Đ", "d")
    normalized = normalized.lower()
    normalized = re.sub(r"[;:()\[\]{}]", " ", normalized)
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    normalized = re.sub(r"\s*,\s*", ", ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def split_values(raw_values: str) -> frozenset[str]:
    values: list[str] = []
    for value in VALUE_SPLIT_RE.split(raw_values.strip()):
        cleaned = re.sub(r"\s+", "", value.strip().lower())
        if cleaned:
            values.append(cleaned)
    return frozenset(values)


def infer_document_id(normalized_text: str) -> str | None:
    if LAW_CODE_2019_RE.search(normalized_text):
        return "bo_luat_2019"
    if DECREE_145_RE.search(normalized_text):
        return "nghi_dinh_145"
    return None


def parse_citation(text: str) -> CitationRef:
    normalized = normalize_vietnamese_citation(text)
    articles = [match.group("value").lower() for match in ARTICLE_RE.finditer(normalized)]
    clauses: set[str] = set()
    points: set[str] = set()

    for match in CLAUSE_GROUP_RE.finditer(normalized):
        clauses.update(split_values(match.group("values")))
    for match in POINT_GROUP_RE.finditer(normalized):
        points.update(split_values(match.group("values")))

    document_id = infer_document_id(normalized)
    return CitationRef(
        document_id=document_id,
        law_id=document_id,
        article=articles[0] if len(set(articles)) == 1 else None,
        clauses=frozenset(clauses),
        points=frozenset(points),
        raw=str(text or ""),
    )


def citation_contains(
    retrieved: CitationRef,
    expected: CitationRef,
    *,
    allow_broad_retrieved: bool = True,
) -> bool:
    if not (expected.document_id or expected.law_id or expected.article or expected.clauses or expected.points):
        return False

    if expected.document_id and retrieved.document_id and expected.document_id != retrieved.document_id:
        return False
    if expected.law_id and retrieved.law_id and expected.law_id != retrieved.law_id:
        return False

    if expected.article:
        if not retrieved.article or retrieved.article != expected.article:
            return False
    elif retrieved.article and expected.article and retrieved.article != expected.article:
        return False

    if expected.clauses:
        if retrieved.clauses:
            if not expected.clauses.issubset(retrieved.clauses):
                return False
        elif not allow_broad_retrieved:
            return False

    if expected.points:
        if retrieved.points:
            if not expected.points.issubset(retrieved.points):
                return False
        elif not allow_broad_retrieved:
            return False

    return True


def literal_citation_matches(retrieved_text: str, expected_text: str) -> bool:
    retrieved = normalize_vietnamese_citation(retrieved_text)
    expected = normalize_vietnamese_citation(expected_text)
    return bool(expected and retrieved and (expected in retrieved or retrieved in expected))


def citation_matches(retrieved_text: str, expected_text: str, mode: str = DEFAULT_CITATION_MATCH_MODE) -> bool:
    normalized_mode = normalize_match_mode(mode)
    if normalized_mode == "literal":
        return literal_citation_matches(retrieved_text, expected_text)

    retrieved = parse_citation(retrieved_text)
    expected = parse_citation(expected_text)
    if not (expected.document_id or expected.law_id or expected.article or expected.clauses or expected.points):
        return literal_citation_matches(retrieved_text, expected_text)
    if normalized_mode == "normalized":
        return citation_contains(retrieved, expected, allow_broad_retrieved=False)
    return citation_contains(retrieved, expected, allow_broad_retrieved=True)


def normalize_match_mode(mode: str | None) -> str:
    normalized = str(mode or DEFAULT_CITATION_MATCH_MODE).strip().lower()
    if normalized not in SUPPORTED_CITATION_MATCH_MODES:
        raise ValueError(
            "EVAL_CITATION_MATCH_MODE must be one of: "
            + ", ".join(sorted(SUPPORTED_CITATION_MATCH_MODES))
        )
    return normalized


__all__ = [
    "CitationRef",
    "DEFAULT_CITATION_MATCH_MODE",
    "SUPPORTED_CITATION_MATCH_MODES",
    "citation_contains",
    "citation_matches",
    "literal_citation_matches",
    "normalize_match_mode",
    "normalize_vietnamese_citation",
    "parse_citation",
]
