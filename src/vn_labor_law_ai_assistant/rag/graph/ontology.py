from __future__ import annotations

from enum import Enum


class NodeType(str, Enum):
    LEGAL_DOCUMENT = "Legal_Document"
    LEGAL_CHAPTER = "Legal_Chapter"
    LEGAL_SECTION = "Legal_Section"
    LEGAL_ARTICLE = "Legal_Article"
    LEGAL_CLAUSE = "Legal_Clause"
    LEGAL_POINT = "Legal_Point"
    LEGAL_APPENDIX = "Legal_Appendix"
    LEGAL_TOPIC = "Legal_Topic"
    LEGAL_ACTOR = "Legal_Actor"
    LEGAL_ISSUE_TYPE = "Legal_IssueType"
    LEGAL_RULE = "Legal_Rule"
    SUBJECT = "Subject"
    LEGAL_CONCEPT = "Legal_Concept"
    ACTION = "Action"
    CONDITION = "Condition"
    EXCEPTION = "Exception"
    RIGHT = "Right"
    OBLIGATION = "Obligation"
    CONSEQUENCE = "Consequence"
    SANCTION = "Sanction"
    DEADLINE = "Deadline"
    FORMULA = "Formula"
    PROCEDURE = "Procedure"
    EVIDENCE_CHUNK = "Evidence_Chunk"


class EdgeType(str, Enum):
    HAS_CHAPTER = "HAS_CHAPTER"
    HAS_SECTION = "HAS_SECTION"
    HAS_ARTICLE = "HAS_ARTICLE"
    HAS_CLAUSE = "HAS_CLAUSE"
    HAS_POINT = "HAS_POINT"
    HAS_APPENDIX = "HAS_APPENDIX"
    HAS_SOURCE_CHUNK = "HAS_SOURCE_CHUNK"
    SOURCE_OF = "SOURCE_OF"
    REFERENCES = "REFERENCES"
    GUIDED_BY = "GUIDED_BY"
    GUIDES = "GUIDES"
    DETAILS = "DETAILS"
    SUPERIOR_TO = "SUPERIOR_TO"
    SUBORDINATE_TO = "SUBORDINATE_TO"
    MUST_COMPLY_WITH = "MUST_COMPLY_WITH"
    IMPLEMENTS = "IMPLEMENTS"
    AMENDS = "AMENDS"
    REPLACES = "REPLACES"
    APPLIES_TO = "APPLIES_TO"
    MENTIONS_TOPIC = "MENTIONS_TOPIC"
    APPLIES_TO_ACTOR = "APPLIES_TO_ACTOR"
    HAS_ISSUE_TYPE = "HAS_ISSUE_TYPE"
    MENTIONS_CONCEPT = "MENTIONS_CONCEPT"
    REGULATES_ACTION = "REGULATES_ACTION"
    HAS_CONDITION = "HAS_CONDITION"
    HAS_EXCEPTION = "HAS_EXCEPTION"
    HAS_DEADLINE = "HAS_DEADLINE"
    HAS_FORMULA = "HAS_FORMULA"
    GRANTS_RIGHT = "GRANTS_RIGHT"
    IMPOSES_OBLIGATION = "IMPOSES_OBLIGATION"
    TRIGGERS_CONSEQUENCE = "TRIGGERS_CONSEQUENCE"
    PROHIBITS = "PROHIBITS"
    PERMITS = "PERMITS"
    REQUIRES = "REQUIRES"


GRAPH_EXPANSION_EDGE_TYPES: tuple[EdgeType, ...] = (
    EdgeType.REFERENCES,
    EdgeType.DETAILS,
    EdgeType.GUIDED_BY,
    EdgeType.GUIDES,
    EdgeType.SUPERIOR_TO,
    EdgeType.SUBORDINATE_TO,
    EdgeType.MUST_COMPLY_WITH,
    EdgeType.IMPLEMENTS,
    EdgeType.MENTIONS_TOPIC,
    EdgeType.APPLIES_TO_ACTOR,
    EdgeType.HAS_ISSUE_TYPE,
    EdgeType.HAS_SOURCE_CHUNK,
    EdgeType.SOURCE_OF,
)


def node_type_label(node_type: NodeType | str) -> str:
    value = node_type.value if isinstance(node_type, NodeType) else str(node_type)
    if value not in {item.value for item in NodeType}:
        raise ValueError(f"Unsupported legal graph node type: {value}")
    return value


def edge_type_label(edge_type: EdgeType | str) -> str:
    value = edge_type.value if isinstance(edge_type, EdgeType) else str(edge_type)
    if value not in {item.value for item in EdgeType}:
        raise ValueError(f"Unsupported legal graph edge type: {value}")
    return value


__all__ = [
    "EdgeType",
    "GRAPH_EXPANSION_EDGE_TYPES",
    "NodeType",
    "edge_type_label",
    "node_type_label",
]
