from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vn_labor_law_ai_assistant.retriever import HybridRetriever, query_intent_from_metadata
from vn_labor_law_ai_assistant.query_router import (
    DEFAULT_ROUTER_MODEL,
    QueryMetadata,
    analyze_query_smart,
    build_query_router_messages,
    metadata_to_retrieval_filters,
    parse_query_metadata,
    query_metadata_json_schema,
)


class QueryRouterTests(unittest.TestCase):
    def test_schema_is_strict_and_exposes_router_fields(self) -> None:
        schema = query_metadata_json_schema()

        self.assertFalse(schema["additionalProperties"])
        self.assertIn("actor", schema["required"])
        self.assertIn("topics", schema["required"])
        self.assertIn("issues", schema["required"])

    def test_prompt_warns_against_keyword_only_maternity_matches(self) -> None:
        messages = build_query_router_messages(
            "Cong ty cam nhan vien noi chuyen ve phu nu mang thai co dung luat khong?"
        )

        system_prompt = messages[0]["content"]
        self.assertIn("Classify the user's legal intent, not isolated keywords", system_prompt)
        self.assertIn("do not output thai_san", system_prompt)
        self.assertIn("bao_ve_thai_san", system_prompt)

    def test_parse_query_metadata_sanitizes_unknown_labels_and_aliases(self) -> None:
        metadata = parse_query_metadata(
            json.dumps(
                {
                    "actor": "lao_dong_nu",
                    "actors": ["nguoi_lao_dong", "unknown_actor"],
                    "topics": ["thai_san", "cham_dut_hop_dong", "unknown_topic"],
                    "issues": ["boi_thuong", "bhxh", "unknown_issue"],
                    "document_ids": ["45-2019-qh14", "missing-document"],
                    "query_types": ["yes_no", "nonsense"],
                    "article_numbers": ["Dieu 35", "41", "abc"],
                    "clause_refs": ["khoan 2", "3"],
                    "point_refs": ["diem a", "b.1"],
                }
            )
        )

        self.assertEqual(metadata.actor, "lao_dong_nu")
        self.assertEqual(metadata.actors, ["lao_dong_nu", "nguoi_lao_dong"])
        self.assertEqual(
            metadata.topics,
            ["bao_ve_thai_san", "cham_dut_hop_dong_lao_dong"],
        )
        self.assertEqual(metadata.issues, ["boi_thuong", "bao_hiem_xa_hoi"])
        self.assertEqual(metadata.document_ids, ["45-2019-qh14"])
        self.assertEqual(metadata.query_types, ["yes_no"])
        self.assertEqual(metadata.article_numbers, ["35", "41"])
        self.assertEqual(metadata.clause_refs, ["2", "3"])
        self.assertEqual(metadata.point_refs, ["a", "b.1"])

    def test_parse_query_metadata_accepts_json_inside_code_fence(self) -> None:
        metadata = parse_query_metadata(
            """```json
{"actor": null, "actors": [], "topics": [], "issues": [], "document_ids": [], "query_types": [], "article_numbers": [], "clause_refs": [], "point_refs": []}
```"""
        )

        self.assertIsNone(metadata.actor)
        self.assertEqual(metadata.topics, [])

    def test_analyze_query_smart_calls_completion_with_schema(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "actor": "lao_dong_nu",
                        "actors": ["lao_dong_nu"],
                        "topics": ["bao_ve_thai_san", "cham_dut_hop_dong_lao_dong"],
                        "issues": ["boi_thuong"],
                        "document_ids": [],
                        "query_types": ["yes_no"],
                        "article_numbers": [],
                        "clause_refs": [],
                        "point_refs": [],
                    }
                )
            )

        metadata = analyze_query_smart(
            "Ba bau nghi ngang co phai boi thuong khong?",
            provider="groq",
            model="openai/gpt-oss-20b",
            completion_fn=fake_completion,
        )

        self.assertEqual(metadata.actor, "lao_dong_nu")
        self.assertIn("bao_ve_thai_san", metadata.topics)
        self.assertEqual(calls[0]["provider"], "groq")
        self.assertEqual(calls[0]["model"], "openai/gpt-oss-20b")
        self.assertEqual(calls[0]["json_schema_name"], "query_metadata")
        self.assertIn("properties", calls[0]["json_schema"])

    def test_analyze_query_smart_defaults_to_small_router_model(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "actor": None,
                        "actors": [],
                        "topics": [],
                        "issues": [],
                        "document_ids": [],
                        "query_types": [],
                        "article_numbers": [],
                        "clause_refs": [],
                        "point_refs": [],
                    }
                )
            )

        analyze_query_smart("Noi quy cong ty quy dinh the nao?", completion_fn=fake_completion)

        self.assertEqual(calls[0]["model"], DEFAULT_ROUTER_MODEL)

    def test_metadata_to_retrieval_filters_uses_qdrant_payload_keys(self) -> None:
        metadata = parse_query_metadata(
            json.dumps(
                {
                    "actor": "nguoi_lao_dong",
                    "actors": ["nguoi_lao_dong"],
                    "topics": ["tro_cap"],
                    "issues": ["tro_cap_thoi_viec"],
                    "document_ids": ["45-2019-qh14"],
                    "query_types": ["money_percentage"],
                    "article_numbers": ["46"],
                    "clause_refs": [],
                    "point_refs": [],
                }
            )
        )

        filters = metadata_to_retrieval_filters(metadata)

        self.assertEqual(filters["actor"], ["nguoi_lao_dong"])
        self.assertEqual(filters["topic"], ["tro_cap"])
        self.assertEqual(filters["issue_type"], ["tro_cap_thoi_viec"])
        self.assertEqual(filters["document_id"], ["45-2019-qh14"])
        self.assertEqual(filters["article_number"], ["46"])

    def test_query_intent_from_metadata_uses_llm_labels(self) -> None:
        metadata = QueryMetadata.model_validate(
            {
                "actor": "nguoi_lao_dong",
                "actors": ["nguoi_lao_dong"],
                "topics": ["tien_luong"],
                "issues": ["tien_luong"],
                "document_ids": [],
                "query_types": ["yes_no"],
                "article_numbers": [],
                "clause_refs": [],
                "point_refs": [],
            }
        )

        intent = query_intent_from_metadata(
            "Cong ty cam nhan vien noi chuyen ve phu nu mang thai co dung luat khong?",
            metadata,
        )

        self.assertEqual(intent.actor_filters, ("nguoi_lao_dong",))
        self.assertEqual(intent.topic_filters, ("tien_luong",))
        self.assertNotIn("bao_ve_thai_san", intent.topic_filters)
        self.assertIn("97", intent.inferred_article_numbers)
        self.assertTrue(intent.query_expansions)

    def test_query_intent_from_metadata_merges_rule_based_article_anchors(self) -> None:
        metadata = QueryMetadata.model_validate(
            {
                "actor": "nguoi_lao_dong",
                "actors": ["nguoi_lao_dong"],
                "topics": ["hop_dong_lao_dong"],
                "issues": [],
                "document_ids": [],
                "query_types": ["yes_no"],
                "article_numbers": [],
                "clause_refs": [],
                "point_refs": [],
            }
        )

        intent = query_intent_from_metadata(
            "Cong ty giu CCCD ban goc cua toi co duoc khong?",
            metadata,
        )

        self.assertIn("17", intent.inferred_article_numbers)
        self.assertIn("giu_giay_to_goc", intent.issue_filters)
        self.assertTrue(any("giay to tuy than" in value for value in intent.query_expansions))

    def test_hybrid_retriever_uses_llm_router_when_enabled(self) -> None:
        metadata = QueryMetadata.model_validate(
            {
                "actor": "lao_dong_nu",
                "actors": ["lao_dong_nu"],
                "topics": ["bao_ve_thai_san"],
                "issues": ["bao_ve_thai_san"],
                "document_ids": [],
                "query_types": ["yes_no"],
                "article_numbers": [],
                "clause_refs": [],
                "point_refs": [],
            }
        )
        expected_intent = query_intent_from_metadata("Ba bau nghi ngang duoc khong?", metadata)
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._query_router_enabled = True
        retriever._query_router_provider = "groq"
        retriever._query_router_model = DEFAULT_ROUTER_MODEL
        retriever._query_router_fallback_to_heuristic = True

        with patch(
            "vn_labor_law_ai_assistant.retriever.route_query_with_llm",
            return_value=expected_intent,
        ) as route_with_llm:
            intent = HybridRetriever._route_query(retriever, "Ba bau nghi ngang duoc khong?")

        self.assertEqual(intent, expected_intent)
        route_with_llm.assert_called_once_with(
            "Ba bau nghi ngang duoc khong?",
            provider="groq",
            model=DEFAULT_ROUTER_MODEL,
        )

    def test_hybrid_retriever_can_fallback_to_heuristic_router(self) -> None:
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._query_router_enabled = True
        retriever._query_router_provider = "groq"
        retriever._query_router_model = DEFAULT_ROUTER_MODEL
        retriever._query_router_fallback_to_heuristic = True

        with patch(
            "vn_labor_law_ai_assistant.retriever.route_query_with_llm",
            side_effect=RuntimeError("router unavailable"),
        ):
            intent = HybridRetriever._route_query(
                retriever,
                "Toi xin nghi viec phai bao truoc bao lau?",
            )

        self.assertIn("bao_truoc", intent.topic_filters)


if __name__ == "__main__":
    unittest.main()
