from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vn_labor_law_ai_assistant.answering import (
    contextual_answer_override,
    format_answer_for_user,
    parse_answer_payload,
)
from vn_labor_law_ai_assistant.retriever import RetrievalContext
from vn_labor_law_ai_assistant.rule_loader import (
    AnswerOverrideLoader,
    load_answer_override_rules,
)


def make_context(
    *,
    citation_text: str,
    text: str,
    matched_citations: tuple[str, ...] | None = None,
) -> RetrievalContext:
    return RetrievalContext(
        chunk_id=citation_text,
        citation_text=citation_text,
        text=text,
        payload={},
        score=1.0,
        matched_chunk_ids=(citation_text,),
        matched_citations=matched_citations or (citation_text,),
    )


class AnswerOverrideRuleTests(unittest.TestCase):
    def test_answer_override_yaml_loads(self) -> None:
        rules = load_answer_override_rules()
        names = {rule.name for rule in rules}

        self.assertIn("employment_contract_types", names)
        self.assertIn("indefinite_contract_notice_45_days", names)
        self.assertIn("probation_salary_85_percent", names)
        self.assertIn("overtime_weekly_rest_day_200_percent", names)

    def test_answer_override_rules_sorted_by_priority(self) -> None:
        priorities = [rule.priority for rule in load_answer_override_rules()]

        self.assertEqual(priorities, sorted(priorities, reverse=True))

    def test_answer_override_invalid_rule_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_name_path = Path(tmp_dir) / "missing_name.yaml"
            missing_name_path.write_text(
                "answer_overrides:\n"
                "  - enabled: true\n"
                "    priority: 1\n"
                "    answer: Missing name\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "name"):
                AnswerOverrideLoader(missing_name_path).load()

            missing_answer_path = Path(tmp_dir) / "missing_answer.yaml"
            missing_answer_path.write_text(
                "answer_overrides:\n"
                "  - name: missing_answer\n"
                "    enabled: true\n"
                "    priority: 1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "answer"):
                AnswerOverrideLoader(missing_answer_path).load()

    def test_contextual_override_employment_contract_types(self) -> None:
        contexts = (
            make_context(
                citation_text="Bo luat so 45/2019/QH14, Dieu 20, khoan 1",
                text=(
                    "Hop dong lao dong phai duoc giao ket theo mot trong cac loai sau day: "
                    "hop dong lao dong khong xac dinh thoi han va hop dong lao dong xac dinh thoi han."
                ),
            ),
        )

        override = contextual_answer_override("Hop dong lao dong co may loai?", contexts)

        self.assertIsNotNone(override)
        assert override is not None
        self.assertEqual(
            override.answer,
            "Co 2 loai hop dong lao dong: hop dong lao dong khong xac dinh thoi han va hop dong lao dong xac dinh thoi han.",
        )
        self.assertEqual(override.evidence_quote.citation, "Bo luat so 45/2019/QH14, Dieu 20, khoan 1")
        self.assertIn("khong xac dinh thoi han", override.evidence_quote.quote)
        self.assertIn("xac dinh thoi han", override.evidence_quote.quote)

    def test_contextual_override_45_days(self) -> None:
        contexts = (
            make_context(
                citation_text="Bo luat so 45/2019/QH14, Dieu 35, khoan 1",
                matched_citations=("Bo luat so 45/2019/QH14, Dieu 35, khoan 1, diem a",),
                text=(
                    "Nguoi lao dong co quyen don phuong cham dut hop dong lao dong nhung phai bao truoc "
                    "it nhat 45 ngay neu lam viec theo hop dong lao dong khong xac dinh thoi han;"
                ),
            ),
        )

        override = contextual_answer_override(
            "Nguoi lao dong ky hop dong khong xac dinh thoi han muon nghi viec thi phai bao truoc bao lau?",
            contexts,
        )

        self.assertIsNotNone(override)
        assert override is not None
        self.assertIn("45 ngay", override.answer)
        self.assertEqual(
            override.evidence_quote.citation,
            "Bo luat so 45/2019/QH14, Dieu 35, khoan 1, diem a",
        )

    def test_contextual_override_85_percent(self) -> None:
        contexts = (
            make_context(
                citation_text="Bo luat so 45/2019/QH14, Dieu 26",
                text="Tien luong cua nguoi lao dong trong thoi gian thu viec it nhat phai bang 85% muc luong cua cong viec do.",
            ),
        )

        override = contextual_answer_override("Luong thu viec toi thieu la bao nhieu?", contexts)

        self.assertIsNotNone(override)
        assert override is not None
        self.assertEqual(
            override.answer,
            "Muc luong thu viec it nhat phai bang 85% muc luong cua cong viec do.",
        )
        self.assertIn("85%", override.evidence_quote.quote)

    def test_contextual_override_fixed_contract_30_days(self) -> None:
        contexts = (
            make_context(
                citation_text="Bo luat so 45/2019/QH14, Dieu 35, khoan 1",
                matched_citations=("Bo luat so 45/2019/QH14, Dieu 35, khoan 1, diem b",),
                text=(
                    "It nhat 30 ngay neu lam viec theo hop dong lao dong xac dinh thoi han "
                    "co thoi han tu 12 thang den 36 thang;"
                ),
            ),
        )

        override = contextual_answer_override(
            "Hop dong lao dong xac dinh thoi han tu 12 den 36 thang phai bao truoc bao lau?",
            contexts,
        )

        self.assertIsNotNone(override)
        assert override is not None
        self.assertIn("30 ngay", override.answer)
        self.assertEqual(
            override.evidence_quote.citation,
            "Bo luat so 45/2019/QH14, Dieu 35, khoan 1, diem b",
        )

    def test_contextual_override_under_12_months_3_working_days(self) -> None:
        contexts = (
            make_context(
                citation_text="Bo luat so 45/2019/QH14, Dieu 35, khoan 1",
                matched_citations=("Bo luat so 45/2019/QH14, Dieu 35, khoan 1, diem c",),
                text="It nhat 03 ngay lam viec neu lam viec theo hop dong lao dong xac dinh thoi han co thoi han duoi 12 thang;",
            ),
        )

        override = contextual_answer_override(
            "Hop dong lao dong duoi 12 thang phai bao truoc bao lau?",
            contexts,
        )

        self.assertIsNotNone(override)
        assert override is not None
        self.assertIn("03 ngay lam viec", override.answer)

    def test_contextual_override_probation_college_60_days(self) -> None:
        contexts = (
            make_context(
                citation_text="Bo luat so 45/2019/QH14, Dieu 25",
                text="Khong qua 60 ngay doi voi cong viec co chuc danh nghe nghiep can trinh do chuyen mon, ky thuat tu cao dang tro len;",
            ),
        )

        override = contextual_answer_override("Thu viec trinh do cao dang toi da bao lau?", contexts)

        self.assertIsNotNone(override)
        assert override is not None
        self.assertIn("60 ngay", override.answer)

    def test_contextual_override_overtime_weekly_rest_day_200_percent(self) -> None:
        contexts = (
            make_context(
                citation_text="Bo luat so 45/2019/QH14, Dieu 98",
                text="Vao ngay nghi hang tuan, it nhat bang 200% so voi don gia tien luong hoac tien luong thuc tra theo cong viec dang lam.",
            ),
        )

        override = contextual_answer_override("Lam them vao ngay nghi hang tuan duoc tra luong the nao?", contexts)

        self.assertIsNotNone(override)
        assert override is not None
        self.assertIn("200%", override.answer)

    def test_contextual_override_no_match_returns_none(self) -> None:
        contexts = (
            make_context(
                citation_text="Bo luat so 45/2019/QH14, Dieu 113",
                text="Nguoi lao dong co ngay nghi hang nam theo quy dinh.",
            ),
        )

        self.assertIsNone(contextual_answer_override("Toi co duoc nghi phep nam khong?", contexts))

    def test_compat_exports(self) -> None:
        self.assertTrue(callable(contextual_answer_override))
        self.assertTrue(callable(parse_answer_payload))
        self.assertTrue(callable(format_answer_for_user))


if __name__ == "__main__":
    unittest.main()
