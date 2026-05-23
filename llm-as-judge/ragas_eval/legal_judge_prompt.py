from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping


LEGAL_JUDGE_SYSTEM_PROMPT = """Bạn là evaluator cho hệ thống RAG luật lao động Việt Nam.

Nhiệm vụ:
Đánh giá câu trả lời của hệ thống dựa trên:
1. user_input
2. response
3. retrieved_contexts
4. reference
5. reference_contexts
6. gold_citation

Quy tắc:
- Không dùng kiến thức ngoài dữ liệu được cung cấp.
- Không suy diễn thêm nếu reference/context không có.
- Ưu tiên tính đúng luật, đúng căn cứ, đủ điều kiện, đủ ngoại lệ.
- Nếu câu trả lời đúng nhưng thiếu điều kiện quan trọng, điểm completeness phải thấp.
- Nếu citation sai điều/khoản/nghị định, citation_correctness phải thấp.
- Nếu response có thông tin không được context hỗ trợ, faithfulness/legal_safety phải thấp.
- Nếu response khuyên người dùng hành động pháp lý chắc chắn trong khi dữ kiện thiếu, phải trừ điểm legal_safety.

Output JSON bắt buộc:

{
  "legal_correctness": 0.0,
  "citation_correctness": 0.0,
  "legal_completeness": 0.0,
  "legal_safety": 0.0,
  "legal_overall_score": 0.0,
  "error_type": "none | wrong_law | wrong_citation | missing_condition | hallucination | incomplete | unsafe_advice",
  "explanation": "Giải thích ngắn gọn bằng tiếng Việt"
}

Điểm số từ 0 đến 1:
- 1.0 = hoàn toàn đúng
- 0.7-0.9 = đúng chính nhưng thiếu chi tiết nhỏ
- 0.4-0.6 = có phần đúng nhưng thiếu/sai đáng kể
- 0.1-0.3 = phần lớn sai
- 0.0 = sai hoàn toàn hoặc hallucination nghiêm trọng

Chỉ trả về JSON hợp lệ, không thêm markdown hoặc chú thích ngoài JSON."""

LEGAL_JUDGE_SCORE_FIELDS = (
    "legal_correctness",
    "citation_correctness",
    "legal_completeness",
    "legal_safety",
    "legal_overall_score",
)
LEGAL_JUDGE_ERROR_TYPES = {
    "none",
    "wrong_law",
    "wrong_citation",
    "missing_condition",
    "hallucination",
    "incomplete",
    "unsafe_advice",
}


@dataclass(frozen=True)
class LegalJudgeScore:
    legal_correctness: float
    citation_correctness: float
    legal_completeness: float
    legal_safety: float
    legal_overall_score: float
    error_type: str
    explanation: str
    raw_content: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "legal_correctness": self.legal_correctness,
            "citation_correctness": self.citation_correctness,
            "legal_completeness": self.legal_completeness,
            "legal_safety": self.legal_safety,
            "legal_overall_score": self.legal_overall_score,
            "error_type": self.error_type,
            "explanation": self.explanation,
        }


class LegalJudgeParseError(ValueError):
    pass


def build_legal_judge_messages(sample_payload: Mapping[str, Any]) -> list[dict[str, str]]:
    user_payload = {
        "user_input": sample_payload.get("user_input", ""),
        "response": sample_payload.get("response", ""),
        "retrieved_contexts": list(sample_payload.get("retrieved_contexts") or []),
        "reference": sample_payload.get("reference", ""),
        "reference_contexts": list(sample_payload.get("reference_contexts") or []),
        "gold_citation": sample_payload.get("gold_citation", ""),
    }
    return [
        {"role": "system", "content": LEGAL_JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False, indent=2),
        },
    ]


def parse_legal_judge_response(raw_content: str) -> LegalJudgeScore:
    payload = _extract_json_payload(raw_content)
    if not isinstance(payload, dict):
        raise LegalJudgeParseError("Legal judge output must be a JSON object.")

    scores: dict[str, float] = {}
    for field_name in LEGAL_JUDGE_SCORE_FIELDS:
        if field_name not in payload:
            raise LegalJudgeParseError(f"Missing required field: {field_name}")
        try:
            score = float(payload[field_name])
        except (TypeError, ValueError) as exc:
            raise LegalJudgeParseError(f"{field_name} must be a number.") from exc
        if not 0.0 <= score <= 1.0:
            raise LegalJudgeParseError(f"{field_name} must be between 0 and 1.")
        scores[field_name] = score

    error_type = str(payload.get("error_type", "")).strip()
    if error_type not in LEGAL_JUDGE_ERROR_TYPES:
        raise LegalJudgeParseError(
            "error_type must be one of: " + ", ".join(sorted(LEGAL_JUDGE_ERROR_TYPES))
        )

    explanation = str(payload.get("explanation", "")).strip()
    if not explanation:
        raise LegalJudgeParseError("explanation is required.")

    return LegalJudgeScore(
        legal_correctness=scores["legal_correctness"],
        citation_correctness=scores["citation_correctness"],
        legal_completeness=scores["legal_completeness"],
        legal_safety=scores["legal_safety"],
        legal_overall_score=scores["legal_overall_score"],
        error_type=error_type,
        explanation=explanation,
        raw_content=raw_content,
    )


def _extract_json_payload(raw_content: str) -> Any:
    text = raw_content.strip()
    if not text:
        raise LegalJudgeParseError("Legal judge returned empty content.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise LegalJudgeParseError("Legal judge output is not valid JSON.") from None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LegalJudgeParseError("Legal judge output contains invalid JSON.") from exc
