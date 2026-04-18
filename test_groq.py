from __future__ import annotations

import os
import sys

from vn_labor_law_ai_assistant.config import load_repo_env
from vn_labor_law_ai_assistant.llm import build_groq_client


load_repo_env()

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
PROMPT = "Chao, hay tra loi ngan gon bang tieng Viet."

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    client = build_groq_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": PROMPT}],
    )
    print(response.choices[0].message.content or "")


if __name__ == "__main__":
    main()
