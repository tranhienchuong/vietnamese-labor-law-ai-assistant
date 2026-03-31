from __future__ import annotations

import os
import sys

import ollama


MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3:4b")
PROMPT = "Chào, hãy trả lời ngắn gọn bằng tiếng Việt."

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": PROMPT}],
    )
    print(response["message"]["content"])


if __name__ == "__main__":
    main()
