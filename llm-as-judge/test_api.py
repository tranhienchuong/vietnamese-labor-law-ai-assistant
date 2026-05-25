from __future__ import annotations

from ragas_eval.config import load_thesparkdaily_config
from ragas_eval.thesparkdaily_llm import build_chat_openai, invoke_chat_model


def main() -> int:
    config = load_thesparkdaily_config()
    chat_model = build_chat_openai(model=config.judge_model, config=config)
    response = invoke_chat_model(
        chat_model,
        [
            {
                "role": "system",
                "content": "You are a concise Vietnamese assistant.",
            },
            {
                "role": "user",
                "content": "Xin chao, ban dang hoat dong khong?",
            },
        ],
    )
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
