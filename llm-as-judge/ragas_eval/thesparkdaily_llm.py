from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .config import TheSparkDailyConfig, load_thesparkdaily_config


def build_chat_openai(
    *,
    model: str | None = None,
    config: TheSparkDailyConfig | None = None,
    json_mode: bool = False,
):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "langchain-openai is required for TheSparkDaily judge calls. "
            "Install project dependencies before running evaluation."
        ) from exc

    resolved_config = config or load_thesparkdaily_config()
    api_key = resolved_config.require_api_key()
    model_kwargs: dict[str, Any] = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    return ChatOpenAI(
        model=model or resolved_config.judge_model,
        api_key=api_key,
        base_url=resolved_config.base_url,
        temperature=resolved_config.temperature,
        max_retries=resolved_config.max_retries,
        model_kwargs=model_kwargs,
    )


def build_judge_llm(
    *,
    model: str | None = None,
    config: TheSparkDailyConfig | None = None,
):
    try:
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise RuntimeError(
            "ragas is required for LangchainLLMWrapper. Install project dependencies first."
        ) from exc

    return LangchainLLMWrapper(
        build_chat_openai(
            model=model,
            config=config,
        )
    )


def invoke_chat_model(chat_model: Any, messages: Sequence[dict[str, str]]) -> str:
    response = chat_model.invoke(list(messages))
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(str(part) for part in content)
    return str(content or "")
