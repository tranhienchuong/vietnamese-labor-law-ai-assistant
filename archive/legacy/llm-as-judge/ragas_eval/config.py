from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


LLM_AS_JUDGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_THESPARKDAILY_BASE_URL = "https://api.thesparkdaily.com/v1"
DEFAULT_THESPARKDAILY_JUDGE_MODEL = "gpt-5.4-pro"
DEFAULT_THESPARKDAILY_FAST_JUDGE_MODEL = "gpt-5.4-nano"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRIES = 3


def _strip_optional_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_files(*, override: bool = False) -> tuple[Path, ...]:
    """Load repo-local .env files without adding a python-dotenv dependency."""

    loaded: list[Path] = []
    for env_path in (REPO_ROOT / ".env", LLM_AS_JUDGE_ROOT / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            if not override and key in os.environ:
                continue
            os.environ[key] = _strip_optional_quotes(value.strip())
        loaded.append(env_path)
    return tuple(loaded)


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


@dataclass(frozen=True)
class TheSparkDailyConfig:
    api_key: str
    base_url: str = DEFAULT_THESPARKDAILY_BASE_URL
    judge_model: str = DEFAULT_THESPARKDAILY_JUDGE_MODEL
    fast_judge_model: str = DEFAULT_THESPARKDAILY_FAST_JUDGE_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_retries: int = DEFAULT_MAX_RETRIES

    def require_api_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                "THESPARKDAILY_API_KEY is not set. Set it in the environment or .env before "
                "running a real evaluation. The key is never printed to logs."
            )
        return self.api_key

    def model_for_mode(self, mode: str) -> str:
        if mode == "fast":
            return self.fast_judge_model
        return self.judge_model


def load_thesparkdaily_config() -> TheSparkDailyConfig:
    load_env_files()
    return TheSparkDailyConfig(
        api_key=os.getenv("THESPARKDAILY_API_KEY", "").strip(),
        base_url=os.getenv("THESPARKDAILY_BASE_URL", DEFAULT_THESPARKDAILY_BASE_URL).strip()
        or DEFAULT_THESPARKDAILY_BASE_URL,
        judge_model=os.getenv(
            "THESPARKDAILY_JUDGE_MODEL",
            DEFAULT_THESPARKDAILY_JUDGE_MODEL,
        ).strip()
        or DEFAULT_THESPARKDAILY_JUDGE_MODEL,
        fast_judge_model=os.getenv(
            "THESPARKDAILY_FAST_JUDGE_MODEL",
            DEFAULT_THESPARKDAILY_FAST_JUDGE_MODEL,
        ).strip()
        or DEFAULT_THESPARKDAILY_FAST_JUDGE_MODEL,
        temperature=_env_float("THESPARKDAILY_TEMPERATURE", DEFAULT_TEMPERATURE),
        max_retries=max(0, _env_int("THESPARKDAILY_MAX_RETRIES", DEFAULT_MAX_RETRIES)),
    )


def resolve_output_path(output_path: Path | None, *, suffix: str = ".csv") -> Path:
    from datetime import datetime

    if output_path is not None:
        return output_path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return LLM_AS_JUDGE_ROOT / "outputs" / f"ragas_results_{timestamp}{suffix}"
