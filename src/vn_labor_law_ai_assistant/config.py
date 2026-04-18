from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"


def _strip_optional_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_repo_env(*, override: bool = False) -> Path | None:
    if not ENV_FILE.exists():
        return None

    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
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

    return ENV_FILE


__all__ = ["ENV_FILE", "REPO_ROOT", "load_repo_env"]
