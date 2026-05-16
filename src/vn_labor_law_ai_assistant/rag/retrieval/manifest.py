from __future__ import annotations

import json
from pathlib import Path


def load_manifest(index_path: Path) -> dict[str, object]:
    if index_path.is_dir():
        manifest_path = index_path / "current.json"
    else:
        manifest_path = index_path
    return json.loads(manifest_path.read_text(encoding="utf-8"))


__all__ = ["load_manifest"]

