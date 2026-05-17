from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vn_labor_law_ai_assistant.core.config import load_repo_env, load_settings
from vn_labor_law_ai_assistant.indexing import (
    QDRANT_KEYWORD_PAYLOAD_INDEX_FIELDS,
    build_qdrant_client,
    require_qdrant,
)
from vn_labor_law_ai_assistant.rag.retrieval.manifest import load_manifest


EXTRA_CANDIDATE_FIELDS = (
    "article_id",
    "law_id",
    "reference",
    "metadata.article_id",
    "metadata.article_number",
    "metadata.document_id",
    "metadata.law_id",
    "metadata.reference",
    "metadata.chunk_type",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Qdrant keyword payload indexes used by retrieval filters."
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=REPO_ROOT / "artifacts" / "index",
        help="Path to artifacts/index or current.json; used for local Qdrant path fallback.",
    )
    parser.add_argument(
        "--collection-name",
        default="",
        help="Qdrant collection name. Defaults to QDRANT_COLLECTION or index manifest.",
    )
    return parser.parse_args()


def payload_has_path(payload: dict[str, object], field_name: str) -> bool:
    current: object = payload
    for part in field_name.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def index_already_exists_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "already exists" in message or "same field" in message


def main() -> None:
    load_repo_env()
    args = parse_args()
    settings = load_settings()
    manifest = load_manifest(args.index_path)
    collection_name = (
        args.collection_name.strip()
        or settings.qdrant_collection.strip()
        or str(manifest.get("collection_name") or "").strip()
    )
    if not collection_name:
        raise SystemExit("No Qdrant collection name configured.")

    qdrant_client_cls, models = require_qdrant()
    qdrant_path = None if settings.qdrant_url.strip() else Path(str(manifest["qdrant_path"]))
    client = build_qdrant_client(qdrant_client_cls, qdrant_path)
    try:
        points, _ = client.scroll(
            collection_name=collection_name,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        sample_payload = dict(points[0].payload or {}) if points else {}
        candidate_fields = tuple(
            dict.fromkeys((*QDRANT_KEYWORD_PAYLOAD_INDEX_FIELDS, *EXTRA_CANDIDATE_FIELDS))
        )
        fields = [
            field_name
            for field_name in candidate_fields
            if field_name in QDRANT_KEYWORD_PAYLOAD_INDEX_FIELDS
            or payload_has_path(sample_payload, field_name)
        ]

        for field_name in fields:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            except Exception as exc:
                if index_already_exists_error(exc):
                    print(f"Payload index already exists: {field_name}")
                    continue
                raise
            print(f"Payload index ready: {field_name}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
