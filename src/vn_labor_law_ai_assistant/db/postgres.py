from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from ..core.config import Settings, load_settings


class SupabasePostgresDatabase:
    def __init__(self, dsn: str | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.dsn = dsn or self.settings.optional_secret_value(self.settings.supabase_db_url)
        if not self.dsn:
            raise RuntimeError("SUPABASE_DB_URL is required when APP_DATA_BACKEND=supabase.")
        self._validate_dsn()

    def connect(self) -> Any:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "psycopg[binary] is required for Supabase Postgres app storage."
            ) from exc
        return psycopg.connect(self.dsn, row_factory=dict_row)

    def _validate_dsn(self) -> None:
        if "[" in self.dsn or "]" in self.dsn:
            raise RuntimeError(
                "SUPABASE_DB_URL appears to contain square-bracket placeholder text. "
                "Remove placeholder brackets and URL-encode special characters in the database password."
            )
        try:
            parsed = urlsplit(self.dsn)
        except ValueError as exc:
            raise RuntimeError(
                "SUPABASE_DB_URL is not a valid PostgreSQL connection URL. "
                "URL-encode special characters in the database password."
            ) from exc
        if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
            raise RuntimeError("SUPABASE_DB_URL must be a valid PostgreSQL connection URL.")
