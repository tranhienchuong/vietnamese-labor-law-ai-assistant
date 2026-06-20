from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..core.config import Settings


class SupabaseAuthError(RuntimeError):
    """Raised when a Supabase access token cannot be verified."""


@dataclass(frozen=True)
class SupabaseUser:
    id: str
    email: str
    name: str
    avatar_url: str | None


def _metadata_value(metadata: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def parse_supabase_user(payload: dict[str, Any]) -> SupabaseUser:
    user_id = str(payload.get("id") or payload.get("sub") or "").strip()
    email = str(payload.get("email") or "").strip().lower()
    metadata = payload.get("user_metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    name = _metadata_value(metadata, "full_name", "name", "display_name") or email
    avatar_url = _metadata_value(metadata, "avatar_url", "picture") or None

    if not user_id or not email:
        raise SupabaseAuthError("Supabase token did not include a user id and email.")

    return SupabaseUser(
        id=user_id,
        email=email,
        name=name,
        avatar_url=avatar_url,
    )


def verify_supabase_access_token(token: str, settings: Settings) -> SupabaseUser:
    supabase_url = settings.supabase_url.strip().rstrip("/")
    anon_key = settings.optional_secret_value(settings.supabase_anon_key)
    if not supabase_url or not anon_key:
        raise SupabaseAuthError("Supabase authentication is not configured.")

    request = Request(
        f"{supabase_url}/auth/v1/user",
        headers={
            "apikey": anon_key,
            "Authorization": f"Bearer {token}",
        },
        method="GET",
    )

    try:
        with urlopen(request, timeout=10) as response:
            raw_body = response.read().decode("utf-8")
    except HTTPError as exc:
        raise SupabaseAuthError("Invalid Supabase session.") from exc
    except (OSError, URLError) as exc:
        raise SupabaseAuthError("Could not verify Supabase session.") from exc

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise SupabaseAuthError("Supabase user response was not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise SupabaseAuthError("Supabase user response was invalid.")

    return parse_supabase_user(payload)


__all__ = [
    "SupabaseAuthError",
    "SupabaseUser",
    "parse_supabase_user",
    "verify_supabase_access_token",
]
