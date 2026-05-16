from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any, TYPE_CHECKING

from .config import load_settings

if TYPE_CHECKING:
    from ..auth.models import AuthUser


DEFAULT_SESSION_TTL_SECONDS = 60 * 60 * 24 * 7
PASSWORD_ITERATIONS = 260_000


def is_production() -> bool:
    return load_settings().is_production


def _b64_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _token_secret() -> bytes:
    return load_settings().require_auth_secret().encode("utf-8")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    return "pbkdf2_sha256${}${}${}".format(
        PASSWORD_ITERATIONS,
        _b64_encode(salt),
        _b64_encode(digest),
    )


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations_text, salt_text, digest_text = password_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_text)
        salt = _b64_decode(salt_text)
        expected = _b64_decode(digest_text)
    except (ValueError, TypeError):
        return False

    observed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(observed, expected)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_access_token(
    *,
    session_id: str,
    user: AuthUser,
    ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
) -> tuple[str, int]:
    expires_at = int(time.time()) + ttl_seconds
    payload = {
        "sid": session_id,
        "sub": user.id,
        "email": user.email,
        "name": user.name,
        "role": user.role,
        "exp": expires_at,
    }
    encoded_payload = _b64_encode(_json_dumps(payload).encode("utf-8"))
    signature = hmac.new(_token_secret(), encoded_payload.encode("ascii"), hashlib.sha256)
    token = "{}.{}".format(encoded_payload, _b64_encode(signature.digest()))
    return token, expires_at


def decode_and_verify_token(token: str) -> dict[str, Any] | None:
    try:
        payload_text, signature_text = token.split(".", 1)
    except ValueError:
        return None

    expected = hmac.new(_token_secret(), payload_text.encode("ascii"), hashlib.sha256)
    try:
        observed = _b64_decode(signature_text)
    except (ValueError, TypeError):
        return None
    if not hmac.compare_digest(observed, expected.digest()):
        return None

    try:
        payload = json.loads(_b64_decode(payload_text).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None
    if int(payload.get("exp") or 0) < int(time.time()):
        return None
    return payload if isinstance(payload, dict) else None
