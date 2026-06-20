from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse

from ...auth_store import AuthUser, user_payload
from ...core.config import get_settings
from ..deps import extract_bearer_token, get_auth_store, require_current_user


router = APIRouter()


@router.post("/auth/login")
async def login(request: Request):
    if get_settings().auth_provider == "supabase":
        return JSONResponse(
            {"error": "Use Supabase Google OAuth to sign in."},
            status_code=400,
        )

    payload = await request.json()
    if not isinstance(payload, dict):
        return JSONResponse({"error": "Request body must be a JSON object."}, status_code=400)

    email = str(payload.get("email") or "").strip().lower()
    password = str(payload.get("password") or "")
    if not email or not password:
        return JSONResponse({"error": "Email and password are required."}, status_code=400)

    store = get_auth_store()
    user = store.authenticate_user(email, password)
    if user is None:
        return JSONResponse({"error": "Invalid email or password."}, status_code=401)

    access_token, expires_at = store.create_session(user)
    return {
        "accessToken": access_token,
        "expiresAt": expires_at,
        "user": user_payload(user),
    }


@router.post("/auth/logout")
def logout(
    authorization: str | None = Header(default=None),
    current_user: AuthUser = Depends(require_current_user),
) -> dict[str, bool]:
    token = extract_bearer_token(authorization)
    if token and get_settings().auth_provider == "local":
        get_auth_store().revoke_session(token)
    return {"ok": True}


@router.get("/auth/me")
def me(current_user: AuthUser = Depends(require_current_user)) -> dict[str, Any]:
    return {"user": user_payload(current_user)}
