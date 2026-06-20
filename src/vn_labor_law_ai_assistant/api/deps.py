from __future__ import annotations

from threading import Lock
from typing import cast

from fastapi import Depends, Header, HTTPException

from ..auth.models import Role
from ..auth.supabase import SupabaseAuthError, verify_supabase_access_token
from ..auth_store import AuthStore, AuthUser
from ..core.config import get_settings
from ..retriever import DEFAULT_RERANKER_TOP_N, HybridRetriever
from ..supabase_store import SupabaseAppStore


_retriever: HybridRetriever | None = None
_retriever_lock = Lock()
_auth_store: AuthStore | None = None
_auth_store_lock = Lock()
_app_store: AuthStore | SupabaseAppStore | None = None
_app_store_backend: str | None = None
_app_store_lock = Lock()


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is not None:
        return _retriever

    with _retriever_lock:
        if _retriever is None:
            settings = get_settings()
            _retriever = HybridRetriever(
                index_path=settings.index_path,
                reranker_model=settings.reranker_model.strip(),
                reranker_top_n=max(
                    1,
                    int(settings.reranker_top_n or DEFAULT_RERANKER_TOP_N),
                ),
            )
    return _retriever


def close_retriever() -> None:
    if _retriever is not None:
        _retriever.close()


def get_auth_store() -> AuthStore:
    global _auth_store
    if _auth_store is not None:
        return _auth_store

    with _auth_store_lock:
        if _auth_store is None:
            _auth_store = AuthStore()
    return _auth_store


def get_app_store() -> AuthStore | SupabaseAppStore:
    global _app_store, _app_store_backend
    settings = get_settings()
    backend = settings.app_data_backend
    if backend == "sqlite":
        return get_auth_store()

    if _app_store is not None and _app_store_backend == backend:
        return _app_store

    with _app_store_lock:
        if _app_store is None or _app_store_backend != backend:
            if backend == "supabase":
                _app_store = SupabaseAppStore(settings=settings)
                _app_store_backend = backend
            else:
                _app_store = get_auth_store()
                _app_store_backend = "sqlite"
    return _app_store


def extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        return ""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return ""
    return token.strip()


def require_current_user(authorization: str | None = Header(default=None)) -> AuthUser:
    token = extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required.")

    settings = get_settings()
    if settings.auth_provider == "supabase":
        try:
            supabase_user = verify_supabase_access_token(token, settings)
        except SupabaseAuthError as exc:
            raise HTTPException(status_code=401, detail="Invalid or expired session.") from exc

        role = cast(Role, settings.role_for_email(supabase_user.email))
        return get_app_store().upsert_external_user(
            user_id=supabase_user.id,
            name=supabase_user.name,
            email=supabase_user.email,
            auth_provider="supabase",
            provider_id=supabase_user.id,
            role=role,
            avatar_url=supabase_user.avatar_url,
        )

    user = get_auth_store().get_user_by_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")
    return user


def require_admin_user(current_user: AuthUser = Depends(require_current_user)) -> AuthUser:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required.")
    return current_user
