from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ...admin.service import AdminService
from ...auth_store import AuthUser
from ..deps import get_app_store, require_admin_user


router = APIRouter()


@router.get("/admin/stats")
def admin_stats(current_user: AuthUser = Depends(require_admin_user)):
    store = get_app_store()
    return AdminService(store).get_stats(current_user)


@router.get("/admin/health")
def admin_health(current_user: AuthUser = Depends(require_admin_user)):
    store = get_app_store()
    return AdminService(store).get_health()


@router.get("/admin/retrieval-config")
def admin_retrieval_config(current_user: AuthUser = Depends(require_admin_user)):
    store = get_app_store()
    return AdminService(store).get_retrieval_config()


@router.get("/admin/traces")
def admin_traces(
    limit: int = Query(50, ge=1, le=200),
    user_id: str | None = Query(None, alias="userId"),
    conversation_id: str | None = Query(None, alias="conversationId"),
    insufficient_only: bool = Query(False, alias="insufficientOnly"),
    error_only: bool = Query(False, alias="errorOnly"),
    current_user: AuthUser = Depends(require_admin_user),
):
    store = get_app_store()
    return AdminService(store).list_recent_traces(
        limit=limit,
        user_id=user_id,
        conversation_id=conversation_id,
        insufficient_only=insufficient_only,
        error_only=error_only,
    )


@router.get("/admin/traces/{trace_id}")
def admin_trace_detail(
    trace_id: str,
    current_user: AuthUser = Depends(require_admin_user),
):
    store = get_app_store()
    trace = AdminService(store).get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found.")
    return trace
