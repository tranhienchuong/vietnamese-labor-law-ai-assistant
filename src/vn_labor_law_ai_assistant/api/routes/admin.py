from __future__ import annotations

from fastapi import APIRouter, Depends

from ...admin.service import AdminService
from ...auth_store import AuthUser
from ..deps import get_auth_store, require_admin_user


router = APIRouter()


@router.get("/admin/stats")
def admin_stats(current_user: AuthUser = Depends(require_admin_user)):
    store = get_auth_store()
    return AdminService(store).get_stats(current_user)


@router.get("/admin/health")
def admin_health(current_user: AuthUser = Depends(require_admin_user)):
    store = get_auth_store()
    return AdminService(store).get_health()


@router.get("/admin/retrieval-config")
def admin_retrieval_config(current_user: AuthUser = Depends(require_admin_user)):
    store = get_auth_store()
    return AdminService(store).get_retrieval_config()
