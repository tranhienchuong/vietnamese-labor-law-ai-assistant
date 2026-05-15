from __future__ import annotations

from fastapi import APIRouter, Depends

from ...auth_store import AuthUser, user_payload
from ..deps import get_auth_store, require_admin_user


router = APIRouter()


@router.get("/admin/stats")
def admin_stats(current_user: AuthUser = Depends(require_admin_user)):
    store = get_auth_store()
    return {
        "user": user_payload(current_user),
        "conversations": len(store.list_conversations(user_id=current_user.id)),
    }
