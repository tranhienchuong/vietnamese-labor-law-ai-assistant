from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ...auth_store import AuthUser
from ..deps import get_auth_store, require_current_user


router = APIRouter()


@router.get("/conversations")
def list_conversations(current_user: AuthUser = Depends(require_current_user)):
    return {
        "conversations": get_auth_store().list_conversations(user_id=current_user.id)
    }


@router.post("/conversations")
async def create_conversation(
    request: Request,
    current_user: AuthUser = Depends(require_current_user),
):
    payload = await request.json()
    title = str(payload.get("title") or "Cuá»™c trÃ² chuyá»‡n má»›i")
    conversation = get_auth_store().create_conversation(
        user_id=current_user.id,
        title=title,
    )
    return {"conversation": conversation}


@router.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str,
    current_user: AuthUser = Depends(require_current_user),
):
    store = get_auth_store()
    conversation = store.get_conversation(
        user_id=current_user.id,
        conversation_id=conversation_id,
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    messages = store.list_messages(
        user_id=current_user.id,
        conversation_id=conversation_id,
    )
    return {"conversation": conversation, "messages": messages or []}


@router.get("/conversations/{conversation_id}/messages")
def list_messages(
    conversation_id: str,
    current_user: AuthUser = Depends(require_current_user),
):
    messages = get_auth_store().list_messages(
        user_id=current_user.id,
        conversation_id=conversation_id,
    )
    if messages is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"messages": messages}
