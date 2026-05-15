from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import load_repo_env
from .deps import close_retriever
from .routes import admin, auth, chat, conversations, health


load_repo_env()


def create_app() -> FastAPI:
    app = FastAPI(title="Vietnamese Labor Law AI Assistant")

    allow_origins = [
        origin.strip()
        for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(conversations.router)
    app.include_router(chat.router)
    app.include_router(admin.router)

    return app


app = create_app()


@app.on_event("shutdown")
def shutdown() -> None:
    close_retriever()
