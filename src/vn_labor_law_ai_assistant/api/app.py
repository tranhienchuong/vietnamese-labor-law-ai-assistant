from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..core.config import get_settings, load_repo_env
from .deps import close_retriever
from .routes import admin, auth, chat, conversations, health


load_repo_env()
settings = get_settings()


def create_app() -> FastAPI:
    app = FastAPI(title="Vietnamese Labor Law AI Assistant")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list(),
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
