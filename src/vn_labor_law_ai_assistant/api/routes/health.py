from __future__ import annotations

from hmac import compare_digest

from fastapi import APIRouter, Header, HTTPException, status
from neo4j import GraphDatabase

from ...core.config import get_settings


router = APIRouter()


@router.get("/")
def root() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/internal/warmup/neo4j")
def warmup_neo4j(x_warmup_token: str = Header(default="")) -> dict[str, str]:
    settings = get_settings()
    expected_token = settings.optional_secret_value(settings.warmup_token)

    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Warmup token is not configured.",
        )

    if not compare_digest(x_warmup_token, expected_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized.")

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.optional_secret_value(settings.neo4j_password)),
    )

    try:
        with driver.session(database=settings.neo4j_database) as session:
            session.run("RETURN 1 AS ok").consume()
    finally:
        driver.close()

    return {"status": "ok", "service": "neo4j"}
