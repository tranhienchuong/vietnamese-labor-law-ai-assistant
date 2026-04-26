from __future__ import annotations

import os
from typing import Union

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)

app = FastAPI(title="Free Embedding API")

try:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL = SentenceTransformer(MODEL_NAME, device=DEVICE)
except Exception as exc:  # pragma: no cover - startup failure is surfaced by the endpoint.
    print(f"Failed to load embedding model: {exc}")
    DEVICE = "unavailable"
    MODEL = None


class EmbedRequest(BaseModel):
    input: Union[str, list[str]]


class EmbedResponse(BaseModel):
    embedding: Union[list[float], list[list[float]]]
    model: str


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "status": "ok" if MODEL is not None else "model_unavailable",
        "model": MODEL_NAME,
        "device": DEVICE,
    }


@app.post("/v1/embeddings", response_model=EmbedResponse)
async def get_embeddings(request: EmbedRequest) -> EmbedResponse:
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Embedding model was not loaded.")

    sentences = request.input
    is_single_input = isinstance(sentences, str)
    if is_single_input:
        sentences = [sentences]

    sentences = [sentence for sentence in sentences if sentence.strip()]
    if not sentences:
        raise HTTPException(status_code=400, detail="Input must not be empty.")

    try:
        embeddings = MODEL.encode(sentences, normalize_embeddings=True)
        embeddings_list = embeddings.tolist()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}") from exc

    if is_single_input:
        embeddings_list = embeddings_list[0]

    return EmbedResponse(embedding=embeddings_list, model=MODEL_NAME)
