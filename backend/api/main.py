"""
backend/api/main.py

FastAPI publishing service. Owns the Storyblok Management token (it is the only
process that reads STORYBLOK_MANAGEMENT_TOKEN) and exposes:

  GET  /health             → liveness + whether Storyblok is configured
  POST /publish/storyblok  → publish blog pieces as draft stories (X-API-Key auth)

Run locally:
  uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException

from backend.api.models import (
    HealthResponse,
    PublishRequest,
    PublishResponse,
)
from backend.publishing.storyblok.client import StoryblokError
from backend.publishing.storyblok.config import StoryblokConfig
from backend.publishing.storyblok.schema import SchemaError
from backend.publishing.storyblok.service import publish_blogs

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("storyblok.api")

app = FastAPI(title="Content Foundry — Storyblok Publisher", version="1.0.0")


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Authenticate the internal caller (the Streamlit app) via a shared key."""
    expected = os.environ.get("PUBLISHER_API_KEY", "").strip()
    if not expected:
        # Fail closed: publishing is disabled until an internal key is configured.
        raise HTTPException(
            status_code=503, detail="Publishing API key not configured."
        )
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        storyblok_configured=StoryblokConfig.from_env().is_configured(),
    )


@app.post(
    "/publish/storyblok",
    response_model=PublishResponse,
    dependencies=[Depends(require_api_key)],
)
def publish_storyblok(request: PublishRequest) -> PublishResponse:
    pieces = [p.model_dump() for p in request.pieces]
    try:
        results = publish_blogs(pieces, publish=request.publish)
    except RuntimeError as exc:  # not configured
        logger.warning("publish rejected: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except SchemaError as exc:  # space contract mismatch
        logger.warning("publish schema error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except StoryblokError as exc:  # upstream Management API failure
        logger.error("storyblok upstream error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))
    return PublishResponse(results=results)
