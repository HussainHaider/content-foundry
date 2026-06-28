"""
backend/api/models.py

Pydantic request/response models for the Storyblok publishing endpoint.
The Streamlit app sends only content (topic + markdown) — never the token.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BlogPiece(BaseModel):
    topic: str = Field(..., min_length=1)
    draft: str = Field(..., min_length=1)
    week: int = Field(default=1, ge=1)


class PublishRequest(BaseModel):
    pieces: list[BlogPiece] = Field(..., min_length=1)
    publish: bool = False  # False = create as draft (publish:0)


class PublishResult(BaseModel):
    topic: str
    status: str  # created | updated | error
    story_id: int | None = None
    url: str | None = None
    full_slug: str | None = None
    error: str | None = None


class PublishResponse(BaseModel):
    results: list[PublishResult]


class HealthResponse(BaseModel):
    status: str
    storyblok_configured: bool
