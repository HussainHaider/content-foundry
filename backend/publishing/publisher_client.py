"""
backend/publishing/publisher_client.py

Client the Streamlit app uses to reach the FastAPI publishing service over HTTP.
Knows only PUBLISHER_API_URL + PUBLISHER_API_KEY — never the Storyblok token.
"""

from __future__ import annotations

import os

import httpx


def _base_url() -> str:
    return os.environ.get("PUBLISHER_API_URL", "http://localhost:8000").rstrip("/")


def _headers() -> dict:
    return {"X-API-Key": os.environ.get("PUBLISHER_API_KEY", "")}


def check_health(timeout: float = 3.0) -> dict | None:
    """Return the /health body, or None if the service is unreachable."""
    try:
        resp = httpx.get(f"{_base_url()}/health", timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except httpx.HTTPError:
        return None
    return None


def publish_blogs(pieces: list[dict], publish: bool = False, timeout: float = 60.0) -> dict:
    """
    Call POST /publish/storyblok.

    Returns {"results": [...]} on success. Raises RuntimeError with a readable
    message on auth/upstream errors so the UI can surface it.
    """
    resp = httpx.post(
        f"{_base_url()}/publish/storyblok",
        headers=_headers(),
        json={"pieces": pieces, "publish": publish},
        timeout=timeout,
    )
    if resp.status_code == 200:
        return resp.json()
    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        detail = resp.text
    raise RuntimeError(f"Publishing failed ({resp.status_code}): {detail}")
