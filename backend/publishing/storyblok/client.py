"""
backend/publishing/storyblok/client.py

Thin httpx wrapper around the Storyblok Management API with bounded
retry/backoff and token-redacted logging.

Retry policy: retry only 429 / 5xx (honoring Retry-After); never retry 4xx
(those are bugs to surface). Accept 200 and 201 as success on create.
"""

from __future__ import annotations

import logging
import time

import httpx

from backend.publishing.storyblok.config import StoryblokConfig

logger = logging.getLogger("storyblok.client")

_TIMEOUT = httpx.Timeout(15.0)
_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 0.5  # seconds


class StoryblokError(RuntimeError):
    """Raised when the Management API cannot satisfy a request after retries."""


class StoryblokManagementClient:
    def __init__(self, config: StoryblokConfig, client: httpx.Client | None = None):
        self.config = config
        # Allow dependency injection of an httpx.Client for testing.
        self._client = client or httpx.Client(timeout=_TIMEOUT)

    # ── public API ───────────────────────────────────────────────────────────
    def get_components(self) -> list[dict]:
        """GET /components → list of component schema dicts."""
        resp = self._request("GET", f"{self.config.spaces_url}/components")
        return resp.json().get("components", [])

    def find_story_by_slug(self, slug: str) -> dict | None:
        """Return the existing story dict for a slug, or None. Used for idempotency."""
        resp = self._request(
            "GET",
            f"{self.config.spaces_url}/stories",
            params={"with_slug": slug},
        )
        stories = resp.json().get("stories", [])
        return stories[0] if stories else None

    def create_story(self, payload: dict) -> dict:
        """POST /stories → created story dict (accepts 200/201)."""
        resp = self._request("POST", f"{self.config.spaces_url}/stories", json=payload)
        return resp.json().get("story", {})

    def update_story(self, story_id: int | str, payload: dict) -> dict:
        """PUT /stories/{id} → updated story dict."""
        resp = self._request(
            "PUT", f"{self.config.spaces_url}/stories/{story_id}", json=payload
        )
        return resp.json().get("story", {})

    def close(self) -> None:
        self._client.close()

    # ── internals ────────────────────────────────────────────────────────────
    @property
    def _headers(self) -> dict:
        # Management API uses the raw token in Authorization (NOT a Bearer prefix).
        return {
            "Authorization": self.config.management_token,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                resp = self._client.request(
                    method, url, headers=self._headers, **kwargs
                )
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning(
                    "storyblok %s %s transport error (attempt %d/%d): %s",
                    method,
                    _safe_url(url),
                    attempt,
                    _MAX_ATTEMPTS,
                    exc,
                )
                self._sleep(attempt)
                continue

            if resp.status_code in (200, 201):
                return resp

            # Retry only on rate-limit / server errors.
            if resp.status_code == 429 or resp.status_code >= 500:
                logger.warning(
                    "storyblok %s %s -> %d (attempt %d/%d)",
                    method,
                    _safe_url(url),
                    resp.status_code,
                    attempt,
                    _MAX_ATTEMPTS,
                )
                self._sleep(attempt, resp.headers.get("Retry-After"))
                last_exc = StoryblokError(
                    f"{method} {_safe_url(url)} failed: {resp.status_code}"
                )
                continue

            # 4xx (other than 429): do not retry — surface immediately.
            raise StoryblokError(
                f"{method} {_safe_url(url)} failed: {resp.status_code} {resp.text[:300]}"
            )

        raise StoryblokError(
            f"{method} {_safe_url(url)} failed after {_MAX_ATTEMPTS} attempts"
        ) from last_exc

    @staticmethod
    def _sleep(attempt: int, retry_after: str | None = None) -> None:
        if attempt >= _MAX_ATTEMPTS:
            return
        if retry_after:
            try:
                time.sleep(min(float(retry_after), 30.0))
                return
            except ValueError:
                pass
        time.sleep(_BACKOFF_BASE * (2 ** (attempt - 1)))


def _safe_url(url: str) -> str:
    """URLs contain no secrets (token is a header), but keep logs tidy."""
    return url
