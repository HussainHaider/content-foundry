"""
backend/publishing/storyblok/service.py

Orchestrates publishing generated blogs to Storyblok:
  1. discover + validate the space's component schema once,
  2. map each blog to a story payload,
  3. create-or-update per slug (idempotent), isolating per-piece failures.

This is plain Python invoked by the FastAPI service — NOT a LangGraph node.
"""

from __future__ import annotations

import logging

from backend.publishing.storyblok.client import StoryblokManagementClient
from backend.publishing.storyblok.config import StoryblokConfig
from backend.publishing.storyblok.mapper import piece_to_story
from backend.publishing.storyblok.schema import validate_contract

logger = logging.getLogger("storyblok.service")


def publish_blogs(
    pieces: list[dict],
    *,
    publish: bool = False,
    config: StoryblokConfig | None = None,
    client: StoryblokManagementClient | None = None,
) -> list[dict]:
    """
    Publish each blog piece (dict with keys: topic, draft, week) to Storyblok.

    Returns one result dict per piece:
      {topic, status: created|updated|error, story_id, url, full_slug, error}

    Raises on global failures (not configured / schema discovery) so the caller
    can surface a single clear error; per-piece failures are captured in results.
    """
    config = config or StoryblokConfig.from_env()
    if not config.is_configured():
        raise RuntimeError(
            "Storyblok is not configured (set STORYBLOK_MANAGEMENT_TOKEN and "
            "STORYBLOK_SPACE_ID)."
        )

    owns_client = client is None
    client = client or StoryblokManagementClient(config)
    try:
        schema = validate_contract(client.get_components())

        results: list[dict] = []
        for idx, piece in enumerate(pieces):
            topic = piece.get("topic", f"Blog {idx + 1}")
            week = piece.get("week", idx + 1)
            result = {
                "topic": topic,
                "status": "error",
                "story_id": None,
                "url": None,
                "full_slug": None,
                "error": None,
            }
            try:
                draft = piece_to_story(
                    topic=topic,
                    draft=piece.get("draft", ""),
                    week=week,
                    schema=schema,
                    config=config,
                )
                body = {"story": draft.story, "publish": 1 if publish else 0}

                existing = client.find_story_by_slug(draft.slug)
                if existing:
                    story_obj = client.update_story(existing["id"], body)
                    result["status"] = "updated"
                else:
                    story_obj = client.create_story(body)
                    result["status"] = "created"

                sid = story_obj.get("id")
                result["story_id"] = sid
                result["full_slug"] = story_obj.get("full_slug") or draft.slug
                result["url"] = config.editor_url(sid) if sid else None
                logger.info(
                    "storyblok %s story %s (%s)", result["status"], sid, draft.slug
                )
            except Exception as exc:  # isolate per-piece failures
                result["status"] = "error"
                result["error"] = str(exc)
                logger.warning("storyblok publish failed for '%s': %s", topic, exc)

            results.append(result)
        return results
    finally:
        if owns_client:
            client.close()
