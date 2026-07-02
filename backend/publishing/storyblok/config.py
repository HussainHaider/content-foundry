"""
backend/publishing/storyblok/config.py

Environment-driven configuration for the Storyblok Management API.
Read at construction time (not import time) so the FastAPI process can be
started without these vars and report `storyblok_configured: false` via /health
instead of crashing — mirrors the optional-publishing pattern in
backend/agents/publisher.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Storyblok Management API host per space region.
# https://www.storyblok.com/docs/api/management/getting-started/region-parameter
_REGION_HOSTS = {
    "eu": "https://mapi.storyblok.com/v1",
    "us": "https://api-us.storyblok.com/v1",
    "ap": "https://api-ap.storyblok.com/v1",
    "ca": "https://api-ca.storyblok.com/v1",
    "cn": "https://app.storyblokchina.cn/v1",
}

# Storyblok visual-editor base, for building a human-friendly link back to a story.
_EDITOR_BASE = "https://app.storyblok.com"


@dataclass(frozen=True)
class StoryblokConfig:
    management_token: str
    space_id: str
    region: str = "eu"
    blog_parent_id: str | None = None

    @classmethod
    def from_env(cls) -> StoryblokConfig:
        return cls(
            management_token=os.environ.get("STORYBLOK_MANAGEMENT_TOKEN", "").strip(),
            space_id=os.environ.get("STORYBLOK_SPACE_ID", "").strip(),
            region=(os.environ.get("STORYBLOK_REGION", "eu").strip().lower() or "eu"),
            blog_parent_id=(
                os.environ.get("STORYBLOK_BLOG_PARENT_ID", "").strip() or None
            ),
        )

    def is_configured(self) -> bool:
        return bool(self.management_token and self.space_id)

    @property
    def base_url(self) -> str:
        return _REGION_HOSTS.get(self.region, _REGION_HOSTS["eu"])

    @property
    def spaces_url(self) -> str:
        return f"{self.base_url}/spaces/{self.space_id}"

    def editor_url(self, story_id: int | str) -> str:
        """Visual-editor link a human can open to review the draft."""
        return f"{_EDITOR_BASE}/#/me/spaces/{self.space_id}/stories/0/0/{story_id}"
