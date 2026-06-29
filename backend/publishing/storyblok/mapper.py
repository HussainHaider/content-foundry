"""
backend/publishing/storyblok/mapper.py

Map a generated blog piece (topic + markdown draft) onto a Storyblok story
payload: the leading H1 becomes the story name + deterministic slug, and the
remaining markdown becomes a `page.body` of `text` bloks.

Deterministic slugs make re-publishing idempotent (the service updates the same
story instead of creating duplicates).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from backend.publishing.storyblok.config import StoryblokConfig
from backend.publishing.storyblok.markdown_blocks import markdown_to_text_bloks
from backend.publishing.storyblok.schema import ResolvedSchema

_H1_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class StoryDraft:
    slug: str
    name: str
    story: dict  # the "story" object for the Management API request body


def slugify(text: str, max_len: int = 60) -> str:
    slug = _SLUG_STRIP.sub("-", (text or "").lower()).strip("-")
    return slug[:max_len].strip("-") or "untitled"


def split_h1(draft: str) -> tuple[str | None, str]:
    """Return (title from first H1 or None, body markdown with that H1 removed)."""
    match = _H1_RE.search(draft or "")
    if not match:
        return None, (draft or "").strip()
    title = match.group(1).strip()
    body = (draft[: match.start()] + draft[match.end():]).strip()
    return title, body


def piece_to_story(
    *,
    topic: str,
    draft: str,
    week: int,
    schema: ResolvedSchema,
    config: StoryblokConfig,
) -> StoryDraft:
    h1_title, body_md = split_h1(draft)
    name = (h1_title or topic or "Untitled").strip()
    slug = slugify(name)

    bloks = markdown_to_text_bloks(body_md, schema)
    story: dict = {
        "name": name,
        "slug": slug,
        "tag_list": ["blog", f"week{week}"],
        "content": {
            "component": schema.root_component,
            schema.body_field: bloks,
        },
        "is_startpage": False,
    }
    if config.blog_parent_id:
        try:
            story["parent_id"] = int(config.blog_parent_id)
        except ValueError:
            story["parent_id"] = config.blog_parent_id

    return StoryDraft(slug=slug, name=name, story=story)
