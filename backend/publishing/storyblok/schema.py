"""
backend/publishing/storyblok/schema.py

Discover Storyblok component schemas at runtime (GET /components) and validate
that the space exposes the contract we map onto: a `page`-style root with a
`bloks` field, and a `text`-style block with a text field + a `tag` field.

Resolving field names from discovery (rather than hardcoding) keeps App A from
depending on App B's source, and tolerates renamed fields.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("storyblok.schema")

# Preferred names; discovery falls back to the first field of the right type.
_ROOT_CANDIDATES = ("page",)
_TEXT_CANDIDATES = ("text",)
_BODY_FIELD_CANDIDATES = ("body", "content", "blocks")
_TEXT_FIELD_CANDIDATES = ("text", "content", "body")
_TAG_FIELD_CANDIDATES = ("tag", "level", "headline_tag")
_ALIGN_FIELD_CANDIDATES = ("align", "alignment")


class SchemaError(RuntimeError):
    """Raised when the space does not expose the components/fields we need."""


@dataclass(frozen=True)
class ResolvedSchema:
    root_component: str  # e.g. "page"
    body_field: str  # bloks field on the root, e.g. "body"
    text_component: str  # e.g. "text"
    text_field: str  # string field on text, e.g. "text"
    tag_field: str  # option/text field for h1-h6/p, e.g. "tag"
    align_field: str | None  # optional alignment field


def _index(components: list[dict]) -> dict[str, dict]:
    return {c.get("name", ""): (c.get("schema") or {}) for c in components}


def _pick(
    schema: dict, candidates: tuple[str, ...], *, field_type: str | None = None
) -> str | None:
    for name in candidates:
        field = schema.get(name)
        if field is not None and (
            field_type is None or field.get("type") == field_type
        ):
            return name
    if field_type is not None:
        for name, field in schema.items():
            if isinstance(field, dict) and field.get("type") == field_type:
                return name
    return None


def validate_contract(components: list[dict]) -> ResolvedSchema:
    """Resolve the component/field names we publish into, or raise SchemaError."""
    by_name = _index(components)

    root = next((c for c in _ROOT_CANDIDATES if c in by_name), None)
    if root is None:
        raise SchemaError(
            f"No root container component found (looked for {_ROOT_CANDIDATES}). "
            f"Available: {sorted(by_name)}"
        )
    body_field = _pick(by_name[root], _BODY_FIELD_CANDIDATES, field_type="bloks")
    if body_field is None:
        raise SchemaError(
            f"Component '{root}' has no 'bloks' field to hold the article body."
        )

    text_comp = next((c for c in _TEXT_CANDIDATES if c in by_name), None)
    if text_comp is None:
        raise SchemaError(
            f"No text block component found (looked for {_TEXT_CANDIDATES}). "
            f"Available: {sorted(by_name)}"
        )
    text_schema = by_name[text_comp]
    text_field = _pick(text_schema, _TEXT_FIELD_CANDIDATES)
    if text_field is None:
        raise SchemaError(f"Component '{text_comp}' has no text field.")
    tag_field = _pick(text_schema, _TAG_FIELD_CANDIDATES)
    if tag_field is None:
        raise SchemaError(
            f"Component '{text_comp}' has no 'tag' field for headings/paragraphs."
        )
    align_field = _pick(text_schema, _ALIGN_FIELD_CANDIDATES)

    resolved = ResolvedSchema(
        root_component=root,
        body_field=body_field,
        text_component=text_comp,
        text_field=text_field,
        tag_field=tag_field,
        align_field=align_field,
    )
    logger.info("storyblok schema resolved: %s", resolved)
    return resolved
