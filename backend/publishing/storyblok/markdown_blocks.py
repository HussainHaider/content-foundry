"""
backend/publishing/storyblok/markdown_blocks.py

Convert a blog's markdown draft into an ordered list of Storyblok `text` bloks
that slot into a `page.body` field. We reuse App B's existing `text` component
(string + tag h1-h6/p), so inline formatting (bold/italic/links) and native
list semantics are intentionally flattened to plain text.

Field/component names come from the resolved schema, never hardcoded.
"""

from __future__ import annotations

from markdown_it import MarkdownIt

from backend.publishing.storyblok.schema import ResolvedSchema


def _heading_tag(level: int) -> str:
    """Clamp to h2-h6 — the leading H1 is consumed as the story title upstream."""
    return f"h{min(max(level, 2), 6)}"


def _plain(inline_token) -> str:
    """Flatten an inline token's children to plain text (drops marks/link URLs)."""
    if inline_token is None:
        return ""
    if inline_token.type != "inline":
        return (getattr(inline_token, "content", "") or "").strip()
    parts: list[str] = []
    for child in inline_token.children or []:
        if child.type in ("text", "code_inline"):
            parts.append(child.content)
        elif child.type in ("softbreak", "hardbreak"):
            parts.append(" ")
    return "".join(parts).strip()


def _text_blok(schema: ResolvedSchema, tag: str, text: str) -> dict:
    blok = {
        "component": schema.text_component,
        schema.text_field: text,
        schema.tag_field: tag,
    }
    if schema.align_field:
        blok[schema.align_field] = "left"
    return blok


def markdown_to_text_bloks(md_text: str, schema: ResolvedSchema) -> list[dict]:
    md = MarkdownIt("commonmark")
    tokens = md.parse(md_text or "")
    bloks: list[dict] = []
    list_stack: list[dict] = []   # {"ordered": bool, "counter": int}
    pending_prefix: str | None = None

    i, n = 0, len(tokens)
    while i < n:
        t = tokens[i]
        ttype = t.type

        if ttype == "heading_open":
            level = int(t.tag[1:]) if t.tag[1:].isdigit() else 2
            text = _plain(tokens[i + 1]) if i + 1 < n else ""
            if text:
                bloks.append(_text_blok(schema, _heading_tag(level), text))
            i += 3
            continue

        if ttype in ("bullet_list_open", "ordered_list_open"):
            ordered = ttype == "ordered_list_open"
            start = 1
            if ordered:
                try:
                    start = int(t.attrs.get("start", 1))
                except (TypeError, ValueError):
                    start = 1
            list_stack.append({"ordered": ordered, "counter": start})
            i += 1
            continue

        if ttype in ("bullet_list_close", "ordered_list_close"):
            if list_stack:
                list_stack.pop()
            i += 1
            continue

        if ttype == "list_item_open":
            if list_stack:
                ctx = list_stack[-1]
                if ctx["ordered"]:
                    pending_prefix = f"{ctx['counter']}. "
                    ctx["counter"] += 1
                else:
                    pending_prefix = "• "
            i += 1
            continue

        if ttype == "paragraph_open":
            text = _plain(tokens[i + 1]) if i + 1 < n else ""
            if pending_prefix is not None:
                text = f"{pending_prefix}{text}".strip()
                pending_prefix = None
            if text:
                bloks.append(_text_blok(schema, "p", text))
            i += 3
            continue

        if ttype in ("fence", "code_block"):
            text = (t.content or "").strip()
            if text:
                bloks.append(_text_blok(schema, "p", text))
            i += 1
            continue

        i += 1

    return bloks
