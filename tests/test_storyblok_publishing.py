"""
tests/test_storyblok_publishing.py

Unit tests for the Storyblok publishing extension. No live network calls —
httpx is mocked. Does not import the LangGraph graph, so it collects without
LLM keys.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.publishing.storyblok.config import StoryblokConfig
from backend.publishing.storyblok.client import StoryblokManagementClient, StoryblokError
from backend.publishing.storyblok.markdown_blocks import markdown_to_text_bloks
from backend.publishing.storyblok.mapper import slugify, split_h1, piece_to_story
from backend.publishing.storyblok.schema import (
    ResolvedSchema,
    SchemaError,
    validate_contract,
)


@pytest.fixture
def schema() -> ResolvedSchema:
    return ResolvedSchema(
        root_component="page",
        body_field="body",
        text_component="text",
        text_field="text",
        tag_field="tag",
        align_field="align",
    )


@pytest.fixture
def config() -> StoryblokConfig:
    return StoryblokConfig(management_token="tok", space_id="123", region="eu")


def _resp(status, json_body=None, headers=None, text=""):
    return SimpleNamespace(
        status_code=status,
        json=lambda: (json_body or {}),
        headers=headers or {},
        text=text,
    )


# ── markdown → text bloks ─────────────────────────────────────────────────────
def test_markdown_headings_and_paragraphs(schema):
    md = "## Section\n\nA paragraph here.\n\n### Sub\n\nMore text."
    bloks = markdown_to_text_bloks(md, schema)
    assert [b["tag"] for b in bloks] == ["h2", "p", "h3", "p"]
    assert all(b["component"] == "text" and b["align"] == "left" for b in bloks)
    assert bloks[0]["text"] == "Section"
    assert bloks[1]["text"] == "A paragraph here."


def test_markdown_bullet_and_ordered_lists(schema):
    md = "- first\n- second\n\n1. one\n2. two"
    texts = [b["text"] for b in markdown_to_text_bloks(md, schema)]
    assert texts == ["• first", "• second", "1. one", "2. two"]


def test_markdown_inline_formatting_is_flattened(schema):
    md = "Some **bold** and [a link](https://x.com) inline."
    bloks = markdown_to_text_bloks(md, schema)
    assert bloks[0]["text"] == "Some bold and a link inline."


def test_h1_in_body_is_demoted_to_h2(schema):
    bloks = markdown_to_text_bloks("# Big title\n\ntext", schema)
    assert bloks[0]["tag"] == "h2"


# ── mapper ────────────────────────────────────────────────────────────────────
def test_split_h1_extracts_title_and_strips_it():
    title, body = split_h1("# My Title\n\nBody paragraph.")
    assert title == "My Title"
    assert "My Title" not in body
    assert body == "Body paragraph."


def test_split_h1_missing_returns_none():
    title, body = split_h1("No heading here.")
    assert title is None
    assert body == "No heading here."


def test_slugify_is_deterministic():
    assert slugify("Why AI Sounds Off-Brand!") == "why-ai-sounds-off-brand"
    assert slugify("Why AI Sounds Off-Brand!") == slugify("Why AI Sounds Off-Brand!")


def test_piece_to_story_shape(schema, config):
    draft = "# Week One Topic\n\n## Intro\n\nHello world."
    out = piece_to_story(
        topic="Week One Topic", draft=draft, week=1, schema=schema, config=config
    )
    assert out.slug == "week-1-week-one-topic"
    assert out.name == "Week One Topic"
    assert out.story["content"]["component"] == "page"
    body = out.story["content"]["body"]
    assert body[0]["tag"] == "h2" and body[0]["text"] == "Intro"


def test_piece_to_story_adds_parent_id_when_configured(schema):
    cfg = StoryblokConfig(management_token="t", space_id="1", blog_parent_id="42")
    out = piece_to_story(topic="T", draft="# T\n\nx", week=2, schema=schema, config=cfg)
    assert out.story["parent_id"] == 42


# ── schema discovery / validation ─────────────────────────────────────────────
def test_validate_contract_resolves_fields():
    components = [
        {"name": "page", "schema": {"body": {"type": "bloks"}}},
        {"name": "text", "schema": {"text": {"type": "text"}, "tag": {"type": "option"}, "align": {"type": "option"}}},
    ]
    resolved = validate_contract(components)
    assert resolved.root_component == "page"
    assert resolved.body_field == "body"
    assert resolved.text_component == "text"
    assert resolved.tag_field == "tag"
    assert resolved.align_field == "align"


def test_validate_contract_missing_text_raises():
    components = [{"name": "page", "schema": {"body": {"type": "bloks"}}}]
    with pytest.raises(SchemaError):
        validate_contract(components)


def test_validate_contract_missing_bloks_field_raises():
    components = [
        {"name": "page", "schema": {"title": {"type": "text"}}},
        {"name": "text", "schema": {"text": {"type": "text"}, "tag": {"type": "option"}}},
    ]
    with pytest.raises(SchemaError):
        validate_contract(components)


# ── client retry / branching ──────────────────────────────────────────────────
def test_client_sends_raw_auth_header(config):
    http = MagicMock()
    http.request.return_value = _resp(200, {"components": []})
    client = StoryblokManagementClient(config, client=http)
    client.get_components()
    _, kwargs = http.request.call_args
    assert kwargs["headers"]["Authorization"] == "tok"  # raw token, no "Bearer "


@patch("backend.publishing.storyblok.client.time.sleep", lambda *a: None)
def test_client_retries_on_429_then_succeeds(config):
    http = MagicMock()
    http.request.side_effect = [
        _resp(429, headers={"Retry-After": "0"}),
        _resp(200, {"components": [{"name": "page"}]}),
    ]
    client = StoryblokManagementClient(config, client=http)
    assert client.get_components() == [{"name": "page"}]
    assert http.request.call_count == 2


@patch("backend.publishing.storyblok.client.time.sleep", lambda *a: None)
def test_client_does_not_retry_on_4xx(config):
    http = MagicMock()
    http.request.return_value = _resp(401, text="unauthorized")
    client = StoryblokManagementClient(config, client=http)
    with pytest.raises(StoryblokError):
        client.get_components()
    assert http.request.call_count == 1  # no retry on 4xx


def test_client_find_story_by_slug(config):
    http = MagicMock()
    http.request.return_value = _resp(200, {"stories": [{"id": 9, "slug": "x"}]})
    client = StoryblokManagementClient(config, client=http)
    assert client.find_story_by_slug("x")["id"] == 9


# ── service create-vs-update ──────────────────────────────────────────────────
def test_service_creates_when_no_existing(schema, config):
    from backend.publishing.storyblok import service

    client = MagicMock()
    client.get_components.return_value = [
        {"name": "page", "schema": {"body": {"type": "bloks"}}},
        {"name": "text", "schema": {"text": {"type": "text"}, "tag": {"type": "option"}}},
    ]
    client.find_story_by_slug.return_value = None
    client.create_story.return_value = {"id": 5, "full_slug": "blog/week-1-t"}

    results = service.publish_blogs(
        [{"topic": "T", "draft": "# T\n\nbody", "week": 1}],
        config=config,
        client=client,
    )
    assert results[0]["status"] == "created"
    assert results[0]["story_id"] == 5
    client.create_story.assert_called_once()
    client.update_story.assert_not_called()


def test_service_updates_when_slug_exists(schema, config):
    from backend.publishing.storyblok import service

    client = MagicMock()
    client.get_components.return_value = [
        {"name": "page", "schema": {"body": {"type": "bloks"}}},
        {"name": "text", "schema": {"text": {"type": "text"}, "tag": {"type": "option"}}},
    ]
    client.find_story_by_slug.return_value = {"id": 7}
    client.update_story.return_value = {"id": 7, "full_slug": "blog/week-1-t"}

    results = service.publish_blogs(
        [{"topic": "T", "draft": "# T\n\nbody", "week": 1}],
        config=config,
        client=client,
    )
    assert results[0]["status"] == "updated"
    assert results[0]["story_id"] == 7
    client.update_story.assert_called_once()
    client.create_story.assert_not_called()


def test_service_isolates_per_piece_failure(config):
    from backend.publishing.storyblok import service

    client = MagicMock()
    client.get_components.return_value = [
        {"name": "page", "schema": {"body": {"type": "bloks"}}},
        {"name": "text", "schema": {"text": {"type": "text"}, "tag": {"type": "option"}}},
    ]
    client.find_story_by_slug.return_value = None
    client.create_story.side_effect = [StoryblokError("boom"), {"id": 2, "full_slug": "b"}]

    results = service.publish_blogs(
        [
            {"topic": "A", "draft": "# A\n\nx", "week": 1},
            {"topic": "B", "draft": "# B\n\ny", "week": 2},
        ],
        config=config,
        client=client,
    )
    assert results[0]["status"] == "error" and results[0]["error"]
    assert results[1]["status"] == "created"


# ── FastAPI auth / endpoint ───────────────────────────────────────────────────
def test_api_rejects_without_key(monkeypatch):
    from fastapi.testclient import TestClient
    from backend.api.main import app

    monkeypatch.setenv("PUBLISHER_API_KEY", "secret")
    client = TestClient(app)
    resp = client.post("/publish/storyblok", json={"pieces": [{"topic": "t", "draft": "d"}]})
    assert resp.status_code == 401


def test_api_503_when_key_unset(monkeypatch):
    from fastapi.testclient import TestClient
    from backend.api.main import app

    monkeypatch.delenv("PUBLISHER_API_KEY", raising=False)
    client = TestClient(app)
    resp = client.post(
        "/publish/storyblok",
        headers={"X-API-Key": "whatever"},
        json={"pieces": [{"topic": "t", "draft": "d"}]},
    )
    assert resp.status_code == 503


def test_api_publish_success(monkeypatch):
    from fastapi.testclient import TestClient
    import backend.api.main as main

    monkeypatch.setenv("PUBLISHER_API_KEY", "secret")
    monkeypatch.setattr(
        main,
        "publish_blogs",
        lambda pieces, publish: [
            {"topic": "t", "status": "created", "story_id": 1, "url": "u", "full_slug": "f", "error": None}
        ],
    )
    client = TestClient(main.app)
    resp = client.post(
        "/publish/storyblok",
        headers={"X-API-Key": "secret"},
        json={"pieces": [{"topic": "t", "draft": "d", "week": 1}]},
    )
    assert resp.status_code == 200
    assert resp.json()["results"][0]["status"] == "created"


def test_api_health(monkeypatch):
    from fastapi.testclient import TestClient
    from backend.api.main import app

    monkeypatch.delenv("STORYBLOK_MANAGEMENT_TOKEN", raising=False)
    monkeypatch.delenv("STORYBLOK_SPACE_ID", raising=False)
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["storyblok_configured"] is False
