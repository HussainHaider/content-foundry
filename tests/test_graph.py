"""tests/test_graph.py"""

from unittest.mock import MagicMock, patch

import pytest

from backend.graph.builder import fan_out_to_writers, route_after_qa
from backend.graph.state import ContentState


@pytest.fixture
def base_state() -> ContentState:
    return {
        "brief": "Launch WriteAI v2",
        "brand_name": "WriteAI",
        "target_audience": "B2B SaaS content marketers",
        "channels": ["blog", "social", "email", "ad"],
        "brand_context": "WriteAI is friendly, expert, jargon-free.",
        "rag_sources": ["style_guide.pdf"],
        "trending_keywords": ["AI writing tools", "brand voice AI"],
        "competitor_gaps": ["no one covers AI content ROI"],
        "search_results": [],
        "content_calendar": [
            {
                "week": 1,
                "channel": "blog",
                "topic": "Why AI sounds off-brand",
                "keywords": ["brand voice AI"],
                "cta": "Try WriteAI",
                "notes": "Lead with pain",
            },
            {
                "week": 1,
                "channel": "social",
                "topic": "5 AI myths",
                "keywords": ["AI myths"],
                "cta": "Read more",
                "notes": "Hook-first",
            },
        ],
        "monthly_themes": ["Brand voice problem", "AI vs human"],
        "content_pieces": [],
        "current_calendar_entry": {},
        "revision_target": {},
        "qa_feedback": {},
        "approved_pieces": [],
        "rejected_pieces": [],
        "published_urls": {},
        "revision_round": 0,
        "messages": [],
    }


def test_route_after_qa_all_approved(base_state):
    state = {**base_state, "rejected_pieces": [], "revision_round": 0}
    assert route_after_qa(state) == "publisher"


def test_route_after_qa_hits_revision_limit(base_state):
    rejected = {
        "channel": "blog",
        "topic": "T",
        "draft": "",
        "seo_score": 0.4,
        "qa_passed": False,
        "revision_count": 2,
        "published_url": None,
    }
    state = {**base_state, "rejected_pieces": [rejected], "revision_round": 2}
    assert route_after_qa(state) == "publisher"


def test_route_after_qa_with_rejections(base_state):
    from langgraph.types import Send

    rejected = {
        "channel": "blog",
        "topic": "T",
        "draft": "",
        "seo_score": 0.5,
        "qa_passed": False,
        "qa_feedback": "fix the intro",
        "revision_count": 0,
        "published_url": None,
    }
    state = {**base_state, "rejected_pieces": [rejected], "revision_round": 0}
    result = route_after_qa(state)
    assert isinstance(result, list)
    assert any(isinstance(r, Send) for r in result)


def test_route_after_qa_feedback_does_not_collide(base_state):
    """Two rejected pieces on the SAME channel must each receive their own
    feedback — the regression this guards against was feedback keyed by channel."""
    from langgraph.types import Send

    rejected = [
        {
            "channel": "blog",
            "topic": "Week 1 blog",
            "draft": "",
            "seo_score": 0.5,
            "qa_passed": False,
            "qa_feedback": "FEEDBACK_ONE",
            "revision_count": 0,
            "published_url": None,
        },
        {
            "channel": "blog",
            "topic": "Week 2 blog",
            "draft": "",
            "seo_score": 0.4,
            "qa_passed": False,
            "qa_feedback": "FEEDBACK_TWO",
            "revision_count": 0,
            "published_url": None,
        },
    ]
    state = {**base_state, "rejected_pieces": rejected, "revision_round": 0}
    result = route_after_qa(state)

    sends = [r for r in result if isinstance(r, Send)]
    assert len(sends) == 2
    notes_by_topic = {
        s.arg["current_calendar_entry"]["topic"]: s.arg["current_calendar_entry"]["notes"]
        for s in sends
    }
    assert "FEEDBACK_ONE" in notes_by_topic["Week 1 blog"]
    assert "FEEDBACK_TWO" in notes_by_topic["Week 2 blog"]


def test_fan_out_creates_one_send_per_entry(base_state):
    from langgraph.types import Send

    result = fan_out_to_writers(base_state)
    assert len(result) == len(base_state["content_calendar"])
    assert all(isinstance(r, Send) for r in result)


def test_graph_compiles_with_checkpointer():
    """The shipped graph must carry a checkpointer so runs are resumable."""
    from backend.graph.builder import content_graph

    assert content_graph.checkpointer is not None


def test_build_graph_without_checkpointer():
    from backend.graph.builder import build_graph

    graph = build_graph(checkpointer=None)
    assert graph.checkpointer is None


@patch("backend.agents.writers.llm")
def test_blog_writer_appends_piece(mock_llm, base_state):
    from backend.agents.writers import blog_writer_node

    mock_llm.invoke.return_value = MagicMock(
        content="# Test blog post\n\nContent here..."
    )
    state = {**base_state, "current_calendar_entry": base_state["content_calendar"][0]}
    result = blog_writer_node(state)
    assert len(result["content_pieces"]) == 1
    assert result["content_pieces"][0]["channel"] == "blog"
