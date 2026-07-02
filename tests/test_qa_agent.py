"""tests/test_qa_agent.py — QA scoring + per-piece feedback keying."""

import json
from unittest.mock import MagicMock, patch


def _piece(channel: str, topic: str) -> dict:
    return {
        "channel": channel,
        "topic": topic,
        "draft": f"draft for {topic}",
        "seo_score": 0.0,
        "qa_passed": False,
        "qa_feedback": "",
        "revision_count": 0,
        "published_url": None,
    }


@patch("backend.agents.qa_agent.llm")
def test_qa_feedback_keyed_per_piece_not_per_channel(mock_llm):
    """Two rejected blog pieces must keep distinct feedback (no channel collision)."""
    from backend.agents.qa_agent import qa_node

    qa_json = {
        "pieces": [
            {
                "channel": "blog",
                "topic": "Week 1 blog",
                "passed": False,
                "seo_score": 0.5,
                "feedback": "FEEDBACK_ONE",
                "issues": ["a"],
            },
            {
                "channel": "blog",
                "topic": "Week 2 blog",
                "passed": False,
                "seo_score": 0.6,
                "feedback": "FEEDBACK_TWO",
                "issues": ["b"],
            },
        ]
    }
    mock_llm.invoke.return_value = MagicMock(content=json.dumps(qa_json))

    state = {
        "brand_name": "WriteAI",
        "target_audience": "marketers",
        "brand_context": "friendly expert",
        "content_pieces": [_piece("blog", "Week 1 blog"), _piece("blog", "Week 2 blog")],
        "approved_pieces": [],
        "revision_round": 0,
    }
    result = qa_node(state)

    assert len(result["rejected_pieces"]) == 2
    # Observability dict is keyed per-piece, so both survive.
    assert result["qa_feedback"]["blog::Week 1 blog"] == "FEEDBACK_ONE"
    assert result["qa_feedback"]["blog::Week 2 blog"] == "FEEDBACK_TWO"
    # Each piece carries its own authoritative feedback.
    fb = {p["topic"]: p["qa_feedback"] for p in result["rejected_pieces"]}
    assert fb["Week 1 blog"] == "FEEDBACK_ONE"
    assert fb["Week 2 blog"] == "FEEDBACK_TWO"


@patch("backend.agents.qa_agent.llm")
def test_qa_approved_piece_has_no_feedback(mock_llm):
    from backend.agents.qa_agent import qa_node

    qa_json = {
        "pieces": [
            {
                "channel": "blog",
                "topic": "Great post",
                "passed": True,
                "seo_score": 0.9,
                "feedback": "",
                "issues": [],
            }
        ]
    }
    mock_llm.invoke.return_value = MagicMock(content=json.dumps(qa_json))

    state = {
        "brand_name": "WriteAI",
        "target_audience": "marketers",
        "brand_context": "friendly expert",
        "content_pieces": [_piece("blog", "Great post")],
        "approved_pieces": [],
        "revision_round": 0,
    }
    result = qa_node(state)

    assert len(result["approved_pieces"]) == 1
    assert result["approved_pieces"][0]["qa_passed"] is True
    assert result["rejected_pieces"] == []
    assert result["qa_feedback"] == {}
