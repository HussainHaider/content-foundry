"""tests/test_tools.py — agent tool-calling (web_search / news_search)."""

from unittest.mock import patch, MagicMock


# ── Tool unit tests ─────────────────────────────────────────────────
@patch("backend.agents.tools.GoogleSerperAPIWrapper")
def test_web_search_returns_results(mock_wrapper):
    from backend.agents.tools import web_search

    mock_wrapper.return_value.run.return_value = "web results for x"
    assert web_search.invoke({"query": "x"}) == "web results for x"
    mock_wrapper.return_value.run.assert_called_once_with("x")


@patch("backend.agents.tools.GoogleSerperAPIWrapper")
def test_news_search_uses_news_type(mock_wrapper):
    from backend.agents.tools import news_search

    mock_wrapper.return_value.run.return_value = "news results"
    assert news_search.invoke({"query": "y"}) == "news results"
    # news_search must construct the wrapper with type="news"
    _, kwargs = mock_wrapper.call_args
    assert kwargs.get("type") == "news"


@patch("backend.agents.tools.GoogleSerperAPIWrapper")
def test_web_search_swallows_errors(mock_wrapper):
    from backend.agents.tools import web_search

    mock_wrapper.return_value.run.side_effect = RuntimeError("boom")
    out = web_search.invoke({"query": "z"})
    assert "unavailable" in out.lower()


# ── Node tool-calling loop ──────────────────────────────────────────
@patch("backend.agents.trend_researcher.llm")
@patch("backend.agents.trend_researcher.tool_llm")
@patch.dict(
    "backend.agents.trend_researcher.TOOL_REGISTRY",
    {"web_search": MagicMock(invoke=MagicMock(return_value="stub search result"))},
    clear=True,
)
def test_trend_researcher_runs_tools_then_extracts(mock_tool_llm, mock_llm):
    from backend.agents.trend_researcher import trend_researcher_node

    # 1st invoke: model asks to call web_search. 2nd invoke: no tool calls → stop.
    call_msg = MagicMock(
        tool_calls=[{"name": "web_search", "args": {"query": "q"}, "id": "1"}]
    )
    stop_msg = MagicMock(tool_calls=[])
    mock_tool_llm.invoke.side_effect = [call_msg, stop_msg]

    # Final extraction returns JSON.
    mock_llm.invoke.return_value = MagicMock(
        content='{"trending_keywords": ["a", "b"], "competitor_gaps": ["g1"]}'
    )

    state = {
        "brand_name": "WriteAI",
        "target_audience": "B2B marketers",
        "brief": "Launch v2",
        "channels": ["blog"],
    }
    result = trend_researcher_node(state)

    # The tool actually executed and its output was captured.
    assert mock_tool_llm.invoke.call_count == 2
    assert result["search_results"] == [{"query": "q", "result": "stub search result"}]
    # The structured keys flow through to state.
    assert result["trending_keywords"] == ["a", "b"]
    assert result["competitor_gaps"] == ["g1"]
