"""
backend/graph/state.py
Central shared state for the entire LangGraph pipeline.
Every node receives the full state and returns a partial dict to update it.
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages


def _merge_content_pieces(left: list, right: list | None) -> list:
    """Reducer for content_pieces: None resets the list; a list is appended."""
    if right is None:
        return []
    return (left or []) + right


class ContentPiece(TypedDict):
    """Represents one generated content item (one channel, one week)."""

    channel: Literal["blog", "social", "email", "ad"]
    topic: str
    draft: str
    seo_score: float  # 0.0 to 1.0, set by QA agent
    qa_passed: bool  # Set by QA agent
    revision_count: int  # Incremented each time QA rejects
    published_url: str | None  # Set by publisher agent


class ContentState(TypedDict):
    # ── User input (set at pipeline start) ─────────────────────────
    brief: str  # Plain English brief from user
    brand_name: str
    target_audience: str
    channels: list[Literal["blog", "social", "email", "ad"]]

    # ── RAG output (set by rag_retriever node) ──────────────────────
    brand_context: str  # Concatenated brand doc chunks
    rag_sources: list[str]  # Source filenames used

    # ── Trend researcher output ─────────────────────────────────────
    trending_keywords: list[str]  # Top 10 SEO keywords found
    competitor_gaps: list[str]  # 5 content gaps competitors miss
    search_results: list[dict]  # Raw search result dicts

    # ── Planner output ──────────────────────────────────────────────
    content_calendar: list[dict]  # [{week, channel, topic, keywords, cta, notes}]
    monthly_themes: list[str]  # 4 weekly themes

    # ── Writer outputs ──────────────────────────────────────────────
    content_pieces: Annotated[list[ContentPiece], _merge_content_pieces]
    current_calendar_entry: dict  # Set by Send() fan-out, one per writer
    revision_target: dict  # Set during re-write loops

    # ── QA outputs ─────────────────────────────────────────────────
    qa_feedback: dict[str, str]  # {channel: "feedback text"}
    approved_pieces: list[ContentPiece]  # Passed QA
    rejected_pieces: list[ContentPiece]  # Failed QA, need revision

    # ── Publisher output ────────────────────────────────────────────
    published_urls: dict[str, str]  # {channel_index: url}

    # ── Control flow ────────────────────────────────────────────────
    revision_round: int  # Incremented by QA. Max = 2
    messages: Annotated[list, add_messages]  # LangSmith trace
