"""tests/test_plan_review.py — human-in-the-loop plan approval gate.

These tests drive the real interrupt/resume machinery through a minimal graph
(just the plan_review node + a checkpointer), so no LLM or external service is
touched. They guard three behaviours:
  - require_approval falsy  → pass-through, no pause
  - require_approval True   → graph pauses with the calendar payload
  - resume with edits       → the edited calendar replaces the original
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from backend.agents.plan_review import plan_review_node
from backend.graph.state import ContentState

CALENDAR = [
    {"week": 1, "channel": "blog", "topic": "Original topic", "keywords": [],
     "cta": "", "notes": ""},
]


def _mini_graph():
    g = StateGraph(ContentState)
    g.add_node("plan_review", plan_review_node)
    g.add_edge(START, "plan_review")
    g.add_edge("plan_review", END)
    return g.compile(checkpointer=MemorySaver())


def _base_state(require_approval: bool) -> dict:
    return {
        "content_calendar": CALENDAR,
        "monthly_themes": ["theme"],
        "require_approval": require_approval,
    }


def test_pass_through_when_approval_disabled():
    graph = _mini_graph()
    config = {"configurable": {"thread_id": "no-approval"}}

    graph.invoke(_base_state(require_approval=False), config)

    snap = graph.get_state(config)
    assert not snap.next  # ran straight to END, never paused
    assert snap.values["content_calendar"] == CALENDAR


def test_pauses_for_approval_with_calendar_payload():
    graph = _mini_graph()
    config = {"configurable": {"thread_id": "pause"}}

    result = graph.invoke(_base_state(require_approval=True), config)

    # The graph is paused at plan_review, awaiting human input.
    snap = graph.get_state(config)
    assert snap.next == ("plan_review",)
    # The interrupt payload (surfaced to the UI) carries the proposed plan.
    interrupts = result["__interrupt__"]
    assert interrupts[0].value["type"] == "plan_approval"
    assert interrupts[0].value["content_calendar"] == CALENDAR


def test_resume_with_edited_calendar_applies_edits():
    graph = _mini_graph()
    config = {"configurable": {"thread_id": "edit"}}
    graph.invoke(_base_state(require_approval=True), config)

    edited = [
        {"week": 1, "channel": "blog", "topic": "EDITED topic", "keywords": [],
         "cta": "Try it", "notes": "sharper angle"},
    ]
    graph.invoke(
        Command(resume={"approved": True, "content_calendar": edited}), config
    )

    snap = graph.get_state(config)
    assert not snap.next  # resumed to completion
    assert snap.values["content_calendar"] == edited


def test_resume_without_edits_keeps_original_calendar():
    graph = _mini_graph()
    config = {"configurable": {"thread_id": "approve-asis"}}
    graph.invoke(_base_state(require_approval=True), config)

    graph.invoke(Command(resume={"approved": True}), config)

    snap = graph.get_state(config)
    assert not snap.next
    assert snap.values["content_calendar"] == CALENDAR
