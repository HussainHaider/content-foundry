"""
backend/agents/plan_review.py

Node: plan_review — the human-in-the-loop approval gate.

Sits between the planner and the writer fan-out. When
``state['require_approval']`` is True, it pauses the graph with ``interrupt()``,
surfacing the proposed content calendar to a human. Execution resumes only when
the caller sends a ``Command(resume=<decision>)``:

    {"approved": True}                          → write the plan as-is
    {"approved": True, "content_calendar": [...]} → write the EDITED plan

When ``require_approval`` is falsy (the default) the node is a no-op pass-through,
so the non-interactive pipeline — and the checkpointer-less default graph — is
completely unaffected.

NOTE: ``interrupt()`` requires the graph to be compiled WITH a checkpointer.
Enabling approval on a checkpointer-less graph will raise at runtime; build the
graph via ``build_graph(checkpointer=...)`` for the interactive flow.
"""

from langgraph.types import interrupt

from backend.graph.state import ContentState


def plan_review_node(state: ContentState) -> dict:
    if not state.get("require_approval"):
        # Pass-through: no pause, plan flows straight to the writers.
        return {}

    # Pause here. The payload is what the UI shows the reviewer. On resume,
    # interrupt() returns whatever the caller passed via Command(resume=...).
    decision = interrupt(
        {
            "type": "plan_approval",
            "content_calendar": state.get("content_calendar", []),
            "monthly_themes": state.get("monthly_themes", []),
        }
    )

    # Apply an edited calendar if the reviewer supplied one; otherwise approve
    # the plan unchanged. fan_out_to_writers reads the (possibly edited)
    # content_calendar immediately after this node returns.
    if isinstance(decision, dict) and decision.get("content_calendar"):
        return {"content_calendar": decision["content_calendar"]}
    return {}
