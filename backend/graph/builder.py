"""
backend/graph/builder.py
Constructs and compiles the LangGraph StateGraph.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from backend.agents.planner import planner_node
from backend.agents.publisher import publisher_node
from backend.agents.qa_agent import qa_node
from backend.agents.trend_researcher import trend_researcher_node
from backend.agents.writers import (
    ad_copy_writer_node,
    blog_writer_node,
    email_writer_node,
    social_writer_node,
)
from backend.graph.state import ContentState
from backend.rag.retriever import rag_retriever_node

MAX_REVISIONS = 2


def route_after_qa(state: ContentState) -> list[Send] | str:
    """
    Conditional edge function called after QA node completes.

    Logic:
    - If revision_round >= MAX_REVISIONS → force route to publisher
    - If no rejected pieces → route to publisher
    - If rejected pieces exist → fan-out Send() to each rejected piece's writer node

    Each Send() carries the full state PLUS the specific rejected piece
    as 'revision_target' so the writer knows what to fix and what feedback to use.
    """
    if state["revision_round"] >= MAX_REVISIONS:
        return "publisher"

    rejected = state.get("rejected_pieces", [])
    if not rejected:
        return "publisher"

    sends = []
    for piece in rejected:
        node_name = f"{piece['channel']}_writer"
        # Feedback is read from the piece itself (authoritative, per-piece) so a
        # channel with multiple rejected pieces gets the right notes for each.
        feedback = piece.get("qa_feedback", "")
        sends.append(Send(node_name, {
            **state,
            "revision_target": piece,
            # Inject QA feedback so writer knows exactly what to fix
            "current_calendar_entry": {
                "channel": piece["channel"],
                "topic": piece["topic"],
                "keywords": [],
                "cta": "",
                "notes": f"REVISION NEEDED: {feedback}"
            }
        }))
    return sends


def fan_out_to_writers(state: ContentState) -> list[Send]:
    """
    Conditional edge function called after planner node.
    Creates one Send() per calendar entry so all writers run in parallel.
    Each Send() carries the full state PLUS current_calendar_entry for that specific piece.
    """
    sends = []
    for entry in state["content_calendar"]:
        channel = entry["channel"]
        # Only fan out to channels the user selected
        if channel not in state["channels"]:
            continue
        node_name = f"{channel}_writer"
        sends.append(Send(node_name, {
            **state,
            "current_calendar_entry": entry,
            "revision_target": {}
        }))
    return sends


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> StateGraph:
    """Build and compile the content pipeline.

    By default the graph is compiled without a checkpointer. Pass one to make
    runs resumable and to enable future human-in-the-loop interrupts — e.g.
    ``build_graph(checkpointer=MemorySaver())`` (or a ``SqliteSaver`` /
    ``PostgresSaver`` for durability across restarts). Callers that pass a
    checkpointer MUST invoke the graph with a
    ``config={"configurable": {"thread_id": ...}}``.
    """
    graph = StateGraph(ContentState)

    # Register all nodes
    graph.add_node("rag_retriever",    rag_retriever_node)
    graph.add_node("trend_researcher", trend_researcher_node)
    graph.add_node("planner",          planner_node)
    graph.add_node("blog_writer",      blog_writer_node)
    graph.add_node("social_writer",    social_writer_node)
    graph.add_node("email_writer",     email_writer_node)
    graph.add_node("ad_writer",        ad_copy_writer_node)
    graph.add_node("qa",               qa_node)
    graph.add_node("publisher",        publisher_node)

    # Linear edges: START → rag → trend → planner
    graph.add_edge(START, "rag_retriever")
    graph.add_edge("rag_retriever", "trend_researcher")
    graph.add_edge("trend_researcher", "planner")

    # Fan-out: planner → parallel writers via Send()
    graph.add_conditional_edges(
        "planner",
        fan_out_to_writers,
        ["blog_writer", "social_writer", "email_writer", "ad_writer"],
    )

    # All writers converge to QA
    for writer in ["blog_writer", "social_writer", "email_writer", "ad_writer"]:
        graph.add_edge(writer, "qa")

    # QA: conditional route to re-writers OR publisher
    graph.add_conditional_edges(
        "qa",
        route_after_qa,
        ["blog_writer", "social_writer", "email_writer", "ad_writer", "publisher"],
    )

    graph.add_edge("publisher", END)

    return graph.compile(checkpointer=checkpointer)


# Compile once at import time — reused across all requests.
# Checkpointer-less by default so importers/tests don't need a thread_id.
# Opt in to resumability by calling build_graph(checkpointer=MemorySaver()).
content_graph = build_graph()
