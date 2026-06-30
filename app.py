"""
app.py — Streamlit UI for AI Content Marketing Engine
Run with: streamlit run app.py
"""

import asyncio
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

load_dotenv()

from backend.graph.builder import build_graph, content_graph
from backend.graph.state import ContentState
from backend.publishing import publisher_client


@st.cache_resource
def get_hitl_graph():
    """Checkpointed graph for the human-in-the-loop flow.

    Cached as a process-wide singleton so the in-memory checkpoint survives the
    Streamlit reruns between pausing for plan approval and resuming after it.
    """
    return build_graph(checkpointer=MemorySaver())


# Per-node progress weights + labels, shared by every streaming pass.
NODE_WEIGHTS = {
    "rag_retriever": 0.10,
    "trend_researcher": 0.18,
    "planner": 0.13,
    "plan_review": 0.04,
    "blog_writer": 0.08,
    "social_writer": 0.08,
    "email_writer": 0.08,
    "ad_writer": 0.08,
    "qa": 0.13,
    "publisher": 0.10,
}
AGENT_LABELS = {
    "rag_retriever": "🔍 RAG Retriever — fetching brand context",
    "trend_researcher": "📊 Trend Researcher — searching web for keywords",
    "planner": "📅 Planner — building content calendar",
    "plan_review": "📋 Plan Review — approval gate",
    "blog_writer": "✍️  Blog Writer — drafting blog post",
    "social_writer": "📱 Social Writer — drafting social copy",
    "email_writer": "📧 Email Writer — drafting newsletter",
    "ad_writer": "🎯 Ad Writer — drafting ad copy",
    "qa": "🔎 QA Agent — scoring brand voice & SEO",
    "publisher": "📤 Publisher — pushing to Notion & Buffer",
}


def render_and_stream(graph, graph_input, config=None, *, expect_pause=False):
    """Stream a graph run, rendering live progress, and return the collected
    node outputs. With ``expect_pause`` the run is expected to interrupt (HITL),
    so the final "complete" banner is suppressed."""
    st.subheader("⚡ Agent progress")
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    agent_log = st.empty()
    log_lines: list[str] = []

    async def _run():
        _progress = 0.0
        collected: dict = {}
        async for event in graph.astream_events(
            graph_input, config=config, version="v2"
        ):
            kind = event.get("event", "")
            node = event.get("name", "")

            if kind == "on_chain_start" and node in AGENT_LABELS:
                label = AGENT_LABELS[node]
                status_text.info(f"Running: {label}")
                log_lines.append(f"⏳ {label}")
                agent_log.text("\n".join(log_lines[-8:]))

            elif kind == "on_chain_end" and node in NODE_WEIGHTS:
                _progress = min(_progress + NODE_WEIGHTS.get(node, 0.05), 1.0)
                progress_bar.progress(_progress)
                label = AGENT_LABELS.get(node, node)
                log_lines.append(f"✅ {label}")
                agent_log.text("\n".join(log_lines[-8:]))

                output = event.get("data", {}).get("output", {})
                if output:
                    collected.update(output)
        return collected

    collected = asyncio.run(_run())
    if not expect_pause:
        progress_bar.progress(1.0)
        status_text.success("✅ All agents complete!")
    return collected

st.set_page_config(
    page_title="AI Content Marketing Engine",
    page_icon="📣",
    layout="wide",
)

st.title("📣 AI Content Marketing Engine")
st.caption(
    "Multi-agent LangGraph system — generates a full month of content from a single brief"
)

# ── Sidebar: User inputs ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("Your brief")

    brand_name = st.text_input("Brand name", placeholder="e.g. WriteAI")
    target_audience = st.text_input(
        "Target audience", placeholder="e.g. B2B SaaS content marketers"
    )
    brief = st.text_area(
        "Campaign brief",
        height=140,
        placeholder="Describe what you're launching, your goals, competitors, and any focus areas...",
    )
    channels = st.multiselect(
        "Content channels",
        ["blog", "social", "email", "ad"],
        default=["blog", "social"],
    )

    review_plan = st.checkbox(
        "👀 Review plan before writing",
        value=False,
        help="Pause after the planner to approve or edit the content calendar "
        "before any drafts are generated (saves tokens on a wrong plan).",
    )

    st.divider()
    st.caption("📁 Brand documents")
    uploaded_files = st.file_uploader(
        "Upload style guide, past content, tone docs",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        brand_docs_dir = Path("./brand_docs")
        brand_docs_dir.mkdir(exist_ok=True)
        for f in uploaded_files:
            (brand_docs_dir / f.name).write_bytes(f.read())
        if st.button("Ingest brand docs into RAG"):
            with st.spinner("Indexing documents..."):
                from backend.rag.ingest import ingest

                try:
                    n = ingest("./brand_docs/")
                    st.success(f"✅ Indexed {n} chunks")
                except RuntimeError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.divider()
    run_btn = st.button(
        "🚀 Generate content",
        type="primary",
        disabled=not (brand_name and brief and channels),
        use_container_width=True,
    )

# ── Main area: run / approval gate ───────────────────────────────────────────
if run_btn:
    if not brand_name or not brief or not channels:
        st.error("Please fill in brand name, brief, and select at least one channel.")
        st.stop()

    initial_state: ContentState = {
        "brief": brief,
        "brand_name": brand_name,
        "target_audience": target_audience,
        "channels": channels,
        "require_approval": review_plan,
        "brand_context": "",
        "rag_sources": [],
        "trending_keywords": [],
        "competitor_gaps": [],
        "search_results": [],
        "content_calendar": [],
        "monthly_themes": [],
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

    if review_plan:
        # Human-in-the-loop: run up to the planner, then pause for approval.
        graph = get_hitl_graph()
        thread_id = str(uuid.uuid4())
        st.session_state["hitl_thread_id"] = thread_id
        config = {"configurable": {"thread_id": thread_id}}

        render_and_stream(graph, initial_state, config, expect_pause=True)
        snapshot = graph.get_state(config)

        if snapshot.next:  # paused at plan_review, awaiting human input
            st.session_state["hitl_phase"] = "awaiting_approval"
            st.session_state["hitl_calendar"] = snapshot.values.get(
                "content_calendar", []
            )
            st.session_state["hitl_themes"] = snapshot.values.get("monthly_themes", [])
            st.session_state.pop("final_state", None)
            st.rerun()
        else:  # nothing to approve (e.g. empty plan) — just take what we have
            st.session_state["final_state"] = snapshot.values
            st.session_state.pop("hitl_phase", None)
    else:
        # Non-interactive: run the checkpointer-less graph straight through.
        st.session_state["final_state"] = render_and_stream(content_graph, initial_state)
        st.session_state.pop("hitl_phase", None)

# ── Plan approval gate (renders while a HITL run is paused) ───────────────────
# The entire gate lives inside ONE st.empty() placeholder so it can be wiped the
# instant the user approves/cancels. Without this, the Approve/Cancel buttons
# linger as clickable "ghost" widgets during the long generating run (Streamlit
# keeps previously-rendered widgets at the same layout slot until the run that
# replaces them finishes) — and a click on a ghost button interrupts the in-
# flight run and re-enters the writers.
#   awaiting_approval → render the editable plan + buttons (the ONLY place they
#                       exist). A click captures the edits, wipes the gate, and
#                       flips to "generating".
#   generating        → gate emptied first → no clickable buttons during the run.
#                       Resume happens once, also guarded by get_state().next.
hitl_phase = st.session_state.get("hitl_phase")

if hitl_phase in ("awaiting_approval", "generating"):
    gate = st.empty()

if hitl_phase == "awaiting_approval":
    with gate.container():
        st.subheader("📋 Review the content plan")
        st.caption(
            "Edit topics, CTAs, keywords, or notes below, then approve to "
            "generate. No drafts have been written yet — nothing is wasted if "
            "you change the plan."
        )

        themes = st.session_state.get("hitl_themes", [])
        if themes:
            st.write("**Monthly themes:**", " · ".join(themes))

        edited_df = st.data_editor(
            pd.DataFrame(st.session_state.get("hitl_calendar", [])),
            use_container_width=True,
            num_rows="dynamic",
            key="plan_editor",
        )

        col_approve, col_cancel, _ = st.columns([2, 1, 4])
        approve = col_approve.button(
            "✅ Approve & generate", type="primary", use_container_width=True
        )
        cancel = col_cancel.button("✖ Cancel", use_container_width=True)

    if cancel:
        gate.empty()  # remove the buttons immediately
        for key in ("hitl_phase", "hitl_calendar", "hitl_themes"):
            st.session_state.pop(key, None)
        st.rerun()

    if approve:
        gate.empty()  # remove the buttons immediately
        # Capture edits now — the editor isn't rendered in the generating phase.
        st.session_state["hitl_edited_calendar"] = edited_df.to_dict("records")
        st.session_state["hitl_phase"] = "generating"
        st.rerun()

elif hitl_phase == "generating":
    gate.empty()  # wipe any lingering approval widgets before the long run
    graph = get_hitl_graph()
    config = {"configurable": {"thread_id": st.session_state["hitl_thread_id"]}}

    # Resume only if the thread is genuinely still paused at the interrupt; once
    # it has moved past plan_review, .next is empty and we must not resume again.
    if graph.get_state(config).next:
        decision = {
            "approved": True,
            "content_calendar": st.session_state.get("hitl_edited_calendar", []),
        }
        render_and_stream(graph, Command(resume=decision), config)

    st.session_state["final_state"] = graph.get_state(config).values
    for key in ("hitl_phase", "hitl_edited_calendar", "hitl_calendar", "hitl_themes"):
        st.session_state.pop(key, None)
    st.rerun()

# ── Results tabs (persist across reruns via session_state) ────────────────────
final_state = st.session_state.get("final_state")
if final_state:
    st.divider()
    st.subheader("📦 Generated content")

    tab_cal, tab_blog, tab_social, tab_email, tab_ad, tab_meta = st.tabs(
        ["📅 Calendar", "📝 Blog", "📱 Social", "📧 Email", "🎯 Ads", "ℹ️ Meta"]
    )

    approved = final_state.get("approved_pieces", [])

    with tab_cal:
        calendar = final_state.get("content_calendar", [])
        if calendar:
            df = pd.DataFrame(calendar)
            st.dataframe(df, use_container_width=True, height=400)
            themes = final_state.get("monthly_themes", [])
            if themes:
                st.write("**Monthly themes:**", " · ".join(themes))
        else:
            st.info("Calendar will appear here after generation.")

    with tab_blog:
        blog_pieces = [p for p in approved if p["channel"] == "blog"]
        if blog_pieces:
            health = publisher_client.check_health()
            calendar = final_state.get("content_calendar", [])
            week_by_topic = {
                e.get("topic"): e.get("week", 1)
                for e in calendar
                if e.get("channel") == "blog"
            }
            for i, p in enumerate(blog_pieces):
                with st.expander(p["topic"], expanded=True):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(p["draft"])
                    with col2:
                        st.metric("QA Score", f"{p['seo_score']:.0%}")
                        st.download_button(
                            "⬇️ Download",
                            p["draft"],
                            file_name=f"blog_{p['topic'][:30]}.md",
                            key=f"dl_blog_{i}",
                        )
                        if health and health.get("storyblok_configured"):
                            if st.button(
                                "📤 Publish",
                                key=f"publish_sb_{i}",
                                use_container_width=True,
                            ):
                                week = week_by_topic.get(p["topic"], i + 1)
                                with st.spinner("Publishing..."):
                                    try:
                                        resp = publisher_client.publish_blogs(
                                            [
                                                {
                                                    "topic": p["topic"],
                                                    "draft": p["draft"],
                                                    "week": week,
                                                }
                                            ],
                                            publish=False,
                                        )
                                        st.session_state[f"storyblok_result_{i}"] = (
                                            resp.get("results", [])
                                        )
                                    except RuntimeError as e:
                                        st.error(str(e))
                            result = st.session_state.get(f"storyblok_result_{i}")
                            if result:
                                r = result[0]
                                if r.get("status") == "error":
                                    st.error(r.get("error") or "Publish failed")
                                else:
                                    st.success(f"✓ {r['status'].capitalize()}")
                                    if r.get("url"):
                                        st.link_button("Open in Storyblok", r["url"])
        else:
            st.info("Blog posts will appear here.")

    with tab_social:
        social_pieces = [p for p in approved if p["channel"] == "social"]
        if social_pieces:
            for i, p in enumerate(social_pieces):
                with st.expander(p["topic"], expanded=True):
                    st.text_area(
                        "Copy",
                        p["draft"],
                        height=300,
                        label_visibility="collapsed",
                        key=f"ta_social_{i}",
                    )
                    st.download_button(
                        "⬇️ Download",
                        p["draft"],
                        f"social_{p['topic'][:30]}.txt",
                        key=f"dl_social_{i}",
                    )
        else:
            st.info("Social copy will appear here.")

    with tab_email:
        email_pieces = [p for p in approved if p["channel"] == "email"]
        if email_pieces:
            for i, p in enumerate(email_pieces):
                with st.expander(p["topic"], expanded=True):
                    st.text_area(
                        "Email",
                        p["draft"],
                        height=300,
                        label_visibility="collapsed",
                        key=f"ta_email_{i}",
                    )
                    st.download_button(
                        "⬇️ Download",
                        p["draft"],
                        f"email_{p['topic'][:30]}.txt",
                        key=f"dl_email_{i}",
                    )
        else:
            st.info("Email newsletters will appear here.")

    with tab_ad:
        ad_pieces = [p for p in approved if p["channel"] == "ad"]
        if ad_pieces:
            for i, p in enumerate(ad_pieces):
                with st.expander(p["topic"], expanded=True):
                    st.text_area(
                        "Ad copy",
                        p["draft"],
                        height=300,
                        label_visibility="collapsed",
                        key=f"ta_ad_{i}",
                    )
                    st.download_button(
                        "⬇️ Download",
                        p["draft"],
                        f"ad_{p['topic'][:30]}.txt",
                        key=f"dl_ad_{i}",
                    )
        else:
            st.info("Ad copy will appear here.")

    with tab_meta:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total pieces generated", len(approved))
            st.metric("Revision rounds", final_state.get("revision_round", 0))
        with col2:
            keywords = final_state.get("trending_keywords", [])
            st.write("**Trending keywords found:**")
            st.write(", ".join(keywords) if keywords else "—")
        with col3:
            st.write("**RAG sources used:**")
            sources = final_state.get("rag_sources", [])
            st.write(", ".join(sources) if sources else "No brand docs uploaded")

        urls = final_state.get("published_urls", {})
        if urls:
            st.write("**Published URLs:**")
            for k, v in urls.items():
                st.write(f"- `{k}`: {v}")
