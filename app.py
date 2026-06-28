"""
app.py — Streamlit UI for AI Content Marketing Engine
Run with: streamlit run app.py
"""

import streamlit as st
import asyncio
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from backend.graph.builder import content_graph
from backend.graph.state import ContentState

st.set_page_config(
    page_title="AI Content Marketing Engine",
    page_icon="📣",
    layout="wide",
)

st.title("📣 AI Content Marketing Engine")
st.caption("Multi-agent LangGraph system — generates a full month of content from a single brief")

# ── Sidebar: User inputs ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("Your brief")

    brand_name = st.text_input("Brand name", placeholder="e.g. WriteAI")
    target_audience = st.text_input(
        "Target audience",
        placeholder="e.g. B2B SaaS content marketers"
    )
    brief = st.text_area(
        "Campaign brief",
        height=140,
        placeholder="Describe what you're launching, your goals, competitors, and any focus areas..."
    )
    channels = st.multiselect(
        "Content channels",
        ["blog", "social", "email", "ad"],
        default=["blog", "social"],
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

# ── Main area: Results ───────────────────────────────────────────────────────
if run_btn:
    if not brand_name or not brief or not channels:
        st.error("Please fill in brand name, brief, and select at least one channel.")
        st.stop()

    initial_state: ContentState = {
        "brief": brief,
        "brand_name": brand_name,
        "target_audience": target_audience,
        "channels": channels,
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

    # ── Live progress section ────────────────────────────────────────────────
    st.subheader("⚡ Agent progress")
    progress_bar  = st.progress(0.0)
    status_text   = st.empty()
    agent_log     = st.empty()

    NODE_WEIGHTS = {
        "rag_retriever": 0.10,
        "trend_researcher": 0.20,
        "planner": 0.15,
        "blog_writer": 0.08,
        "social_writer": 0.08,
        "email_writer": 0.08,
        "ad_writer": 0.08,
        "qa": 0.13,
        "publisher": 0.10,
    }
    log_lines = []

    AGENT_LABELS = {
        "rag_retriever":    "🔍 RAG Retriever — fetching brand context",
        "trend_researcher": "📊 Trend Researcher — searching web for keywords",
        "planner":          "📅 Planner — building content calendar",
        "blog_writer":      "✍️  Blog Writer — drafting blog post",
        "social_writer":    "📱 Social Writer — drafting social copy",
        "email_writer":     "📧 Email Writer — drafting newsletter",
        "ad_writer":        "🎯 Ad Writer — drafting ad copy",
        "qa":               "🔎 QA Agent — scoring brand voice & SEO",
        "publisher":        "📤 Publisher — pushing to Notion & Buffer",
    }

    async def run_with_streaming():
        _progress = 0.0
        collected = {}
        async for event in content_graph.astream_events(initial_state, version="v2"):
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

        progress_bar.progress(1.0)
        status_text.success("✅ All agents complete!")
        return collected

    st.session_state["final_state"] = asyncio.run(run_with_streaming())

# ── Results tabs (persist across reruns via session_state) ────────────────────
final_state = st.session_state.get("final_state")
if final_state:
    st.divider()
    st.subheader("📦 Generated content")

    tab_cal, tab_blog, tab_social, tab_email, tab_ad, tab_meta = st.tabs([
        "📅 Calendar", "📝 Blog", "📱 Social", "📧 Email", "🎯 Ads", "ℹ️ Meta"
    ])

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
        else:
            st.info("Blog posts will appear here.")

    with tab_social:
        social_pieces = [p for p in approved if p["channel"] == "social"]
        if social_pieces:
            for i, p in enumerate(social_pieces):
                with st.expander(p["topic"], expanded=True):
                    st.text_area("Copy", p["draft"], height=300, label_visibility="collapsed", key=f"ta_social_{i}")
                    st.download_button("⬇️ Download", p["draft"], f"social_{p['topic'][:30]}.txt", key=f"dl_social_{i}")
        else:
            st.info("Social copy will appear here.")

    with tab_email:
        email_pieces = [p for p in approved if p["channel"] == "email"]
        if email_pieces:
            for i, p in enumerate(email_pieces):
                with st.expander(p["topic"], expanded=True):
                    st.text_area("Email", p["draft"], height=300, label_visibility="collapsed", key=f"ta_email_{i}")
                    st.download_button("⬇️ Download", p["draft"], f"email_{p['topic'][:30]}.txt", key=f"dl_email_{i}")
        else:
            st.info("Email newsletters will appear here.")

    with tab_ad:
        ad_pieces = [p for p in approved if p["channel"] == "ad"]
        if ad_pieces:
            for i, p in enumerate(ad_pieces):
                with st.expander(p["topic"], expanded=True):
                    st.text_area("Ad copy", p["draft"], height=300, label_visibility="collapsed", key=f"ta_ad_{i}")
                    st.download_button("⬇️ Download", p["draft"], f"ad_{p['topic'][:30]}.txt", key=f"dl_ad_{i}")
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
