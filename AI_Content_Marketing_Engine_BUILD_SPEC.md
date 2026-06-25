# AI Content Marketing Engine — Complete Build Specification
> Hand this entire document to Claude Code CLI to build the project from scratch.

---

## 1. Project Overview

Build a **production-grade multi-agent AI system** that:
1. Takes a user's brand brief as input
2. Researches trending topics via web search
3. Retrieves brand voice context from uploaded documents (RAG)
4. Builds a 4-week content calendar
5. Generates blog posts, social copy, email newsletters, and ad copy **in parallel**
6. Runs a QA agent that scores each piece and sends failed pieces back for revision
7. Publishes approved content to Notion and Buffer
8. Streams live progress to the UI as each agent completes

**Tech stack:**
- LangGraph (multi-agent state machine)
- Anthropic Claude claude-sonnet-4-6 (LLM)
- Qdrant (vector database for RAG)
- Streamlit (UI — replaces FastAPI + Next.js for simplicity)
- Serper API (web search)
- Notion API + Buffer API (publishing)
- Docker Compose (local + deployment)
- Railway (cloud deployment)

---

## 2. Complete Project Folder Structure

Create exactly this structure:

```
content-marketing-engine/
├── backend/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── trend_researcher.py
│   │   ├── planner.py
│   │   ├── writers.py
│   │   ├── qa_agent.py
│   │   └── publisher.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── ingest.py
│   │   └── retriever.py
│   └── graph/
│       ├── __init__.py
│       ├── state.py
│       └── builder.py
├── app.py                  ← Streamlit entry point
├── brand_docs/             ← User drops brand PDFs/TXTs here
│   └── .gitkeep
├── tests/
│   ├── __init__.py
│   └── test_graph.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── .env                    ← Never commit this
├── .gitignore
└── README.md
```

---

## 3. Environment Variables

Create `.env.example` with these keys. User must copy to `.env` and fill in:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Web search (get free key at serper.dev — 2500 searches/month free)
SERPER_API_KEY=...

# Vector DB (local Docker URL, change for cloud Qdrant)
QDRANT_URL=http://localhost:6333

# Publishing — optional, app works without these
NOTION_TOKEN=secret_...
NOTION_DATABASE_ID=...
BUFFER_TOKEN=...
BUFFER_PROFILE_IDS=profile_id_1,profile_id_2

# Observability — optional but recommended (langsmith.com)
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=content-marketing-engine
```

---

## 4. requirements.txt

```txt
# LangGraph + LangChain
langgraph>=0.2.0
langchain>=0.3.0
langchain-anthropic>=0.2.0
langchain-community>=0.3.0
langchain-qdrant>=0.1.0

# LLM
anthropic>=0.34.0

# Vector DB
qdrant-client>=1.11.0

# Document parsing for RAG ingestion
pymupdf>=1.24.0

# Web search
google-search-results>=2.4.2

# UI
streamlit>=1.39.0
pandas>=2.0.0

# Publishing APIs
notion-client>=2.2.3
httpx>=0.27.0

# Observability
langsmith>=0.1.0

# Testing
pytest>=8.3.0
pytest-asyncio>=0.24.0

# Utilities
python-dotenv>=1.0.0
```

---

## 5. LangGraph State Schema — `backend/graph/state.py`

This is the central data object. Every agent reads from and writes to this TypedDict. Nothing else is passed between agents.

```python
"""
backend/graph/state.py
Central shared state for the entire LangGraph pipeline.
Every node receives the full state and returns a partial dict to update it.
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages


class ContentPiece(TypedDict):
    """Represents one generated content item (one channel, one week)."""
    channel: Literal["blog", "social", "email", "ad"]
    topic: str
    draft: str
    seo_score: float        # 0.0 to 1.0, set by QA agent
    qa_passed: bool         # Set by QA agent
    revision_count: int     # Incremented each time QA rejects
    published_url: str | None  # Set by publisher agent


class ContentState(TypedDict):
    # ── User input (set at pipeline start) ─────────────────────────
    brief: str                    # Plain English brief from user
    brand_name: str
    target_audience: str
    channels: list[Literal["blog", "social", "email", "ad"]]

    # ── RAG output (set by rag_retriever node) ──────────────────────
    brand_context: str            # Concatenated brand doc chunks
    rag_sources: list[str]        # Source filenames used

    # ── Trend researcher output ─────────────────────────────────────
    trending_keywords: list[str]  # Top 10 SEO keywords found
    competitor_gaps: list[str]    # 5 content gaps competitors miss
    search_results: list[dict]    # Raw search result dicts

    # ── Planner output ──────────────────────────────────────────────
    content_calendar: list[dict]  # [{week, channel, topic, keywords, cta, notes}]
    monthly_themes: list[str]     # 4 weekly themes

    # ── Writer outputs ──────────────────────────────────────────────
    content_pieces: list[ContentPiece]   # Current round drafts
    current_calendar_entry: dict         # Set by Send() fan-out, one per writer
    revision_target: dict                # Set during re-write loops

    # ── QA outputs ─────────────────────────────────────────────────
    qa_feedback: dict[str, str]           # {channel: "feedback text"}
    approved_pieces: list[ContentPiece]   # Passed QA
    rejected_pieces: list[ContentPiece]   # Failed QA, need revision

    # ── Publisher output ────────────────────────────────────────────
    published_urls: dict[str, str]        # {channel_index: url}

    # ── Control flow ────────────────────────────────────────────────
    revision_round: int           # Incremented by QA. Max = 2
    messages: Annotated[list, add_messages]  # LangSmith trace
```

---

## 6. LangGraph Graph Builder — `backend/graph/builder.py`

This is the most important file. It defines the full agent graph topology.

**Graph topology:**
```
START
  → rag_retriever          (populates brand_context from Qdrant)
  → trend_researcher       (web search → keywords + gaps)
  → planner                (builds content calendar JSON)
  → [Send() fan-out]       (parallel: blog_writer, social_writer, email_writer, ad_writer)
  → qa                     (scores all pieces, routes approved vs rejected)
  → [conditional]          (if rejected & revision_round < 2 → fan back to writers)
  → publisher              (pushes to Notion + Buffer)
END
```

**Key LangGraph patterns used:**
- `Send()` API for parallel fan-out to writer nodes
- `add_conditional_edges()` for QA routing
- Cyclic graph (QA can loop back to writers, max 2 times)
- Shared TypedDict state passed through all nodes

```python
"""
backend/graph/builder.py
Constructs and compiles the LangGraph StateGraph.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

from backend.graph.state import ContentState
from backend.agents.trend_researcher import trend_researcher_node
from backend.agents.planner import planner_node
from backend.agents.writers import (
    blog_writer_node,
    social_writer_node,
    email_writer_node,
    ad_copy_writer_node,
)
from backend.agents.qa_agent import qa_node
from backend.agents.publisher import publisher_node
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
        sends.append(Send(node_name, {
            **state,
            "revision_target": piece,
            # Inject QA feedback so writer knows exactly what to fix
            "current_calendar_entry": {
                "channel": piece["channel"],
                "topic": piece["topic"],
                "keywords": [],
                "cta": "",
                "notes": f"REVISION NEEDED: {state['qa_feedback'].get(piece['channel'], '')}"
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


def build_graph() -> StateGraph:
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

    return graph.compile()


# Compile once at import time — reused across all requests
content_graph = build_graph()
```

---

## 7. RAG Ingestion — `backend/rag/ingest.py`

Run once before starting the app to load brand documents into Qdrant.

```python
"""
backend/rag/ingest.py

Usage:
    python -m backend.rag.ingest --docs ./brand_docs/

Loads all PDF, TXT, MD files from the given folder.
Chunks them with overlap and stores embeddings in Qdrant.
"""

import os
import argparse
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import AnthropicEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

COLLECTION_NAME = "brand_content"
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 120


def ingest(docs_path: str) -> int:
    """Returns number of chunks ingested."""
    loader = DirectoryLoader(docs_path, glob="**/*.{pdf,txt,md}", show_progress=True)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)

    embeddings = AnthropicEmbeddings(
        model="voyage-3",
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
    )

    client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    vectorstore.add_documents(chunks)
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="./brand_docs/")
    args = parser.parse_args()
    n = ingest(args.docs)
    print(f"✅ Ingested {n} chunks into Qdrant")
```

---

## 8. RAG Retriever Node — `backend/rag/retriever.py`

```python
"""
backend/rag/retriever.py
LangGraph node — called first in the pipeline.
Retrieves brand context relevant to the brief from Qdrant.
Result stored in state['brand_context'] and shared with ALL downstream agents.
"""

import os
from langchain_anthropic import AnthropicEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from backend.graph.state import ContentState

COLLECTION_NAME = "brand_content"
TOP_K = 6


def rag_retriever_node(state: ContentState) -> dict:
    """
    Retrieves the TOP_K most relevant brand document chunks.
    Query is constructed from the brief + target audience.
    Returns brand_context (concatenated text) and rag_sources (filenames).
    """
    query = (
        f"Brand style guide, tone of voice, content examples "
        f"for: {state['target_audience']}. Brief: {state['brief']}"
    )

    client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    embeddings = AnthropicEmbeddings(
        model="voyage-3",
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    docs = vectorstore.similarity_search(query, k=TOP_K)

    brand_context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )
    rag_sources = list({doc.metadata.get("source", "unknown") for doc in docs})

    return {
        "brand_context": brand_context,
        "rag_sources": rag_sources,
    }
```

---

## 9. Trend Researcher Agent — `backend/agents/trend_researcher.py`

```python
"""
backend/agents/trend_researcher.py

Node: trend_researcher
Input state keys used: brief, brand_name, target_audience, channels
Output state keys set: trending_keywords, competitor_gaps, search_results

Runs 3 web searches using Serper API, then uses Claude to extract
structured insights (top keywords + competitor content gaps).
Returns JSON parsed into state fields.
"""

import os
import json
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import SerperDevTool
from backend.graph.state import ContentState

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0.3,
    anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
)

search_tool = SerperDevTool(serper_api_key=os.environ.get("SERPER_API_KEY", ""))

SYSTEM_PROMPT = """You are a senior SEO content strategist.
Analyze web search results and extract actionable insights.

Return ONLY valid JSON (no markdown, no explanation):
{
  "trending_keywords": ["keyword1", ...],   // exactly 10 high-value SEO keywords
  "competitor_gaps":   ["gap1", ...],        // exactly 5 topics competitors are missing
  "summary": "2-3 sentence opportunity overview"
}"""


def trend_researcher_node(state: ContentState) -> dict:
    search_queries = [
        f"{state['brief']} trending topics 2025",
        f"{state['brand_name']} competitors content marketing strategy",
        f"{state['target_audience']} biggest pain points questions 2025",
    ]

    search_results = []
    for query in search_queries:
        try:
            result = search_tool.run(query)
        except Exception:
            result = f"Search unavailable for: {query}"
        search_results.append({"query": query, "result": result})

    search_context = "\n\n".join(
        f"Query: {r['query']}\nResults: {r['result']}"
        for r in search_results
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""
Brand: {state['brand_name']}
Audience: {state['target_audience']}
Brief: {state['brief']}
Channels: {', '.join(state['channels'])}

Web research:
{search_context}

Extract keywords and gaps. Return JSON only.
"""),
    ])

    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback if LLM adds markdown
        import re
        match = re.search(r'\{.*\}', response.content, re.DOTALL)
        parsed = json.loads(match.group()) if match else {
            "trending_keywords": [], "competitor_gaps": []
        }

    return {
        "trending_keywords": parsed.get("trending_keywords", []),
        "competitor_gaps":   parsed.get("competitor_gaps", []),
        "search_results":    search_results,
    }
```

---

## 10. Planner Agent — `backend/agents/planner.py`

```python
"""
backend/agents/planner.py

Node: planner
Input state keys used: brief, brand_name, target_audience, channels,
                       trending_keywords, competitor_gaps, brand_context
Output state keys set: monthly_themes, content_calendar,
                       revision_round (initialized to 0),
                       content_pieces, approved_pieces, rejected_pieces (all initialized)

Builds a 4-week content calendar — one entry per active channel per week.
Each entry includes: week number, channel, topic title, focus keywords, CTA, and notes.
The 'notes' field gives the writer a strategic angle to follow.
"""

import os
import json
import re
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from backend.graph.state import ContentState

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0.4,
    anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
)

SYSTEM_PROMPT = """You are a content marketing strategist.
Build a 4-week content calendar based on the brief, keywords, and brand context.

Return ONLY valid JSON (no markdown):
{
  "monthly_themes": ["Week 1 theme", "Week 2 theme", "Week 3 theme", "Week 4 theme"],
  "content_calendar": [
    {
      "week": 1,
      "channel": "blog",
      "topic": "Exact, specific post title (not generic)",
      "keywords": ["primary keyword", "secondary keyword"],
      "cta": "Specific call-to-action text",
      "notes": "Strategic angle: what hook to use, what pain point to address, how to position vs competitors"
    }
  ]
}

Rules:
- Create entries for ALL 4 weeks for EACH requested channel
- Topics must be specific and actionable (not generic like 'AI tips')
- Weave trending keywords naturally into titles
- Blog: SEO long-form angles (1000+ words implied)
- Social: platform-native (LinkedIn insight, X thread, Instagram carousel)
- Email: value-led subject lines, no clickbait
- Ad: pain-point headline, benefit-driven
- The 'notes' field is critical — give the writer a specific creative direction"""


def planner_node(state: ContentState) -> dict:
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""
Brand: {state['brand_name']}
Audience: {state['target_audience']}
Brief: {state['brief']}
Active channels: {', '.join(state['channels'])}

Trending keywords to use: {', '.join(state['trending_keywords'])}
Competitor gaps to exploit: {', '.join(state['competitor_gaps'])}

Brand voice context (from style guide):
{state['brand_context'][:1500]}

Build the 4-week calendar. Return JSON only.
"""),
    ])

    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', response.content, re.DOTALL)
        parsed = json.loads(match.group()) if match else {
            "monthly_themes": [], "content_calendar": []
        }

    return {
        "monthly_themes":    parsed.get("monthly_themes", []),
        "content_calendar":  parsed.get("content_calendar", []),
        "revision_round":    0,
        "content_pieces":    [],
        "approved_pieces":   [],
        "rejected_pieces":   [],
        "qa_feedback":       {},
        "published_urls":    {},
    }
```

---

## 11. Writer Agents — `backend/agents/writers.py`

Four separate nodes, all in one file. Each runs in parallel via `Send()`.

```python
"""
backend/agents/writers.py

Four LangGraph nodes, one per content channel.
All run in PARALLEL via Send() fan-out from the planner node.

Each node:
  - Reads state['current_calendar_entry'] for the specific topic/keywords/CTA/notes
  - Reads state['brand_context'] for brand voice
  - If state['revision_target'] is set, it's a revision round — use qa_feedback
  - Appends one ContentPiece to state['content_pieces']

Nodes:
  blog_writer_node      → SEO blog post (900-1200 words)
  social_writer_node    → LinkedIn + X thread + Instagram caption
  email_writer_node     → Newsletter with subject, preview, body, CTA, PS
  ad_copy_writer_node   → Google Search ads + Meta ads
"""

import os
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from backend.graph.state import ContentState, ContentPiece

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0.7,
    anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
)


def _build_context(state: ContentState) -> str:
    """Shared context string injected into every writer prompt."""
    entry = state.get("current_calendar_entry", {})
    revision = state.get("revision_target", {})
    feedback = state.get("qa_feedback", {})
    channel = entry.get("channel", "")

    base = f"""
Brand: {state['brand_name']}
Target audience: {state['target_audience']}
Topic: {entry.get('topic', '')}
Focus keywords: {', '.join(entry.get('keywords', []))}
CTA: {entry.get('cta', '')}
Strategic notes: {entry.get('notes', '')}

Brand voice & style guide:
{state['brand_context'][:1200]}
"""
    if revision and feedback.get(channel):
        base += f"""
IMPORTANT — THIS IS A REVISION:
Previous draft was rejected by QA. Fix these specific issues:
{feedback[channel]}

Do NOT repeat the same mistakes. Address every issue listed above.
"""
    return base


# ── BLOG WRITER ─────────────────────────────────────────────────────────────

BLOG_SYSTEM = """You are an expert SEO content writer.
Write a complete, publication-ready blog post.

Structure:
- H1 title (includes primary keyword naturally)
- Meta description (max 155 chars)
- Hook introduction (2-3 sentences that create urgency or curiosity)
- 3-5 H2 sections with substantive content and examples
- Bullet lists where they aid clarity
- Conclusion with clear CTA
- Target: 900-1200 words

Match the brand's tone exactly from the style guide.
Do NOT write generic corporate content. Be specific, useful, and opinionated."""


def blog_writer_node(state: ContentState) -> dict:
    entry = state.get("current_calendar_entry", {})
    response = llm.invoke([
        SystemMessage(content=BLOG_SYSTEM),
        HumanMessage(content=_build_context(state)),
    ])
    revision_count = state.get("revision_target", {}).get("revision_count", 0)
    piece: ContentPiece = {
        "channel": "blog",
        "topic": entry.get("topic", ""),
        "draft": response.content,
        "seo_score": 0.0,
        "qa_passed": False,
        "revision_count": revision_count,
        "published_url": None,
    }
    return {"content_pieces": state.get("content_pieces", []) + [piece]}


# ── SOCIAL WRITER ────────────────────────────────────────────────────────────

SOCIAL_SYSTEM = """You are a social media copywriter who writes platform-native content.
Write THREE versions separated by clear headers:

[LINKEDIN]
Professional insight post. 150-200 words. Opens with a bold statement or surprising stat.
Ends with a question to drive comments. No hashtags in the body.

[X/TWITTER THREAD]
Thread of 4 tweets. Tweet 1 is the hook (make people want to read on).
Each tweet on a new line, numbered 1/ 2/ 3/ 4/
No hashtags. Use line breaks for readability.

[INSTAGRAM]
Strong opening line (stop-the-scroll hook).
Storytelling middle (100-150 words).
Clear CTA.
5 relevant hashtags at the end.

Match the brand voice exactly. Write like a human, not a press release."""


def social_writer_node(state: ContentState) -> dict:
    entry = state.get("current_calendar_entry", {})
    response = llm.invoke([
        SystemMessage(content=SOCIAL_SYSTEM),
        HumanMessage(content=_build_context(state)),
    ])
    revision_count = state.get("revision_target", {}).get("revision_count", 0)
    piece: ContentPiece = {
        "channel": "social",
        "topic": entry.get("topic", ""),
        "draft": response.content,
        "seo_score": 0.0,
        "qa_passed": False,
        "revision_count": revision_count,
        "published_url": None,
    }
    return {"content_pieces": state.get("content_pieces", []) + [piece]}


# ── EMAIL WRITER ─────────────────────────────────────────────────────────────

EMAIL_SYSTEM = """You are an email marketing specialist.
Write a complete newsletter email. Use these exact section headers:

SUBJECT: (max 50 chars — no clickbait, no ALL CAPS)
PREVIEW: (max 90 chars — teases the email without giving it away)

BODY:
Opening hook (1-2 sentences — personal, immediate)
Main content (200-300 words — value-first, second person 'you/your')
CTA BUTTON: [button text here]
P.S. [secondary hook that adds value or curiosity]

Rules:
- No spam trigger words (free, guaranteed, limited time, act now)
- One clear CTA only
- Write like a trusted colleague, not a marketer
- P.S. lines often get more reads than the body — make it count"""


def email_writer_node(state: ContentState) -> dict:
    entry = state.get("current_calendar_entry", {})
    response = llm.invoke([
        SystemMessage(content=EMAIL_SYSTEM),
        HumanMessage(content=_build_context(state)),
    ])
    revision_count = state.get("revision_target", {}).get("revision_count", 0)
    piece: ContentPiece = {
        "channel": "email",
        "topic": entry.get("topic", ""),
        "draft": response.content,
        "seo_score": 0.0,
        "qa_passed": False,
        "revision_count": revision_count,
        "published_url": None,
    }
    return {"content_pieces": state.get("content_pieces", []) + [piece]}


# ── AD COPY WRITER ───────────────────────────────────────────────────────────

AD_SYSTEM = """You are a performance marketing copywriter.
Write ad copy for two platforms. Use exact section headers:

[GOOGLE SEARCH ADS]
Headline 1: (MAX 30 chars — count carefully)
Headline 2: (MAX 30 chars — count carefully)
Headline 3: (MAX 30 chars — count carefully)
Description 1: (MAX 90 chars)
Description 2: (MAX 90 chars)
Display URL path: /path1/path2

[META ADS (Facebook/Instagram)]
Primary text: (first 125 chars are shown before 'See more', make them count)
Full primary text: (up to 500 chars)
Headline: (MAX 27 chars)
Link description: (MAX 27 chars)

Alternative angle B:
Primary text: (different angle — try emotional vs logical)
Headline: (MAX 27 chars)

IMPORTANT: Count characters carefully. Exceeding limits causes ad rejection.
Focus on: pain point → solution → CTA structure.
No misleading claims."""


def ad_copy_writer_node(state: ContentState) -> dict:
    entry = state.get("current_calendar_entry", {})
    response = llm.invoke([
        SystemMessage(content=AD_SYSTEM),
        HumanMessage(content=_build_context(state)),
    ])
    revision_count = state.get("revision_target", {}).get("revision_count", 0)
    piece: ContentPiece = {
        "channel": "ad",
        "topic": entry.get("topic", ""),
        "draft": response.content,
        "seo_score": 0.0,
        "qa_passed": False,
        "revision_count": revision_count,
        "published_url": None,
    }
    return {"content_pieces": state.get("content_pieces", []) + [piece]}
```

---

## 12. QA Agent — `backend/agents/qa_agent.py`

```python
"""
backend/agents/qa_agent.py

Node: qa
Input: state['content_pieces'] (all drafts from parallel writers)
Output: state['approved_pieces'], state['rejected_pieces'], state['qa_feedback']
        state['revision_round'] (incremented), state['content_pieces'] (cleared)

Evaluates each piece against:
1. Brand voice compliance — matches style guide tone, vocabulary, persona
2. SEO score — keywords used naturally, proper H1/meta/structure (blog only)
3. Channel fit — correct length, format, CTA for the channel
4. Quality — genuinely useful to the target audience

Pass threshold: score >= 0.75
Failed pieces get specific, actionable feedback that the writer can act on.
"""

import os
import json
import re
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from backend.graph.state import ContentState, ContentPiece

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0.1,  # Low temperature for consistent, strict evaluation
    anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
)

SYSTEM_PROMPT = """You are a senior content QA editor and brand guardian.
Evaluate each content piece strictly but fairly.

Return ONLY valid JSON (no markdown, no explanation):
{
  "pieces": [
    {
      "channel": "blog|social|email|ad",
      "topic": "exact topic from input",
      "passed": true,
      "seo_score": 0.88,
      "feedback": "",
      "issues": []
    },
    {
      "channel": "email",
      "topic": "exact topic",
      "passed": false,
      "seo_score": 0.61,
      "feedback": "Subject line is too salesy and uses 'guaranteed' which is a spam trigger. The P.S. line is missing entirely. Rewrite the subject to be curiosity-driven and add a P.S. with a secondary hook.",
      "issues": ["spam trigger word in subject", "missing P.S. line"]
    }
  ]
}

Evaluation criteria:
BRAND VOICE (all channels): Matches style guide? Right tone/vocabulary/persona?
SEO (blog only): Keywords used naturally? Clear H1? Meta description present and under 155 chars? H2 structure?
CHANNEL FIT: Right length? Right format? Appropriate CTA?
  - Blog: 900-1200 words, H1/H2 structure, meta description
  - Social: 3 platform versions present, right lengths
  - Email: Subject + Preview + Body + CTA + P.S. all present, no spam words
  - Ad: Headlines MUST be under 30 chars (count carefully), descriptions under 90 chars

QUALITY: Is it genuinely useful and specific? Not generic?

Pass threshold: ALL criteria above 0.75
Feedback must be specific sentences a writer can act on immediately. Not vague like 'improve tone'."""


def qa_node(state: ContentState) -> dict:
    pieces = state.get("content_pieces", [])
    if not pieces:
        return {
            "approved_pieces": state.get("approved_pieces", []),
            "rejected_pieces": [],
            "revision_round": state.get("revision_round", 0) + 1,
            "content_pieces": [],
        }

    pieces_text = "\n\n".join(
        f"=== {p['channel'].upper()} | {p['topic']} ===\n{p['draft']}"
        for p in pieces
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""
Brand: {state['brand_name']}
Audience: {state['target_audience']}

Brand voice reference:
{state['brand_context'][:800]}

Content to evaluate:
{pieces_text}

Evaluate every piece. Return JSON only.
"""),
    ])

    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', response.content, re.DOTALL)
        parsed = json.loads(match.group()) if match else {"pieces": []}

    approved: list[ContentPiece] = []
    rejected: list[ContentPiece] = []
    qa_feedback = {}

    for result in parsed.get("pieces", []):
        matching = next(
            (p for p in pieces if p["channel"] == result["channel"]), None
        )
        if not matching:
            continue

        updated: ContentPiece = {
            **matching,
            "seo_score": result.get("seo_score", 0.0),
            "qa_passed": result.get("passed", False),
            "revision_count": matching.get("revision_count", 0) + 1,
        }

        if result.get("passed", False):
            approved.append(updated)
        else:
            qa_feedback[result["channel"]] = result.get("feedback", "")
            rejected.append(updated)

    return {
        "approved_pieces": state.get("approved_pieces", []) + approved,
        "rejected_pieces": rejected,
        "qa_feedback":     qa_feedback,
        "revision_round":  state.get("revision_round", 0) + 1,
        "content_pieces":  [],  # Clear for next round
    }
```

---

## 13. Publisher Agent — `backend/agents/publisher.py`

```python
"""
backend/agents/publisher.py

Node: publisher
Publishes all approved_pieces (and force-publishes rejected_pieces that hit revision cap).

Routing:
  blog  → Notion (creates page in content database)
  social → Buffer (schedules across connected profiles)
  email  → Notion (saved as draft for manual send review)
  ad     → Notion (saved for manual review — never auto-publish ads)

If NOTION_TOKEN or BUFFER_TOKEN not set, saves content to local files in ./output/
"""

import os
import json
import httpx
from datetime import datetime, timedelta
from pathlib import Path
from backend.graph.state import ContentState, ContentPiece

NOTION_TOKEN       = os.environ.get("NOTION_TOKEN", "")
NOTION_DATABASE_ID = os.environ.get("NOTION_DATABASE_ID", "")
BUFFER_TOKEN       = os.environ.get("BUFFER_TOKEN", "")
BUFFER_PROFILE_IDS = [p for p in os.environ.get("BUFFER_PROFILE_IDS", "").split(",") if p]


def _save_to_file(piece: ContentPiece) -> str:
    """Fallback: save to local ./output/ directory if no API keys set."""
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    filename = f"{piece['channel']}_{piece['topic'][:40].replace(' ', '_')}.md"
    filepath = output_dir / filename
    filepath.write_text(piece["draft"])
    return str(filepath)


def _publish_to_notion(piece: ContentPiece) -> str | None:
    if not NOTION_TOKEN or not NOTION_DATABASE_ID:
        return None
    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }
    payload = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "Name":    {"title": [{"text": {"content": piece["topic"]}}]},
            "Channel": {"select": {"name": piece["channel"].capitalize()}},
            "Status":  {"select": {"name": "Draft"}},
            "QA Score":{"number": round(piece.get("seo_score", 0), 2)},
        },
        "children": [{
            "object": "block", "type": "paragraph",
            "paragraph": {"rich_text": [{"type": "text", "text": {"content": piece["draft"][:2000]}}]}
        }],
    }
    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("url")
    except Exception as e:
        print(f"Notion error: {e}")
    return None


def _publish_to_buffer(piece: ContentPiece, index: int = 0) -> str | None:
    if not BUFFER_TOKEN or not BUFFER_PROFILE_IDS:
        return None
    scheduled = (datetime.utcnow() + timedelta(days=index + 1)).isoformat() + "Z"
    for profile_id in BUFFER_PROFILE_IDS:
        try:
            resp = httpx.post(
                "https://api.bufferapp.com/1/updates/create.json",
                data={
                    "access_token":    BUFFER_TOKEN,
                    "profile_ids[]":   profile_id,
                    "text":            piece["draft"][:500],
                    "scheduled_at":    scheduled,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json().get("updates", [{}])[0].get("id", "buffer://scheduled")
        except Exception as e:
            print(f"Buffer error: {e}")
    return None


def publisher_node(state: ContentState) -> dict:
    all_pieces = state.get("approved_pieces", []) + state.get("rejected_pieces", [])
    published_urls: dict[str, str] = {}

    for i, piece in enumerate(all_pieces):
        channel = piece["channel"]
        url = None

        if channel == "blog":
            url = _publish_to_notion(piece) or _save_to_file(piece)
        elif channel == "social":
            url = _publish_to_buffer(piece, i) or _save_to_file(piece)
        elif channel == "email":
            url = _publish_to_notion(piece) or _save_to_file(piece)
        elif channel == "ad":
            url = _publish_to_notion(piece) or _save_to_file(piece)

        published_urls[f"{channel}_{i}"] = url or "saved-locally"

    return {"published_urls": published_urls}
```

---

## 14. Streamlit UI — `app.py`

This is the complete UI. No FastAPI needed — Streamlit talks directly to LangGraph.

```python
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
                n = ingest("./brand_docs/")
                st.success(f"✅ Indexed {n} chunks")

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
    progress_so_far = 0.0
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

    final_state = {}

    async def run_with_streaming():
        nonlocal progress_so_far, final_state
        async for event in content_graph.astream_events(initial_state, version="v2"):
            kind = event.get("event", "")
            node = event.get("name", "")

            if kind == "on_chain_start" and node in AGENT_LABELS:
                label = AGENT_LABELS[node]
                status_text.info(f"Running: {label}")
                log_lines.append(f"⏳ {label}")
                agent_log.text("\n".join(log_lines[-8:]))

            elif kind == "on_chain_end" and node in NODE_WEIGHTS:
                progress_so_far = min(progress_so_far + NODE_WEIGHTS.get(node, 0.05), 1.0)
                progress_bar.progress(progress_so_far)
                label = AGENT_LABELS.get(node, node)
                log_lines.append(f"✅ {label}")
                agent_log.text("\n".join(log_lines[-8:]))

                # Capture final state from last event
                output = event.get("data", {}).get("output", {})
                if output:
                    final_state.update(output)

        progress_bar.progress(1.0)
        status_text.success("✅ All agents complete!")

    asyncio.run(run_with_streaming())

    # ── Results tabs ─────────────────────────────────────────────────────────
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
            for p in blog_pieces:
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
                        )
        else:
            st.info("Blog posts will appear here.")

    with tab_social:
        social_pieces = [p for p in approved if p["channel"] == "social"]
        if social_pieces:
            for p in social_pieces:
                with st.expander(p["topic"], expanded=True):
                    st.text_area("Copy", p["draft"], height=300, label_visibility="collapsed")
                    st.download_button("⬇️ Download", p["draft"], f"social_{p['topic'][:30]}.txt")
        else:
            st.info("Social copy will appear here.")

    with tab_email:
        email_pieces = [p for p in approved if p["channel"] == "email"]
        if email_pieces:
            for p in email_pieces:
                with st.expander(p["topic"], expanded=True):
                    st.text_area("Email", p["draft"], height=300, label_visibility="collapsed")
                    st.download_button("⬇️ Download", p["draft"], f"email_{p['topic'][:30]}.txt")
        else:
            st.info("Email newsletters will appear here.")

    with tab_ad:
        ad_pieces = [p for p in approved if p["channel"] == "ad"]
        if ad_pieces:
            for p in ad_pieces:
                with st.expander(p["topic"], expanded=True):
                    st.text_area("Ad copy", p["draft"], height=300, label_visibility="collapsed")
                    st.download_button("⬇️ Download", p["draft"], f"ad_{p['topic'][:30]}.txt")
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
```

---

## 15. Docker Compose — `docker-compose.yml`

```yaml
version: "3.9"

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 10s
      timeout: 5s
      retries: 5

  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - SERPER_API_KEY=${SERPER_API_KEY}
      - QDRANT_URL=http://qdrant:6333
      - NOTION_TOKEN=${NOTION_TOKEN}
      - NOTION_DATABASE_ID=${NOTION_DATABASE_ID}
      - BUFFER_TOKEN=${BUFFER_TOKEN}
      - BUFFER_PROFILE_IDS=${BUFFER_PROFILE_IDS}
      - LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}
      - LANGCHAIN_TRACING_V2=${LANGCHAIN_TRACING_V2}
      - LANGCHAIN_PROJECT=${LANGCHAIN_PROJECT}
    depends_on:
      qdrant:
        condition: service_healthy
    volumes:
      - ./brand_docs:/app/brand_docs
      - ./output:/app/output

volumes:
  qdrant_data:
```

---

## 16. Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p brand_docs output

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

---

## 17. Tests — `tests/test_graph.py`

```python
"""tests/test_graph.py"""
import pytest
from unittest.mock import patch, MagicMock
from backend.graph.builder import route_after_qa, fan_out_to_writers
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
            {"week": 1, "channel": "blog",   "topic": "Why AI sounds off-brand", "keywords": ["brand voice AI"], "cta": "Try WriteAI", "notes": "Lead with pain"},
            {"week": 1, "channel": "social", "topic": "5 AI myths",              "keywords": ["AI myths"],       "cta": "Read more",   "notes": "Hook-first"},
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
    rejected = {"channel": "blog", "topic": "T", "draft": "", "seo_score": 0.4,
                "qa_passed": False, "revision_count": 2, "published_url": None}
    state = {**base_state, "rejected_pieces": [rejected], "revision_round": 2}
    assert route_after_qa(state) == "publisher"


def test_route_after_qa_with_rejections(base_state):
    from langgraph.constants import Send
    rejected = {"channel": "blog", "topic": "T", "draft": "", "seo_score": 0.5,
                "qa_passed": False, "revision_count": 0, "published_url": None}
    state = {**base_state, "rejected_pieces": [rejected], "revision_round": 0}
    result = route_after_qa(state)
    assert isinstance(result, list)
    assert any(isinstance(r, Send) for r in result)


def test_fan_out_creates_one_send_per_entry(base_state):
    from langgraph.constants import Send
    result = fan_out_to_writers(base_state)
    assert len(result) == len(base_state["content_calendar"])
    assert all(isinstance(r, Send) for r in result)


@patch("backend.agents.writers.llm")
def test_blog_writer_appends_piece(mock_llm, base_state):
    from backend.agents.writers import blog_writer_node
    mock_llm.invoke.return_value = MagicMock(content="# Test blog post\n\nContent here...")
    state = {**base_state, "current_calendar_entry": base_state["content_calendar"][0]}
    result = blog_writer_node(state)
    assert len(result["content_pieces"]) == 1
    assert result["content_pieces"][0]["channel"] == "blog"
```

---

## 18. .gitignore

```
.env
__pycache__/
*.pyc
.pytest_cache/
brand_docs/*.pdf
brand_docs/*.txt
brand_docs/*.md
!brand_docs/.gitkeep
output/
.streamlit/secrets.toml
qdrant_data/
```

---

## 19. Full Data Flow — What the User Inputs and What Each Agent Does

### User input (4 fields in Streamlit sidebar)
```
brand_name:       "WriteAI"
target_audience:  "B2B content marketers at SaaS companies (50-500 employees)"
brief:            "We're launching WriteAI v2 next month — AI writing with real-time
                  brand voice enforcement. Goal: awareness + trial signups.
                  Competitors: Jasper, Copy.ai. Organic-first, tight budget."
channels:         ["blog", "social", "email", "ad"]
```

### Step 1: RAG Retriever Node
- **Input:** brief + target_audience
- **Action:** Queries Qdrant for TOP_K=6 most relevant brand doc chunks
- **Output added to state:**
  - `brand_context`: concatenated text from brand PDFs
  - `rag_sources`: ["style_guide.pdf", "tone_guide.txt"]

### Step 2: Trend Researcher Node
- **Input:** brief, brand_name, target_audience, channels
- **Action:** Runs 3 Serper web searches, feeds results to Claude for extraction
- **Searches:**
  1. "WriteAI v2 AI writing tools launching 2025 trending"
  2. "Jasper Copy.ai competitors content marketing strategy gaps"
  3. "B2B SaaS content marketer biggest pain points 2025"
- **Output added to state:**
  - `trending_keywords`: ["brand voice AI", "AI content ROI", "AI writing consistency", ...]
  - `competitor_gaps`: ["no one covers ROI measurement", "missing brand safety angle", ...]
  - `search_results`: raw search result dicts

### Step 3: Planner Node
- **Input:** brief, channels, trending_keywords, competitor_gaps, brand_context
- **Action:** Claude builds 4-week calendar as structured JSON
- **Output added to state:**
  - `monthly_themes`: ["The brand voice problem", "AI vs human writing", "WriteAI v2 launch", "Measuring AI content ROI"]
  - `content_calendar`: list of {week, channel, topic, keywords, cta, notes} — one per channel per week = 16 entries for 4 channels

### Step 4: Parallel Writers (via Send() fan-out)
- All 4 writers run SIMULTANEOUSLY
- Each receives: full state + `current_calendar_entry` (their specific topic)
- **Blog writer output:** full blog post with H1, meta, H2 sections, ~1050 words
- **Social writer output:** LinkedIn post + X thread (4 tweets) + Instagram caption with hashtags
- **Email writer output:** Subject + Preview + Body + CTA button + P.S.
- **Ad writer output:** Google Search (3 headlines + 2 descriptions) + Meta (primary text + headline + 2 alternatives)
- All outputs appended to `content_pieces`

### Step 5: QA Agent
- **Input:** all `content_pieces` + brand_context
- **Action:** Claude evaluates each piece with low temperature (0.1) for consistency
- **Scoring criteria:**
  - Brand voice compliance (tone, vocabulary, persona match)
  - SEO structure (blog: H1, meta desc, keyword density)
  - Channel fit (correct lengths, formats, no spam triggers)
  - Quality (specific, useful, not generic)
- **Pass threshold:** score >= 0.75
- **Output added to state:**
  - `approved_pieces`: pieces that passed
  - `rejected_pieces`: pieces that failed with specific feedback
  - `qa_feedback`: {channel: "exact feedback string for writer"}
  - `revision_round`: incremented by 1

### Step 5b: Revision Loop (if rejections exist and revision_round < 2)
- `route_after_qa()` creates new `Send()` for each rejected piece
- Each Send() injects `revision_target` (the failed piece) and QA feedback into the prompt
- Writer re-generates with the feedback in its system prompt
- Goes back to QA for second evaluation
- After 2 rounds, everything is force-approved regardless

### Step 6: Publisher Node
- **Blog → Notion** (creates page in content database)
- **Social → Buffer** (schedules across profiles)
- **Email → Notion** (saved as draft, manual send review)
- **Ad → Notion** (saved for manual review, never auto-publish)
- **Fallback:** if no API keys, saves all content to `./output/` as markdown files
- **Output added to state:** `published_urls`: {channel_index: url}

---

## 20. Deployment Instructions

### Option A: Streamlit Community Cloud (FREE — Recommended for capstone)

1. Push code to GitHub (public repo)
2. Go to https://share.streamlit.io
3. Click "New app" → select your repo → set main file to `app.py`
4. In "Advanced settings" → "Secrets", add all env vars from `.env`
5. **Problem:** Streamlit Cloud can't run a local Qdrant instance

**Solution for Qdrant on Streamlit Cloud:**
- Sign up at https://cloud.qdrant.io (free tier: 1GB)
- Create a cluster, copy the cluster URL and API key
- Update `.env` and Streamlit secrets:
  ```
  QDRANT_URL=https://your-cluster.qdrant.io
  QDRANT_API_KEY=your_key
  ```
- Update `retriever.py` and `ingest.py` to pass `api_key`:
  ```python
  client = QdrantClient(
      url=os.environ.get("QDRANT_URL"),
      api_key=os.environ.get("QDRANT_API_KEY"),
  )
  ```
- Deploy → get a public URL like `https://yourapp.streamlit.app` ← this is your "Live URL" deliverable

### Option B: Railway (Docker — Full stack)

1. Push code to GitHub
2. Sign up at https://railway.app
3. "New Project" → "Deploy from GitHub repo"
4. Railway auto-detects Dockerfile
5. Add environment variables in Railway dashboard (Settings → Variables)
6. For Qdrant: add a second Railway service using `qdrant/qdrant` Docker image
7. Set `QDRANT_URL` to the internal Railway URL of your Qdrant service
8. Railway gives you a public HTTPS URL automatically

### Option C: Local + ngrok (Quickest for demo)

```bash
# Terminal 1: Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Terminal 2: Ingest brand docs
python -m backend.rag.ingest --docs ./brand_docs/

# Terminal 3: Start Streamlit
streamlit run app.py

# Terminal 4: Expose publicly for demo
ngrok http 8501
# Use the ngrok HTTPS URL as your "Live URL"
```

### Option D: Full Docker Compose (Production-like)

```bash
# Build and start everything
docker compose up --build

# In a separate terminal, ingest brand docs
docker compose exec app python -m backend.rag.ingest --docs ./brand_docs/

# App is live at http://localhost:8501
```

---

## 21. Setup Commands (Run in Order)

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and fill environment variables
cp .env.example .env
# Edit .env with your API keys

# 4. Start Qdrant locally
docker run -d -p 6333:6333 qdrant/qdrant

# 5. Add brand documents to brand_docs/ folder
# (PDF, TXT, or MD files — style guide, past content, tone guide)

# 6. Ingest brand documents into Qdrant
python -m backend.rag.ingest --docs ./brand_docs/

# 7. Run tests
pytest tests/ -v

# 8. Start the Streamlit app
streamlit run app.py
# Opens at http://localhost:8501

# 9. For Docker (alternative to steps 4-8)
docker compose up --build
```

---

## 22. Evaluation Rubric Checklist

| Criterion | Points | How this project covers it |
|---|---|---|
| Use-case relevance & impact | 20 | Real SaaS pain: content teams spend 40+ hrs/month manually doing exactly this workflow |
| Agent reasoning & architecture depth | 20 | LangGraph parallel Send(), conditional QA routing, cyclic re-write loop, shared TypedDict state |
| Tool/API integrations quality | 15 | Serper web search, Qdrant RAG, Notion API, Buffer API, Anthropic Claude |
| UI/UX polish | 15 | Streamlit: live progress bar, tabs per channel, download buttons, metrics, calendar dataframe |
| Engineering excellence | 15 | Typed state schema, unit tests, error handling, JSON parse fallback, LangSmith tracing |
| Deployment & DevOps | 10 | Docker Compose, Dockerfile, Railway/Streamlit Cloud deploy, env config |
| Presentation clarity & metrics | 5 | Demo: fill brief → watch agents run → calendar appears → download content in 3 min |

**Total: 100 points**

---

## 23. Demo Script (for video)

1. Open app at live URL
2. Fill sidebar: brand name "WriteAI", audience "B2B SaaS marketers", paste 2-sentence brief
3. Select channels: blog + social
4. Click "Generate content"
5. Show live agent progress updating in real time (highlight parallel writers firing together)
6. Switch to Calendar tab — show the 4-week structured calendar
7. Switch to Blog tab — show a complete blog post, download it
8. Switch to Social tab — show LinkedIn + X thread + Instagram
9. Show Meta tab — highlight QA scores and revision round count
10. Briefly show LangSmith dashboard with the full agent trace
11. Show the live URL / Streamlit Cloud deployment

---

*End of build specification. Hand this document to Claude Code CLI with the instruction: "Build this project exactly as specified."*
