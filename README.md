# AI Content Marketing Engine

A **production-grade multi-agent AI system** that turns a single brand brief into a full month of marketing content. Built on LangGraph with a parallel writer fan-out, a QA → revision loop, RAG-backed brand voice, web-search trend research, and one-click publishing to Notion and Buffer.

## What it does

1. Takes a user's brand brief as input
2. Researches trending topics via web search (Serper)
3. Retrieves brand voice context from uploaded documents (RAG via Qdrant)
4. Builds a 4-week content calendar
5. Generates blog posts, social copy, email newsletters, and ad copy **in parallel**
6. Runs a QA agent that scores each piece and sends failed pieces back for revision (max 2 rounds)
7. Publishes approved content to Notion and Buffer (falls back to local `./output/` files)
8. Streams live progress to the Streamlit UI as each agent completes

## Tech stack

- **LangGraph** — multi-agent state machine (parallel `Send()` fan-out, conditional QA routing, cyclic re-write loop)
- **Anthropic Claude** (`claude-sonnet-4-6`) — the LLM behind every agent
- **Qdrant** — vector database for RAG
- **Streamlit** — UI
- **Serper API** — web search
- **Notion API + Buffer API** — publishing
- **Docker Compose** — local + deployment
- **Railway / Streamlit Cloud** — cloud deployment

## Architecture

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

All agents read from and write to a single shared `ContentState` TypedDict (see `backend/graph/state.py`).

## Project structure

```
content-marketing-engine/
├── backend/
│   ├── agents/        # trend_researcher, planner, writers, qa_agent, publisher
│   ├── rag/           # ingest, retriever
│   └── graph/         # state, builder
├── app.py             # Streamlit entry point
├── brand_docs/        # Drop brand PDFs/TXTs/MDs here
├── tests/             # test_graph.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## Setup (run in order)

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

## Environment variables

| Key | Required | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Claude (all agents) |
| `VOYAGE_API_KEY` | ✅ | RAG embeddings (`voyage-3`, 1024 dims) — key at voyageai.com |
| `SERPER_API_KEY` | ✅ | Web search — free key at serper.dev (2500/month) |
| `QDRANT_URL` | ✅ | `http://localhost:6333` locally; cluster URL for cloud |
| `QDRANT_API_KEY` | for cloud | Only needed for Qdrant Cloud |
| `NOTION_TOKEN`, `NOTION_DATABASE_ID` | optional | Publishing — app works without these |
| `BUFFER_TOKEN`, `BUFFER_PROFILE_IDS` | optional | Publishing — app works without these |
| `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT` | optional | LangSmith observability |

If publishing keys are absent, the publisher saves all content to `./output/` as Markdown.

## Deployment

### Option A — Streamlit Community Cloud (free)
1. Push to a public GitHub repo
2. https://share.streamlit.io → New app → main file `app.py`
3. Add all env vars under Advanced settings → Secrets
4. Use **Qdrant Cloud** (https://cloud.qdrant.io, free 1GB) since Streamlit Cloud can't host local Qdrant — set `QDRANT_URL` and `QDRANT_API_KEY` in secrets

### Option B — Railway (Docker, full stack)
1. Push to GitHub → https://railway.app → Deploy from GitHub repo (auto-detects Dockerfile)
2. Add env vars in Settings → Variables
3. Add a second Railway service using the `qdrant/qdrant` image; point `QDRANT_URL` at its internal URL

### Option C — Local + ngrok (quickest demo)
```bash
docker run -p 6333:6333 qdrant/qdrant             # Terminal 1
python -m backend.rag.ingest --docs ./brand_docs/  # Terminal 2
streamlit run app.py                               # Terminal 3
ngrok http 8501                                    # Terminal 4
```

### Option D — Full Docker Compose
```bash
docker compose up --build
docker compose exec app python -m backend.rag.ingest --docs ./brand_docs/
# App live at http://localhost:8501
```

## Testing

```bash
pytest tests/ -v
```

The test suite covers the QA routing logic, the writer fan-out, and the blog writer node (with the LLM mocked).
