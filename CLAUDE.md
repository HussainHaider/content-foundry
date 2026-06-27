# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

All commands assume the virtualenv is active (`venv\Scripts\activate` on Windows).

```bash
# Install / update dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run a single test
pytest tests/test_graph.py::test_route_after_qa_with_rejections -v

# Tests need dummy keys (modules read env at import time)
OPENAI_API_KEY=dummy ANTHROPIC_API_KEY=dummy SERPER_API_KEY=dummy VOYAGE_API_KEY=dummy pytest tests/ -v

# Ingest brand documents into Qdrant (requires Qdrant running + real keys)
python -m backend.rag.ingest --docs ./brand_docs/

# Start the Streamlit app
streamlit run app.py

# Docker (starts Qdrant + app together)
docker compose up --build
docker compose exec app python -m backend.rag.ingest --docs ./brand_docs/
```

## Architecture

The system is a **LangGraph `StateGraph`** compiled once at import in `backend/graph/builder.py` as `content_graph`. Every agent is a plain Python function that receives the full `ContentState` TypedDict and returns a partial dict of keys to update.

### Data flow

```
START → rag_retriever → trend_researcher → planner
  → [Send() fan-out] → blog_writer
                     → social_writer   (all 4 run in parallel)
                     → email_writer
                     → ad_writer
  → qa
  → [conditional edge: route_after_qa]
       if rejected & revision_round < 2  → fan back to the relevant writer(s) via Send()
       else                              → publisher
  → END
```

### Key patterns

**Fan-out with `Send()`** — `fan_out_to_writers()` in `builder.py` creates one `Send()` per calendar entry, injecting `current_calendar_entry` into the state copy each writer receives. The revision fan-out (`route_after_qa`) does the same, additionally injecting `revision_target` and QA feedback into the writer's `current_calendar_entry.notes`.

**Shared state** — `ContentState` in `backend/graph/state.py` is the only thing passed between agents. Writers append to `content_pieces`; QA splits it into `approved_pieces` / `rejected_pieces` and clears `content_pieces` for the next round. `revision_round` is the loop counter (max 2, enforced in `route_after_qa`).

**RAG** — `rag_retriever_node` queries Qdrant with a VoyageAI `voyage-3` embedding (1024 dims). The collection must exist before running the pipeline; run `backend.rag.ingest` first. If Qdrant is unreachable the node raises — there is no silent fallback.

**Publishing fallback** — `publisher_node` checks for `NOTION_TOKEN` / `BUFFER_TOKEN` at call time (module-level vars). If absent, it saves each piece to `./output/<channel>_<topic>.md` silently.

### Module layout

| Path | Role |
|---|---|
| `backend/llm.py` | `get_llm(temperature)` factory — GPT-4o primary, Claude fallback via `with_fallbacks()` |
| `backend/graph/state.py` | `ContentState` + `ContentPiece` TypedDicts |
| `backend/graph/builder.py` | Graph topology, `route_after_qa`, `fan_out_to_writers`, `content_graph` |
| `backend/rag/ingest.py` | One-shot CLI: load docs → chunk → embed → store in Qdrant |
| `backend/rag/retriever.py` | `rag_retriever_node`: queries Qdrant, populates `brand_context` |
| `backend/agents/` | One file per agent: `trend_researcher`, `planner`, `writers`, `qa_agent`, `publisher` |
| `app.py` | Streamlit UI: sidebar inputs → `content_graph.astream_events()` → live progress + results tabs |

## Environment variables

| Key | Required for |
|---|---|
| `OPENAI_API_KEY` | Primary LLM. If absent, falls back to Anthropic. |
| `OPENAI_MODEL` | OpenAI model name. Defaults to `gpt-4o`. |
| `ANTHROPIC_API_KEY` | Fallback LLM. Used automatically when OpenAI fails. |
| `ANTHROPIC_MODEL` | Anthropic model name. Defaults to `claude-sonnet-4-6`. |
| `VOYAGE_API_KEY` | RAG embeddings (`voyage-3`, voyageai.com) |
| `SERPER_API_KEY` | Trend researcher web search (serper.dev) |
| `QDRANT_URL` | Defaults to `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant Cloud only |
| `NOTION_TOKEN`, `NOTION_DATABASE_ID` | Publishing to Notion |
| `BUFFER_TOKEN`, `BUFFER_PROFILE_IDS` | Publishing to Buffer |
| `LANGSMITH_API_KEY` | LangSmith tracing (optional) |

At least one of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` must be set. Both is recommended.

## Known issues / gotchas

- **`from langgraph.constants import Send`** is deprecated as of LangGraph 1.0; the correct import is `from langgraph.types import Send`. Both `builder.py` and `tests/test_graph.py` still use the old path (works until LangGraph 2.0).
- **LLM factory** (`backend/llm.py`) — `get_llm()` is called at module import in each agent (not lazily). Pytest therefore needs dummy env vars (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `SERPER_API_KEY`, `VOYAGE_API_KEY`) to collect tests; see the dummy-key command above. `with_fallbacks()` is only invoked at call time, not import time.
- The Streamlit `asyncio.run(run_with_streaming())` pattern works for Streamlit ≥ 1.39 but will conflict if a running event loop already exists (e.g. inside Jupyter). Use the app via `streamlit run` only.
