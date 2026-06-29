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

# Start the Storyblok publishing service (needed for the "Publish to Storyblok" button)
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000

# Docker (starts Qdrant + app + publishing API together)
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

**Agentic tool-calling (`trend_researcher`)** — the trend researcher is a real tool-calling agent, not code-driven search. `backend/agents/tools.py` defines two LangChain `@tool`s — `web_search` (Serper `/search`) and `news_search` (Serper `/news`) — collected in `RESEARCH_TOOLS` / `TOOL_REGISTRY`. The node binds them via `get_llm_with_tools()` and runs a manual loop (`tool_llm.invoke → execute tool_calls → append ToolMessage → repeat`, capped at `MAX_TOOL_ITERS = 5`): **the LLM chooses the queries**. After the loop, a plain (un-bound) `get_llm()` call extracts the structured `trending_keywords` / `competitor_gaps` JSON. Output keys are unchanged, so `planner` and `ContentState` need no edits. `search_results` records the model-chosen queries for audit.

**Shared state** — `ContentState` in `backend/graph/state.py` is the only thing passed between agents. Writers append to `content_pieces`; QA splits it into `approved_pieces` / `rejected_pieces` and clears `content_pieces` for the next round. `revision_round` is the loop counter (max 2, enforced in `route_after_qa`).

**RAG** — `rag_retriever_node` queries Qdrant with a VoyageAI `voyage-3` embedding (1024 dims). The collection must exist before running the pipeline; run `backend.rag.ingest` first. If Qdrant is unreachable the node raises — there is no silent fallback.

**Publishing fallback** — `publisher_node` checks for `NOTION_TOKEN` / `BUFFER_TOKEN` at call time (module-level vars). If absent, it saves each piece to `./output/<channel>_<topic>.md` silently.

**Storyblok publishing (isolated, outside the graph)** — a separate **FastAPI service** (`backend/api/main.py`) exposes `POST /publish/storyblok`. The Streamlit "Publish to Storyblok" button (in the Blog tab) calls it over HTTP via `backend/publishing/publisher_client.py` with an `X-API-Key`. The FastAPI process is the **only** place the Storyblok Management token lives — Streamlit never sees it. Publishing logic is in `backend/publishing/storyblok/` (config → client → schema discovery → markdown→`text`-bloks → mapper → service). It discovers the space's component schema at runtime (no hardcoding, no dependency on the separate `content-foundry-ui` repo), maps each weekly blog onto a `page` of existing `text` bloks, and creates one **draft** story per blog (idempotent via deterministic slugs). This path does **not** touch the LangGraph graph or any existing agent.

### Module layout

| Path | Role |
|---|---|
| `backend/llm.py` | `get_llm(temperature)` factory — GPT-4o primary, Claude fallback via `with_fallbacks()`; `get_llm_with_tools(tools, temperature)` binds tools per-LLM before the fallback wrap |
| `backend/agents/tools.py` | LangChain `@tool`s exposed to the LLM: `web_search` + `news_search` (both Serper), `RESEARCH_TOOLS` / `TOOL_REGISTRY` |
| `backend/graph/state.py` | `ContentState` + `ContentPiece` TypedDicts |
| `backend/graph/builder.py` | Graph topology, `route_after_qa`, `fan_out_to_writers`, `content_graph` |
| `backend/rag/ingest.py` | One-shot CLI: load docs → chunk → embed → store in Qdrant |
| `backend/rag/retriever.py` | `rag_retriever_node`: queries Qdrant, populates `brand_context` |
| `backend/agents/` | One file per agent: `trend_researcher`, `planner`, `writers`, `qa_agent`, `publisher` |
| `backend/api/` | FastAPI publishing service (`main.py`, `models.py`) — owns the Storyblok token |
| `backend/publishing/storyblok/` | Storyblok logic: `config`, `client`, `schema`, `markdown_blocks`, `mapper`, `service` |
| `backend/publishing/publisher_client.py` | Streamlit → FastAPI HTTP client (no Storyblok token) |
| `app.py` | Streamlit UI: sidebar inputs → `content_graph.astream_events()` → live progress + results tabs |

## Environment variables

| Key | Required for |
|---|---|
| `OPENAI_API_KEY` | Primary LLM. If absent, falls back to Anthropic. |
| `OPENAI_MODEL` | OpenAI model name. Defaults to `gpt-4o`. |
| `ANTHROPIC_API_KEY` | Fallback LLM. Used automatically when OpenAI fails. |
| `ANTHROPIC_MODEL` | Anthropic model name. Defaults to `claude-sonnet-4-6`. |
| `VOYAGE_API_KEY` | RAG embeddings (`voyage-3`, voyageai.com) |
| `SERPER_API_KEY` | Trend researcher `web_search` + `news_search` tools (serper.dev) |
| `QDRANT_URL` | Defaults to `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant Cloud only |
| `NOTION_TOKEN`, `NOTION_DATABASE_ID` | Publishing to Notion |
| `BUFFER_TOKEN`, `BUFFER_PROFILE_IDS` | Publishing to Buffer |
| `STORYBLOK_MANAGEMENT_TOKEN` | Storyblok write token — **FastAPI service only** (never set in the Streamlit process) |
| `STORYBLOK_SPACE_ID` | Target Storyblok space (FastAPI service) |
| `STORYBLOK_REGION` | Storyblok MAPI host: `eu` (default), `us`, `ap`, `ca`, `cn` |
| `STORYBLOK_BLOG_PARENT_ID` | Optional folder story id to nest blogs under |
| `PUBLISHER_API_KEY` | Shared internal key between Streamlit and the FastAPI publisher (both processes) |
| `PUBLISHER_API_URL` | FastAPI base URL seen by Streamlit (`http://localhost:8000` local, `http://api:8000` in compose) |
| `LANGSMITH_API_KEY` | LangSmith tracing (optional) |

At least one of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` must be set. Both is recommended.

## Known issues / gotchas

- **`Send` import** — use `from langgraph.types import Send` (the `langgraph.constants` path is deprecated in LangGraph 1.0 and removed in 2.0). `builder.py` and `tests/test_graph.py` use the `langgraph.types` path.
- **Text splitter import** — `RecursiveCharacterTextSplitter` is imported from `langchain_text_splitters` (the old `langchain.text_splitter` shim was removed in LangChain 1.0). See `backend/rag/ingest.py`.
- **LLM factory** (`backend/llm.py`) — `get_llm()` is called at module import in each agent (not lazily). Pytest therefore needs dummy env vars (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `SERPER_API_KEY`, `VOYAGE_API_KEY`) to collect tests; see the dummy-key command above. `with_fallbacks()` is only invoked at call time, not import time.
- **Tool binding + fallbacks** — `get_llm_with_tools()` binds tools to each underlying LLM *before* `with_fallbacks()`, because `RunnableWithFallbacks` has **no `.bind_tools()`**. Don't try to call `.bind_tools()` on the result of `get_llm()`.
- The Streamlit `asyncio.run(run_with_streaming())` pattern works for Streamlit ≥ 1.39 but will conflict if a running event loop already exists (e.g. inside Jupyter). Use the app via `streamlit run` only.
- **Storyblok token rotation** — the `content-foundry-ui` repo's `.mcp.json` previously contained a literal `sb_pat_...` token; it has been replaced with `${STORYBLOK_MCP_TOKEN}` interpolation. The previously-committed token must be **rotated** in the Storyblok UI (treat as compromised). Use a *separate* write-scoped Management token for `STORYBLOK_MANAGEMENT_TOKEN` (the FastAPI service), distinct from the MCP dev token.
- **Storyblok publishing tests** (`tests/test_storyblok_publishing.py`) mock httpx and do not import the graph, so they collect without LLM keys. The Storyblok modules read env lazily (`StoryblokConfig.from_env()`), so no dummy `STORYBLOK_*` keys are needed to import them.
