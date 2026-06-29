"""
backend/agents/tools.py

LangChain @tool definitions that agents bind for tool-calling. These are the
project's external tool/API integrations exposed to the LLM (the model decides
when and what to search):

  - web_search  → Serper /search  (general web)
  - news_search → Serper /news    (fresh, time-sensitive trends)

Both reuse SERPER_API_KEY. The GoogleSerperAPIWrapper is instantiated lazily
inside each tool so the module imports without a real key and tests can patch
the wrapper. Errors are swallowed into a readable string so a failed search
never crashes the graph.
"""

import os
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper


def _serper_key() -> str:
    return os.environ.get("SERPER_API_KEY", "")


@tool
def web_search(query: str) -> str:
    """Search the general web for information on a topic. Use this for SEO
    keyword discovery, competitor content, and audience pain points."""
    try:
        return GoogleSerperAPIWrapper(serper_api_key=_serper_key()).run(query)
    except Exception as exc:  # noqa: BLE001 - never crash the graph on a bad search
        return f"Search unavailable for '{query}': {exc}"


@tool
def news_search(query: str) -> str:
    """Search recent news articles for fresh, time-sensitive trends and
    current events related to a topic."""
    try:
        return GoogleSerperAPIWrapper(type="news", serper_api_key=_serper_key()).run(
            query
        )
    except Exception as exc:  # noqa: BLE001 - never crash the graph on a bad search
        return f"News search unavailable for '{query}': {exc}"


RESEARCH_TOOLS = [web_search, news_search]
TOOL_REGISTRY = {t.name: t for t in RESEARCH_TOOLS}
