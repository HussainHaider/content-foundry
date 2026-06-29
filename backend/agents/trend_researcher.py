"""
backend/agents/trend_researcher.py

Node: trend_researcher
Input state keys used: brief, brand_name, target_audience, channels
Output state keys set: trending_keywords, competitor_gaps, search_results

Runs as a tool-calling agent: the LLM is bound two Serper-backed tools
(web_search, news_search) and decides for itself what/when to search via a
manual bind_tools loop. After researching, a final un-bound LLM call extracts
structured insights (top keywords + competitor content gaps) as JSON.
"""

import json
import re
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from backend.graph.state import ContentState
from backend.agents.tools import RESEARCH_TOOLS, TOOL_REGISTRY
from backend.llm import get_llm, get_llm_with_tools

# LLM with the research tools bound (model drives the searches)
tool_llm = get_llm_with_tools(RESEARCH_TOOLS, temperature=0.3)
# Plain LLM for the final structured-JSON step (no tools → can't loop)
llm = get_llm(temperature=0.3)

# Safety cap on the tool-calling loop.
MAX_TOOL_ITERS = 5

RESEARCH_PROMPT = """You are a senior SEO content strategist.
Use the web_search and news_search tools to research the brand, its audience,
competitor content, trending topics, and audience pain points. Make several
targeted searches (mix general web and news) before you stop. When you have
gathered enough, stop calling tools and reply with a short plain-text summary
of what you found."""

EXTRACT_PROMPT = """Based on the research above, return ONLY valid JSON
(no markdown, no explanation):
{
  "trending_keywords": ["keyword1", ...],   // exactly 10 high-value SEO keyword
  "competitor_gaps":   ["gap1", ...],        // exactly 5 topics competitors are missing
  "summary": "2-3 sentence opportunity overview"
}"""


def trend_researcher_node(state: ContentState) -> dict:
    messages = [
        SystemMessage(content=RESEARCH_PROMPT),
        HumanMessage(content=f"""
Brand: {state['brand_name']}
Audience: {state['target_audience']}
Brief: {state['brief']}
Channels: {', '.join(state['channels'])}

Research this opportunity, then summarize your findings.
"""),
    ]

    # ── Tool-calling loop: the model picks the queries ──────────────
    search_results: list[dict] = []
    for _ in range(MAX_TOOL_ITERS):
        response = tool_llm.invoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            break

        for call in tool_calls:
            tool = TOOL_REGISTRY.get(call["name"])
            query = call.get("args", {}).get("query", "")
            if tool is None:
                result = f"Unknown tool: {call['name']}"
            else:
                result = tool.invoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
            search_results.append({"query": query, "result": result})

    # ── Final structured extraction (un-bound LLM, JSON only) ───────
    response = llm.invoke(messages + [HumanMessage(content=EXTRACT_PROMPT)])

    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback if LLM adds markdown
        match = re.search(r'\{.*\}', response.content, re.DOTALL)
        parsed = json.loads(match.group()) if match else {
            "trending_keywords": [], "competitor_gaps": []
        }

    return {
        "trending_keywords": parsed.get("trending_keywords", []),
        "competitor_gaps":   parsed.get("competitor_gaps", []),
        "search_results":    search_results,
    }
