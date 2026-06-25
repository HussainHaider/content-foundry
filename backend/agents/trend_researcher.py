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
from langchain_community.utilities import GoogleSerperAPIWrapper
from backend.graph.state import ContentState

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0.3,
    anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
)

search_tool = GoogleSerperAPIWrapper(serper_api_key=os.environ.get("SERPER_API_KEY", ""))

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
