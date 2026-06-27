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
from langchain_core.messages import SystemMessage, HumanMessage
from backend.graph.state import ContentState
from backend.llm import get_llm

llm = get_llm(temperature=0.4)

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
