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
from langchain_core.messages import SystemMessage, HumanMessage
from backend.graph.state import ContentState, ContentPiece
from backend.llm import get_llm

llm = get_llm(temperature=0.1)  # Low temperature for consistent, strict evaluation

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
            "content_pieces": None,
        }

    pieces_index = "\n".join(
        f"{i}. channel={p['channel']} topic={p['topic']}"
        for i, p in enumerate(pieces)
    )
    pieces_text = "\n\n".join(
        f"=== [{i}] {p['channel'].upper()} | {p['topic']} ===\n{p['draft']}"
        for i, p in enumerate(pieces)
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""
Brand: {state['brand_name']}
Audience: {state['target_audience']}

Brand voice reference:
{state['brand_context'][:800]}

Pieces to evaluate (echo the exact topic string in your JSON response):
{pieces_index}

Content:
{pieces_text}

Evaluate every piece. In your JSON, the "topic" field MUST exactly match the topic listed above. Return JSON only.
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
            (p for p in pieces
             if p["channel"] == result["channel"] and p["topic"] == result.get("topic", p["topic"])),
            None,
        )
        if matching is None:
            # fallback: match channel only (handles topic truncation by LLM)
            matched_channels = [p for p in pieces if p["channel"] == result["channel"]]
            used_topics = {p["topic"] for p in approved + rejected}
            matching = next((p for p in matched_channels if p["topic"] not in used_topics), None)
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
        "content_pieces":  None,  # None triggers reset via reducer
    }
