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
