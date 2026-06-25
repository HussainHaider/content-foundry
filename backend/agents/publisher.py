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
