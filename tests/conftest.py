"""Shared pytest setup.

Several backend modules build LLM / embedding clients at import time, which
requires the corresponding API keys to be present. Tests never make real calls
(everything is mocked), so we inject harmless dummy keys here — before any test
module is collected — so `pytest` works without extra environment plumbing.
"""

import os

for _key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "SERPER_API_KEY", "VOYAGE_API_KEY"):
    os.environ.setdefault(_key, "dummy")
