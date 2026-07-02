"""
backend/llm.py
Central LLM factory with OpenAI primary + Anthropic fallback.

Usage in agents:
    from backend.llm import get_llm
    llm = get_llm(temperature=0.7)

Fallback behaviour (LangChain with_fallbacks):
  - Any exception from OpenAI (rate limit, quota, API error, timeout) automatically
    retries the same prompt against Claude claude-sonnet-4-6.
  - If OPENAI_API_KEY is not set, falls back to Anthropic directly with no retry.
  - If neither key is set, raises at call time (not at import time).
"""

import os

from langchain_core.runnables import Runnable


def _build_llms(temperature: float):
    """Construct the (anthropic, openai_or_none) chat models from env."""
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    anthropic_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    from langchain_anthropic import ChatAnthropic

    anthropic_llm = ChatAnthropic(
        model=anthropic_model,
        temperature=temperature,
        anthropic_api_key=anthropic_key,
    )

    if not openai_key:
        return anthropic_llm, None

    from langchain_openai import ChatOpenAI

    openai_llm = ChatOpenAI(
        model=openai_model,
        temperature=temperature,
        api_key=openai_key,
    )
    return anthropic_llm, openai_llm


def get_llm(temperature: float = 0.7) -> Runnable:
    anthropic_llm, openai_llm = _build_llms(temperature)
    if openai_llm is None:
        return anthropic_llm

    # Any exception from OpenAI triggers transparent retry against Anthropic
    return openai_llm.with_fallbacks([anthropic_llm])


def get_llm_with_tools(tools: list, temperature: float = 0.7) -> Runnable:
    """Like get_llm(), but with the given tools bound for tool-calling.

    Tools must be bound to each *underlying* LLM before wrapping in
    with_fallbacks(), because RunnableWithFallbacks has no .bind_tools().
    """
    anthropic_llm, openai_llm = _build_llms(temperature)
    anthropic_bound = anthropic_llm.bind_tools(tools)
    if openai_llm is None:
        return anthropic_bound

    return openai_llm.bind_tools(tools).with_fallbacks([anthropic_bound])
