"""LLM access: pick live (Claude Agent SDK) or replay (recorded fixtures)."""

from __future__ import annotations

import os

from rfq_agent.llm.base import LlmProvider
from rfq_agent.llm.live import LiveProvider, live_available
from rfq_agent.llm.replay import ReplayProvider


def forced_replay() -> bool:
    return os.environ.get("DEMO_REPLAY") == "1" or os.environ.get("LLM_MODE", "auto") == "replay"


async def make_provider(replay_delay_scale: float | None = None) -> LlmProvider:
    """LLM_MODE=live|replay|auto (default auto). DEMO_REPLAY=1 always means replay.

    Auto uses live when an API key is set or the Claude Code login is found, else replay.
    """
    if forced_replay():
        return ReplayProvider(delay_scale=replay_delay_scale)
    mode = os.environ.get("LLM_MODE", "auto")
    if mode == "live" or await live_available():
        return LiveProvider()
    return ReplayProvider(delay_scale=replay_delay_scale)
