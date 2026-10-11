"""Replay provider: answers from responses recorded on real live runs (fixtures/llm/*.json).

Recorded tool calls are executed again against today's tools so traces show real tool spans.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from rfq_agent.llm.app_tools import call_tool
from rfq_agent.llm.base import FixtureMissing, LlmProvider, LlmRequest, LlmResponse

FIXTURE_DIR = Path(os.environ.get("RFQ_FIXTURE_DIR", Path(__file__).resolve().parents[3] / "fixtures" / "llm"))


def fixture_path(request: LlmRequest, root: Path = FIXTURE_DIR) -> Path:
    return root / f"{request.fixture_key()}.json"


class ReplayProvider:
    mode: Literal["live", "replay"] = "replay"

    def __init__(self, root: Path = FIXTURE_DIR, delay_scale: float | None = None) -> None:
        self.root = root
        self.model = "replay"
        # Replays wait a share of the recorded latency so the timeline streams like a live run.
        self.delay_scale = (
            float(os.environ.get("REPLAY_DELAY_SCALE", "0")) if delay_scale is None else delay_scale
        )

    async def complete(self, request: LlmRequest) -> LlmResponse:
        path = fixture_path(request, self.root)
        if not path.exists():
            raise FixtureMissing(
                f"no recorded response for {request.task} ({path.name}); run scripts/record_fixtures.py"
            )
        start = time.perf_counter()
        data = json.loads(path.read_text(encoding="utf-8"))
        recorded = LlmResponse.model_validate(data["response"])
        if self.delay_scale > 0:
            await asyncio.sleep(min(recorded.latency_ms / 1000 * self.delay_scale, 8.0))
        calls = [await call_tool(c.name, c.input) for c in recorded.tool_calls]
        self.model = recorded.model
        return recorded.model_copy(
            update={
                "tool_calls": calls,
                "source": "replay",
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            }
        )


class RecordingProvider:
    """Wraps the live provider and writes every response as a fixture."""

    mode: Literal["live", "replay"] = "live"

    def __init__(self, inner: LlmProvider, root: Path = FIXTURE_DIR) -> None:
        self.inner = inner
        self.root = root
        self.model = inner.model
        self.written: list[Path] = []

    async def complete(self, request: LlmRequest) -> LlmResponse:
        response = await self.inner.complete(request)
        response.recorded_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.root.mkdir(parents=True, exist_ok=True)
        path = fixture_path(request, self.root)
        payload = {
            "key": request.fixture_key(),
            "task": request.task,
            "note": "Recorded from a live Claude Agent SDK run on synthetic data.",
            "prompt_preview": request.prompt[:160],
            "response": response.model_dump(mode="json"),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.written.append(path)
        return response
