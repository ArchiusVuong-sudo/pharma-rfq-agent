"""Provider interface shared by the live (Claude Agent SDK) and replay (recorded fixtures) clients."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

Task = Literal["extract_quote", "draft_follow_up"]


class LlmRequest(BaseModel):
    task: Task
    system: str
    prompt: str
    output_schema: dict[str, Any]
    tools: list[str] = Field(default_factory=list)  # app tool names the model may call
    max_turns: int = 4

    def fixture_key(self) -> str:
        """Stable id of this exact request: same inputs, same recorded answer."""
        blob = json.dumps(
            {
                "task": self.task,
                "system": self.system,
                "prompt": self.prompt,
                "schema": self.output_schema,
                "tools": self.tools,
            },
            sort_keys=True,
        )
        return f"{self.task}-{hashlib.sha256(blob.encode()).hexdigest()[:16]}"


class ToolCall(BaseModel):
    name: str
    input: dict[str, Any]
    output: str
    is_error: bool = False
    latency_ms: float = 0.0
    start_ms: float = 0.0  # wall clock (epoch ms) when the tool ran


class LlmResponse(BaseModel):
    output: dict[str, Any] | None
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0  # API-equivalent cost reported by the SDK, not a bill
    latency_ms: float = 0.0
    tool_calls: list[ToolCall] = Field(default_factory=list)
    transport_retries: int = 0
    source: Literal["live", "replay"] = "live"
    recorded_at: str | None = None


class LlmError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class FixtureMissing(LlmError):
    pass


class LlmProvider(Protocol):
    mode: Literal["live", "replay"]
    model: str

    async def complete(self, request: LlmRequest) -> LlmResponse: ...
