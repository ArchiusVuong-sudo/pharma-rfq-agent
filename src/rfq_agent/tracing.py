"""Run tracer: node, tool and LLM spans with latency, model, tokens and API-equivalent cost."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from rfq_agent.llm.base import LlmResponse
from rfq_agent.store import Store


def _ms() -> float:
    return time.time() * 1000


@dataclass
class Span:
    id: str
    attrs: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"


class Tracer:
    def __init__(self, store: Store, run_id: str) -> None:
        self.store = store
        self.run_id = run_id

    @asynccontextmanager
    async def span(self, kind: str, name: str, parent: Span | None = None,
                   **attrs: Any) -> AsyncIterator[Span]:
        span = Span(id=uuid.uuid4().hex[:12], attrs=dict(attrs))
        start = _ms()
        self.store.x(
            "INSERT INTO spans (id, run_id, parent_id, kind, name, start_ms, status, attrs) "
            "VALUES (?, ?, ?, ?, ?, ?, 'running', ?)",
            (span.id, self.run_id, parent.id if parent else None, kind, name, start, json.dumps(attrs)),
        )
        try:
            yield span
        except BaseException as exc:
            span.status = "error"
            span.attrs.setdefault("error", f"{type(exc).__name__}: {exc}"[:300])
            raise
        finally:
            self.store.x(
                "UPDATE spans SET end_ms = ?, status = ?, attrs = ? WHERE id = ?",
                (_ms(), span.status, json.dumps(span.attrs, default=str), span.id),
            )

    def record(self, kind: str, name: str, parent: Span | None, start_ms: float, end_ms: float,
               status: str = "ok", **attrs: Any) -> None:
        self.store.x(
            "INSERT INTO spans (id, run_id, parent_id, kind, name, start_ms, end_ms, status, attrs) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (uuid.uuid4().hex[:12], self.run_id, parent.id if parent else None, kind, name,
             start_ms, end_ms, status, json.dumps(attrs, default=str)),
        )

    def llm_done(self, span: Span, response: LlmResponse, attempt: int) -> None:
        """Fill an LLM span and add child spans for the tool calls the model made."""
        span.attrs.update(
            model=response.model,
            source=response.source,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cache_read_tokens=response.cache_read_tokens,
            cost_usd=round(response.cost_usd, 6),
            attempt=attempt,
            transport_retries=response.transport_retries,
            recorded_at=response.recorded_at,
        )
        self.store.add_run_usage(self.run_id, response.cost_usd, response.input_tokens, response.output_tokens)
        for call in response.tool_calls:
            start = call.start_ms or _ms()
            self.record(
                "tool", call.name, span, start, start + max(call.latency_ms, 1.0),
                "error" if call.is_error else "ok", input=call.input, output=call.output[:600],
            )
