"""The only tools the model can call. Inputs are validated with Pydantic; errors go back to the model."""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from rfq_agent.catalog import CatalogQuery, search
from rfq_agent.llm.base import ToolCall


@dataclass(frozen=True)
class AppTool:
    name: str
    description: str
    input_model: type[BaseModel]
    run: Callable[[Any], Awaitable[Any]]

    def json_schema(self) -> dict[str, Any]:
        return self.input_model.model_json_schema()


async def _search_catalog(q: CatalogQuery) -> list[dict[str, Any]]:
    return [h.model_dump() for h in search(q)]


SEARCH_CATALOG = AppTool(
    name="search_catalog",
    description=(
        "Search the product catalog with the product text from the email. Returns up to `limit` "
        "matches with product_id, name, unit, pack sizes and a match score (0-100). "
        "Use the product_id of a match only when name and strength agree with the email."
    ),
    input_model=CatalogQuery,
    run=_search_catalog,
)

TOOLS: dict[str, AppTool] = {SEARCH_CATALOG.name: SEARCH_CATALOG}


async def call_tool(name: str, raw_input: dict[str, Any]) -> ToolCall:
    """Validate the input, run the tool and capture its output, error and latency."""
    start = time.perf_counter()
    wall = time.time() * 1000
    tool = TOOLS.get(name)
    if tool is None:
        return ToolCall(name=name, input=raw_input, output=f"unknown tool {name}", is_error=True, start_ms=wall)
    try:
        args = tool.input_model.model_validate(raw_input)
        output = json.dumps(await tool.run(args))
        is_error = False
    except ValidationError as exc:
        output = "invalid input: " + "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        is_error = True
    except Exception as exc:  # a tool failure is reported to the model, never raised into the loop
        output = f"tool error: {type(exc).__name__}"
        is_error = True
    return ToolCall(
        name=name,
        input=raw_input,
        output=output,
        is_error=is_error,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
        start_ms=wall,
    )
