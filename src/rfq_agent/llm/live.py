"""Live provider: the Claude Agent SDK on the local Claude Code login (or ANTHROPIC_API_KEY).

Each call is sandboxed: no built-in Claude Code tools, only this app's tools through an in-process
MCP server, no settings files, strict MCP config, no session persistence, a fresh temp cwd, a turn
cap, a permission mode that denies anything not pre-approved, and a child env from an allowlist.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import tempfile
import time
import warnings
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

import claude_agent_sdk
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    PermissionResultDeny,
    ResultMessage,
    SdkMcpTool,
    create_sdk_mcp_server,
    query,
)

from rfq_agent.llm.app_tools import TOOLS, call_tool
from rfq_agent.llm.base import LlmError, LlmRequest, LlmResponse, ToolCall

ENV_ALLOWLIST = ("PATH", "HOME", "USER", "LANG", "TMPDIR", "ANTHROPIC_API_KEY")
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}
RETRY_ERRORS = {"rate_limit", "server_error"}
WRAPPER = Path(__file__).with_name("clean_env_cli.sh")
MCP_SERVER = "rfq"

warnings.filterwarnings("ignore", message="can_use_tool will not be invoked")


def bundled_cli() -> str:
    path = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
    if path.exists():
        return str(path)
    found = shutil.which("claude")
    if not found:
        raise LlmError("Claude Code CLI not found")
    return found


def child_env() -> dict[str, str]:
    """Only allowlisted variables reach the CLI; the wrapper script drops everything else."""
    env = {k: os.environ[k] for k in ENV_ALLOWLIST if os.environ.get(k)}
    env["RFQ_AGENT_REAL_CLI"] = os.environ.get("RFQ_AGENT_CLI_PATH") or bundled_cli()
    return env


def mcp_name(tool: str) -> str:
    return f"mcp__{MCP_SERVER}__{tool}"


async def _deny_everything(name: str, _input: dict[str, Any], _ctx: Any) -> PermissionResultDeny:
    return PermissionResultDeny(message=f"{name} is not available in this app")


def _sdk_tools(names: list[str], calls: list[ToolCall]) -> list[SdkMcpTool[Any]]:
    out: list[SdkMcpTool[Any]] = []
    for name in names:
        app_tool = TOOLS[name]

        async def handler(args: dict[str, Any], _name: str = name) -> dict[str, Any]:
            result = await call_tool(_name, args)
            calls.append(result)
            return {"content": [{"type": "text", "text": result.output}], "is_error": result.is_error}

        out.append(
            SdkMcpTool(
                name=name,
                description=app_tool.description,
                input_schema=app_tool.json_schema(),
                handler=handler,
            )
        )
    return out


class LiveProvider:
    mode: Literal["live", "replay"] = "live"

    def __init__(
        self,
        model: str | None = None,
        *,
        timeout_s: float = 180.0,
        max_attempts: int = 4,
        concurrency: int = 4,
        max_budget_usd: float = 0.5,
    ) -> None:
        self.model = model or os.environ.get("RFQ_MODEL", "claude-sonnet-5-5")
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.max_budget_usd = max_budget_usd
        self._sem = asyncio.Semaphore(concurrency)

    def options(self, request: LlmRequest, cwd: str, calls: list[ToolCall]) -> ClaudeAgentOptions:
        mcp_servers: dict[str, Any] = {}
        if request.tools:
            mcp_servers[MCP_SERVER] = create_sdk_mcp_server(
                MCP_SERVER, "0.1.0", _sdk_tools(request.tools, calls)
            )
        return ClaudeAgentOptions(
            tools=[],  # no built-in Claude Code tools at all
            mcp_servers=mcp_servers,
            allowed_tools=[mcp_name(t) for t in request.tools],
            strict_mcp_config=True,
            setting_sources=[],  # never load user or project settings, hooks or CLAUDE.md
            permission_mode="dontAsk",  # anything not pre-approved is denied
            can_use_tool=_deny_everything,
            system_prompt=request.system,
            output_format={"type": "json_schema", "schema": request.output_schema},
            model=self.model,
            max_turns=request.max_turns,
            max_budget_usd=self.max_budget_usd,
            cwd=cwd,
            cli_path=str(WRAPPER),
            env=child_env(),
            extra_args={"no-session-persistence": None},
        )

    async def complete(self, request: LlmRequest) -> LlmResponse:
        retries = 0
        delay = 2.0
        async with self._sem:
            while True:
                try:
                    response = await asyncio.wait_for(self._once(request), self.timeout_s)
                    response.transport_retries = retries
                    return response
                except TimeoutError as exc:
                    error = LlmError(f"timed out after {self.timeout_s:.0f}s", retryable=True)
                    error.__cause__ = exc
                except LlmError as exc:
                    error = exc
                except claude_agent_sdk.ClaudeSDKError as exc:
                    text = str(exc).lower()
                    retryable = any(w in text for w in ("overloaded", "rate limit", "529", "503", "500"))
                    error = LlmError(f"SDK error: {type(exc).__name__}", retryable=retryable)
                if not error.retryable or retries + 1 >= self.max_attempts:
                    raise error
                retries += 1
                await asyncio.sleep(delay + random.uniform(0, delay / 2))
                delay = min(delay * 2, 30.0)

    async def _once(self, request: LlmRequest) -> LlmResponse:
        calls: list[ToolCall] = []
        cwd = tempfile.mkdtemp(prefix="rfq-agent-llm-")
        start = time.perf_counter()

        async def prompt() -> AsyncIterator[dict[str, Any]]:
            yield {"type": "user", "message": {"role": "user", "content": request.prompt}}

        result: ResultMessage | None = None
        model = self.model
        try:
            async for message in query(prompt=prompt(), options=self.options(request, cwd, calls)):
                if isinstance(message, AssistantMessage):
                    model = message.model or model
                    if message.error in RETRY_ERRORS:
                        raise LlmError(f"API {message.error}", retryable=True)
                    if message.error:
                        raise LlmError(f"API {message.error}")
                elif isinstance(message, ResultMessage):
                    result = message
        finally:
            shutil.rmtree(cwd, ignore_errors=True)
        if result is None:
            raise LlmError("no result from the model", retryable=True)
        if result.is_error:
            status = result.api_error_status
            raise LlmError(
                f"{result.subtype}: {', '.join(result.errors or []) or result.result or 'error'}"[:300],
                retryable=status in RETRY_STATUS,
                status=status,
            )
        usage = result.usage or {}
        output = result.structured_output
        if output is None and result.result:
            try:
                output = json.loads(result.result)
            except json.JSONDecodeError:
                output = None
        return LlmResponse(
            output=output if isinstance(output, dict) else None,
            model=model,
            input_tokens=int(usage.get("input_tokens") or 0)
            + int(usage.get("cache_creation_input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
            cost_usd=float(result.total_cost_usd or 0.0),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            tool_calls=calls,
            source="live",
        )


async def live_available(timeout_s: float = 20.0) -> bool:
    """True when an API key is set or the CLI reports a signed-in login."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    try:
        env = child_env()
    except LlmError:
        return False
    try:
        proc = await asyncio.create_subprocess_exec(
            str(WRAPPER),
            "auth",
            "status",
            "--json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env={**env, "CLAUDE_AGENT_SDK_VERSION": claude_agent_sdk.__version__},
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout_s)
    except (OSError, TimeoutError):
        return False
    try:
        return bool(json.loads(out or b"{}").get("loggedIn"))
    except json.JSONDecodeError:
        return False
