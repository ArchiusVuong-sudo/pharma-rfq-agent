from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rfq_agent.agent.extract import extract_quote, extract_request
from rfq_agent.llm import make_provider
from rfq_agent.llm.app_tools import call_tool
from rfq_agent.llm.base import FixtureMissing, LlmRequest, LlmResponse
from rfq_agent.llm.live import WRAPPER, LiveProvider, child_env, mcp_name
from rfq_agent.llm.replay import FIXTURE_DIR, RecordingProvider, ReplayProvider
from rfq_agent.models import ExtractedQuote
from rfq_agent.seed import demo_rfq, suppliers_by_id


def test_live_options_are_sandboxed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_SOMETHING", "x")
    monkeypatch.setenv("UNRELATED_SECRET", "x")
    request = extract_request(demo_rfq().emails[0], suppliers_by_id()["S-KES"])
    opts = LiveProvider(model="claude-haiku-5-5").options(request, "/tmp/x", [])
    assert opts.tools == []
    assert opts.allowed_tools == [mcp_name("search_catalog")]
    assert opts.setting_sources == []
    assert opts.strict_mcp_config is True
    assert opts.permission_mode == "dontAsk"
    assert opts.can_use_tool is not None
    assert "no-session-persistence" in opts.extra_args
    assert opts.max_turns == request.max_turns and opts.max_budget_usd
    assert opts.cli_path == str(WRAPPER) and opts.cwd == "/tmp/x"
    assert set(opts.env) <= {"PATH", "HOME", "USER", "LANG", "TMPDIR", "ANTHROPIC_API_KEY", "RFQ_AGENT_REAL_CLI"}
    assert opts.output_format == {"type": "json_schema", "schema": request.output_schema}


def test_child_env_is_an_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    env = child_env()
    assert not any(k == "CLAUDECODE" or k.startswith("CLAUDE_CODE_") for k in env)
    assert "PATH" in env and "RFQ_AGENT_REAL_CLI" in env


def test_wrapper_drops_everything_but_the_allowlist() -> None:
    text = WRAPPER.read_text()
    assert "env -i" in text and os.access(WRAPPER, os.X_OK)


async def test_tool_inputs_are_validated() -> None:
    ok = await call_tool("search_catalog", {"query": "zinc 20mg DT"})
    assert not ok.is_error and json.loads(ok.output)[0]["product_id"] == "P-ZINC-20-DT"
    bad = await call_tool("search_catalog", {"query": "z", "limit": 99})
    assert bad.is_error and bad.output.startswith("invalid input")
    unknown = await call_tool("read_file", {"path": "/etc/passwd"})
    assert unknown.is_error


def test_fixture_key_is_stable_and_input_sensitive() -> None:
    email = demo_rfq().emails[0]
    a = extract_request(email, suppliers_by_id()["S-KES"])
    b = extract_request(email, suppliers_by_id()["S-KES"])
    c = extract_request(email, suppliers_by_id()["S-KES"], feedback="retry")
    assert a.fixture_key() == b.fixture_key() != c.fixture_key()


def test_every_fixture_comes_from_a_recorded_live_run() -> None:
    files = sorted(FIXTURE_DIR.glob("*.json"))
    assert len(files) >= 21
    for path in files:
        data = json.loads(path.read_text())
        assert data["response"]["recorded_at"] and data["response"]["model"].startswith("claude-")
        assert data["response"]["source"] == "live"


async def test_replay_reruns_recorded_tool_calls() -> None:
    email = demo_rfq().emails[1]
    result = await extract_quote(email, suppliers_by_id()[email.supplier_id], ReplayProvider(delay_scale=0))
    response = result.responses[0]
    assert response.source == "replay" and response.tool_calls[0].name == "search_catalog"
    assert result.quote.items[0].product_id == "P-AMOX-250-DT"


async def test_replay_without_a_fixture_fails_loudly(tmp_path: Path) -> None:
    request = LlmRequest(task="extract_quote", system="s", prompt="p", output_schema={})
    with pytest.raises(FixtureMissing):
        await ReplayProvider(root=tmp_path, delay_scale=0).complete(request)


class Fixed:
    mode = "live"
    model = "fixed"

    def __init__(self, outputs: list[dict[str, object] | None]) -> None:
        self.outputs = outputs

    async def complete(self, request: LlmRequest) -> LlmResponse:
        return LlmResponse(output=self.outputs.pop(0), model="fixed")


async def test_recording_writes_a_fixture_replay_can_read(tmp_path: Path) -> None:
    request = LlmRequest(task="draft_follow_up", system="s", prompt="p", output_schema={})
    recorder = RecordingProvider(Fixed([{"subject": "a", "body": "b"}]), root=tmp_path)
    await recorder.complete(request)
    replayed = await ReplayProvider(root=tmp_path, delay_scale=0).complete(request)
    assert replayed.output == {"subject": "a", "body": "b"} and replayed.source == "replay"


async def test_schema_failure_retries_once_then_marks_unreadable() -> None:
    email = demo_rfq().emails[0]
    good = ExtractedQuote(supplier_name="K", items=[], unreadable_reason="none").model_dump()
    retried = await extract_quote(email, None, Fixed([{"items": "not a list"}, good]))
    assert retried.attempts == 2 and retried.quote.unreadable_reason == "none"
    failed = await extract_quote(email, None, Fixed([None, {"bad": 1}]))
    assert (
        failed.attempts == 2 and failed.quote.items == [] and "schema check" in (failed.quote.unreadable_reason or "")
    )


async def test_demo_replay_env_forces_replay() -> None:
    assert (await make_provider()).mode == "replay"
