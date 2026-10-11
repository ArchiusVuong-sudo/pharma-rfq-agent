"""The LangGraph run end to end on recorded model outputs: gate, review, resume, restart, re-check."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from rfq_agent.agent.service import AgentService, ReviewError, RunBusy
from rfq_agent.llm.replay import ReplayProvider
from rfq_agent.store import Store


@asynccontextmanager
async def service(tmp_path: Path) -> AsyncIterator[AgentService]:
    store = Store(tmp_path / "app.db")
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as saver:
        yield AgentService(store, ReplayProvider(delay_scale=0), saver)
    store.close()


async def award(svc: AgentService) -> str:
    run_id = svc.start_award_run()
    await svc.wait(run_id)
    return run_id


def statuses(svc: AgentService) -> dict[str, str]:
    return {k: v["status"] for k, v in svc.store.lines(svc.rfq_id).items()}


def kept_split(review: dict[str, Any], brw: int, kes_ke: int) -> list[dict[str, Any]]:
    allocations = [dict(a) for a in review["payload"]["plan"]["allocations"]]
    for a in allocations:
        if a["supplier_id"] == "S-BRW":
            a["quantity"] = brw
    allocations.append({"buyer_id": "B-KE-01", "supplier_id": "S-KES", "quote_ref": "E1#1", "quantity": kes_ke})
    return allocations


async def test_award_run_routes_each_line(tmp_path: Path) -> None:
    async with service(tmp_path) as svc:
        run_id = await award(svc)
        run = svc.store.one("SELECT * FROM runs WHERE id = ?", (run_id,))
        assert run and run["status"] == "waiting_review" and run["llm_calls"] == 8
        assert statuses(svc) == {"L1": "awarded", "L2": "review", "L3": "blocked"}
        rules = {r["rule"] for r in svc.store.lines(svc.rfq_id)["L2"]["decision"]["rules_fired"]}
        assert rules == {"value_limit", "new_supplier", "gmp_expiring"}
        kinds = sorted(m["kind"] for m in svc.store.outbox())
        assert kinds == ["award_notice", "follow_up", "follow_up", "po_draft"]
        unreadable = next(q for q in svc.store.quotes(svc.rfq_id) if q["email_id"] == "E6")
        assert unreadable["unit_price_usd"] is None  # flagged, never guessed
        spans = svc.store.q("SELECT kind, COUNT(*) AS n FROM spans WHERE run_id = ? GROUP BY kind", (run_id,))
        assert {s["kind"] for s in spans} == {"node", "llm", "tool"}


async def test_bad_edit_comes_back_then_good_edit_awards(tmp_path: Path) -> None:
    async with service(tmp_path) as svc:
        await award(svc)
        [review] = svc.store.reviews("pending")
        result = await svc.resume_review(
            review["id"], {"action": "edit", "allocations": kept_split(review, 50_000, 130_000)}
        )
        assert result["status"] == "pending" and "under its MOQ of 100,000" in result["error"]
        result = await svc.resume_review(
            review["id"], {"action": "edit", "note": "split", "allocations": kept_split(review, 100_000, 80_000)}
        )
        assert result["status"] == "changes"
        assert statuses(svc)["L2"] == "awarded"
        plan = svc.store.lines(svc.rfq_id)["L2"]["plan"]
        assert {(a["supplier_id"], a["quantity"]) for a in plan["allocations"] if a["buyer_id"] == "B-KE-01"} == {
            ("S-BRW", 100_000),
            ("S-KES", 80_000),
        }
        kinds = [m["kind"] for m in svc.store.outbox() if m["line_id"] == "L2"]
        assert kinds.count("po_draft") == 2 and kinds.count("award_notice") == 2
        with pytest.raises(ReviewError):
            await svc.resume_review(review["id"], {"action": "approve"})


async def test_reject_sends_nothing(tmp_path: Path) -> None:
    async with service(tmp_path) as svc:
        await award(svc)
        [review] = svc.store.reviews("pending")
        await svc.resume_review(review["id"], {"action": "reject", "note": "price history unclear"})
        assert statuses(svc)["L2"] == "rejected"
        assert not [m for m in svc.store.outbox() if m["line_id"] == "L2"]


async def test_resume_after_server_restart(tmp_path: Path) -> None:
    async with service(tmp_path) as svc:
        await award(svc)
        [review] = svc.store.reviews("pending")
    # Everything in memory is gone; a new service on the same files resumes the paused run.
    async with service(tmp_path) as fresh:
        result = await fresh.resume_review(review["id"], {"action": "approve", "note": "after restart"})
        assert result["status"] == "approved"
        assert statuses(fresh)["L2"] == "awarded"
        run = fresh.store.one("SELECT status FROM runs WHERE id = ?", (review["run_id"],))
        assert run and run["status"] == "done"


async def test_supplier_reply_rechecks_and_opens_a_cold_chain_review(tmp_path: Path) -> None:
    async with service(tmp_path) as svc:
        await award(svc)
        follow_up = next(m for m in svc.store.outbox() if m["reply_email_id"])
        assert follow_up["supplier_id"] == "S-HAL"
        run_id = svc.simulate_reply(follow_up["id"])
        await svc.wait(run_id)
        assert statuses(svc)["L3"] == "review"
        rules = [r["rule"] for r in svc.store.lines(svc.rfq_id)["L3"]["decision"]["rules_fired"]]
        assert rules == ["cold_chain"]
        halden = next(q for q in svc.store.quotes(svc.rfq_id) if q["supplier_id"] == "S-HAL")
        assert not [f for f in halden["flags"] if f["severity"] in ("hold", "block")]
        with pytest.raises(ReviewError):
            svc.simulate_reply(follow_up["id"])


async def test_one_run_at_a_time(tmp_path: Path) -> None:
    async with service(tmp_path) as svc:
        run_id = svc.start_award_run()
        with pytest.raises(RunBusy):
            svc.start_award_run()
        await svc.wait(run_id)


async def test_tool_fault_is_retried_and_alerted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAULT_ONCE", "registry_timeout")
    async with service(tmp_path) as svc:
        run_id = await award(svc)
        alerts = svc.store.q("SELECT kind, message FROM alerts WHERE run_id = ?", (run_id,))
        assert [a["kind"] for a in alerts] == ["tool_error"] and "recovered on retry" in alerts[0]["message"]
        retry = svc.store.q("SELECT status FROM spans WHERE run_id = ? AND name = 'registry_lookup'", (run_id,))
        assert "retry" in [s["status"] for s in retry]


async def test_review_past_sla_raises_an_alert(tmp_path: Path) -> None:
    async with service(tmp_path) as svc:
        await award(svc)
        svc.store.x("UPDATE reviews SET created_at = '2026-01-01T00:00:00.000000Z'")
        svc.sweep_alerts()
        svc.sweep_alerts()
        assert [a["kind"] for a in svc.store.q("SELECT kind FROM alerts")] == ["review_sla"]
