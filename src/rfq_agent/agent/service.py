"""Starts, resumes and monitors agent runs. One LangGraph thread per run, checkpointed in SQLite."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from rfq_agent.agent.graph import Deps, RunState, build_graph
from rfq_agent.llm.base import LlmProvider
from rfq_agent.policy import Policy, load_policy
from rfq_agent.seed import AS_OF, buyers_by_id, catalog_by_id, demo_rfq, recorded_replies, suppliers_by_id
from rfq_agent.store import Store, new_id, now_iso


class RunBusy(RuntimeError):
    pass


class ReviewError(RuntimeError):
    pass


def make_deps(store: Store, provider: LlmProvider, policy: Policy | None = None) -> Deps:
    rfq = demo_rfq()
    emails = {e.id: e for e in rfq.emails}
    emails.update({e.id: e for e in recorded_replies().values()})
    return Deps(
        store=store,
        provider=provider,
        policy=policy or load_policy(),
        rfq=rfq,
        products=catalog_by_id(),
        suppliers=suppliers_by_id(),
        buyers=buyers_by_id(),
        emails=emails,
        as_of=AS_OF,
    )


class AgentService:
    def __init__(
        self, store: Store, provider: LlmProvider, checkpointer: BaseCheckpointSaver[Any], policy: Policy | None = None
    ) -> None:
        self.store = store
        self.provider = provider
        self.deps = make_deps(store, provider, policy)
        self.graph = build_graph(self.deps).compile(checkpointer=checkpointer)
        self.tasks: dict[str, asyncio.Task[None]] = {}

    @property
    def rfq_id(self) -> str:
        return self.deps.rfq.id

    def _busy(self) -> bool:
        return self.store.one("SELECT id FROM runs WHERE rfq_id = ? AND status = 'running'", (self.rfq_id,)) is not None

    def _start(self, state: RunState, kind: str, parent: str | None = None) -> str:
        if self._busy():
            raise RunBusy("A run is already in progress for this RFQ")
        run_id = state["run_id"]
        self.store.create_run(run_id, self.rfq_id, kind, self.provider.mode, self.provider.model, parent)
        self.store.event(run_id, "start", f"Run started ({kind}, LLM {self.provider.mode})")
        self.tasks[run_id] = asyncio.create_task(self._drive(run_id, state))
        return run_id

    def start_award_run(self) -> str:
        rfq = self.deps.rfq
        with self.store.tx() as c:
            if self._busy():
                raise RunBusy("A run is already in progress for this RFQ")
            c.execute("UPDATE reviews SET status = 'superseded' WHERE rfq_id = ? AND status = 'pending'", (rfq.id,))
            c.execute("UPDATE runs SET status = 'superseded' WHERE rfq_id = ? AND status = 'waiting_review'", (rfq.id,))
            c.execute("DELETE FROM line_state WHERE rfq_id = ?", (rfq.id,))
            c.execute("DELETE FROM quotes WHERE rfq_id = ?", (rfq.id,))
            c.execute("DELETE FROM extractions WHERE rfq_id = ?", (rfq.id,))
        state: RunState = {
            "run_id": new_id("RUN"),
            "rfq_id": rfq.id,
            "kind": "award",
            "email_ids": [e.id for e in rfq.emails],
            "reply_to": {},
            "line_ids": [ln.id for ln in rfq.lines],
        }
        return self._start(state, "award")

    def simulate_reply(self, outbox_id: str) -> str:
        msg = self.store.one("SELECT * FROM outbox WHERE id = ?", (outbox_id,))
        if not msg or not msg["reply_email_id"]:
            raise ReviewError("No recorded reply for this message")
        if msg["reply_simulated"]:
            raise ReviewError("This reply was already simulated")
        if not msg["line_id"]:
            raise ReviewError("The follow-up is not tied to an RFQ line")
        state: RunState = {
            "run_id": new_id("RUN"),
            "rfq_id": self.rfq_id,
            "kind": "recheck",
            "email_ids": [msg["reply_email_id"]],
            "reply_to": {msg["reply_email_id"]: msg["source_email_id"]},
            "line_ids": [msg["line_id"]],
        }
        run_id = self._start(state, "recheck", parent=msg["run_id"])
        self.store.x("UPDATE outbox SET reply_simulated = 1 WHERE id = ?", (outbox_id,))
        return run_id

    async def resume_review(self, review_id: str, answer: dict[str, Any]) -> dict[str, Any]:
        review = self.store.review(review_id)
        if review is None:
            raise ReviewError("Review not found")
        if review["status"] != "pending":
            raise ReviewError(f"Review is already {review['status']}")
        run = self.store.one("SELECT * FROM runs WHERE id = ?", (review["run_id"],))
        if not run or run["status"] != "waiting_review":
            raise ReviewError("The run is not waiting for a review")
        self.store.update_run(run["id"], status="running")
        self.store.event(run["id"], "review", f"Reviewer answered: {answer.get('action')}")
        await self._drive(run["id"], Command(resume=answer))
        updated = self.store.review(review_id)
        assert updated is not None
        return updated

    async def wait(self, run_id: str) -> None:
        task = self.tasks.get(run_id)
        if task:
            await task

    async def _drive(self, run_id: str, payload: Any) -> None:
        config: RunnableConfig = {"configurable": {"thread_id": run_id}}
        try:
            await self.graph.ainvoke(payload, config)
            snap = await self.graph.aget_state(config)
            if snap.next:
                self.store.update_run(run_id, status="waiting_review", active_ms=self._active_ms(run_id))
                self.store.event(run_id, "review", "Paused for human review (checkpoint saved)", "warn")
            else:
                self._finish(run_id)
        except Exception as exc:
            self.store.update_run(
                run_id,
                status="failed",
                ended_at=now_iso(),
                error=f"{type(exc).__name__}: {exc}"[:400],
                active_ms=self._active_ms(run_id),
            )
            self.store.alert(run_id, "run_failed", f"Run failed: {type(exc).__name__}: {exc}"[:300], "error")
            self.store.event(run_id, "finish", f"Run failed: {type(exc).__name__}", "error")

    def _active_ms(self, run_id: str) -> float:
        row = self.store.one(
            "SELECT COALESCE(SUM(end_ms - start_ms), 0) AS ms FROM spans "
            "WHERE run_id = ? AND kind = 'node' AND end_ms IS NOT NULL",
            (run_id,),
        )
        return float(row["ms"]) if row else 0.0

    def _finish(self, run_id: str) -> None:
        active = self._active_ms(run_id)
        lines = self.store.lines(self.rfq_id)
        summary = {k: v["status"] for k, v in lines.items()}
        self.store.update_run(run_id, status="done", ended_at=now_iso(), active_ms=active, summary=json.dumps(summary))
        run = self.store.one("SELECT * FROM runs WHERE id = ?", (run_id,))
        mon = self.deps.policy.monitoring
        if run and run["cost_usd"] > mon.run_cost_budget_usd:
            self.store.alert(
                run_id,
                "cost_budget",
                f"Run cost ${run['cost_usd']:.3f} is over the ${mon.run_cost_budget_usd:.2f} budget (API-equivalent)",
                "warn",
                f"{run_id}-cost",
            )
        if active / 1000 > mon.run_latency_budget_s:
            self.store.alert(
                run_id,
                "latency_budget",
                f"Run took {active / 1000:.0f}s, over the {mon.run_latency_budget_s:.0f}s budget",
                "warn",
                f"{run_id}-latency",
            )

    def sweep_alerts(self) -> None:
        """Alert on reviews pending past the SLA."""
        sla = self.deps.policy.monitoring.review_sla_minutes
        cutoff = (datetime.now(UTC) - timedelta(minutes=sla)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        for r in self.store.q("SELECT * FROM reviews WHERE status = 'pending' AND created_at < ?", (cutoff,)):
            self.store.alert(
                r["run_id"],
                "review_sla",
                f"Review {r['id']} for line {r['line_id']} is pending past the {sla} min SLA",
                "warn",
                f"{r['id']}-sla",
            )
