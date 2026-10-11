"""View models for the pages: plain dicts and models assembled from the store and the seed."""

from __future__ import annotations

import json
import statistics
from typing import Any

from rfq_agent.agent.graph import STEP_LABELS, Deps
from rfq_agent.agent.registry import eligible
from rfq_agent.models import CheckedQuote, LinePlan, Severity
from rfq_agent.store import Store

STEP_ORDER = list(STEP_LABELS)


def latest_run(store: Store, rfq_id: str) -> dict[str, Any] | None:
    return store.one("SELECT * FROM runs WHERE rfq_id = ? ORDER BY started_at DESC LIMIT 1", (rfq_id,))


def timeline(store: Store, run: dict[str, Any] | None) -> list[dict[str, Any]]:
    """One row per graph step: status from its spans, messages from the run's events."""
    if run is None:
        return []
    spans = store.q(
        "SELECT name, status, start_ms, end_ms FROM spans WHERE run_id = ? AND kind = 'node' ORDER BY start_ms",
        (run["id"],),
    )
    events = store.q("SELECT node, level, message, ts FROM events WHERE run_id = ? ORDER BY id", (run["id"],))
    steps = []
    for name in STEP_ORDER:
        mine = [s for s in spans if s["name"] == name]
        if not mine:
            status = "pending"
        elif any(s["status"] == "running" for s in mine):
            status = "running"
        elif mine[-1]["status"] == "waiting":
            status = "waiting"
        elif any(s["status"] == "error" for s in mine):
            status = "error"
        else:
            status = "done"
        ms = sum((s["end_ms"] or s["start_ms"]) - s["start_ms"] for s in mine)
        msgs = [e for e in events if e["node"] == name]
        steps.append({"name": name, "label": STEP_LABELS[name], "status": status, "ms": ms, "events": msgs})
    return steps


def flag_class(severity: str) -> str:
    return {
        Severity.BLOCK.value: "red",
        Severity.HOLD.value: "red",
        Severity.REVIEW.value: "amber",
        Severity.INFO.value: "gray",
    }.get(severity, "gray")


def rfq_page(store: Store, deps: Deps) -> dict[str, Any]:
    rfq = deps.rfq
    lines_state = store.lines(rfq.id)
    quotes = [CheckedQuote.model_validate(q) for q in store.quotes(rfq.id)]
    run = latest_run(store, rfq.id)
    lines = []
    for ln in rfq.lines:
        product = deps.products[ln.product_id]
        state = lines_state.get(ln.id)
        plan = LinePlan.model_validate(state["plan"]) if state and state["plan"] else None
        lines.append(
            {
                "line": ln,
                "product": product,
                "state": state,
                "plan": plan,
                "shortlist": eligible(ln, list(deps.suppliers.values()), deps.buyers),
                "quotes": [q for q in quotes if q.line_id == ln.id],
            }
        )
    unmatched = [q for q in quotes if q.line_id is None]
    return {
        "rfq": rfq,
        "lines": lines,
        "unmatched": unmatched,
        "run": run,
        "steps": timeline(store, run),
        "running": bool(run and run["status"] == "running"),
        "has_quotes": bool(quotes),
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(pct) - 1]


def runs_page(store: Store) -> dict[str, Any]:
    runs = store.q("SELECT * FROM runs ORDER BY started_at DESC")
    finished = [r for r in runs if r["status"] in ("done", "waiting_review")]
    decisions: list[str] = []
    for r in runs:
        if r["status"] == "running":
            continue
        for e in store.q("SELECT data FROM events WHERE run_id = ? AND node = 'gate' AND data IS NOT NULL", (r["id"],)):
            decisions.append(json.loads(e["data"])["route"])
    total = len(decisions) or 1
    latencies = [r["active_ms"] / 1000 for r in finished if r["active_ms"]]
    award_runs = [r for r in finished if r["kind"] == "award"]
    alerts = store.q("SELECT * FROM alerts ORDER BY ts DESC LIMIT 50")
    return {
        "runs": runs,
        "metrics": {
            "runs": len(runs),
            "line_decisions": len(decisions),
            "act_alone": decisions.count("act_alone") / total,
            "review": decisions.count("review") / total,
            "stop": decisions.count("stop") / total,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "cost_per_rfq": statistics.mean([r["cost_usd"] for r in award_runs]) if award_runs else 0.0,
            "llm_calls": sum(r["llm_calls"] for r in runs),
            "tokens": sum(r["input_tokens"] + r["output_tokens"] for r in runs),
        },
        "alerts": alerts,
    }


def waterfall(store: Store, run_id: str) -> list[dict[str, Any]]:
    """Spans grouped into segments; a gap over 5 s (a review wait) starts a new segment."""
    spans = store.q("SELECT * FROM spans WHERE run_id = ? ORDER BY start_ms", (run_id,))
    for s in spans:
        s["attrs"] = json.loads(s["attrs"] or "{}")
        s["end_ms"] = s["end_ms"] or s["start_ms"]
    roots = [s for s in spans if s["parent_id"] is None]
    children: dict[str, list[dict[str, Any]]] = {}
    for s in spans:
        if s["parent_id"]:
            children.setdefault(s["parent_id"], []).append(s)
    segments: list[list[dict[str, Any]]] = []
    last_end = None
    for r in roots:
        if last_end is None or r["start_ms"] - last_end > 5000:
            segments.append([])
        segments[-1].append(r)
        last_end = max(last_end or 0, r["end_ms"])
    out = []
    prev_end = None
    for seg in segments:
        rows: list[dict[str, Any]] = []
        stack = [(r, 0) for r in reversed(seg)]
        while stack:
            span, depth = stack.pop()
            rows.append({**span, "depth": depth})
            kids = sorted(children.get(span["id"], []), key=lambda k: float(k["start_ms"]))
            stack.extend((k, depth + 1) for k in reversed(kids))
        start = min(x["start_ms"] for x in rows)
        end = max(x["end_ms"] for x in rows)
        width = max(end - start, 1.0)
        for x in rows:
            x["left"] = (x["start_ms"] - start) / width * 100
            x["width"] = max((x["end_ms"] - x["start_ms"]) / width * 100, 0.4)
            x["ms"] = x["end_ms"] - x["start_ms"]
        out.append({"rows": rows, "ms": end - start, "gap_ms": (start - prev_end) if prev_end else 0})
        prev_end = end
    return out
