"""FastAPI + Jinja2 + HTMX ops console.

uv run uvicorn rfq_agent.web.app:app --host 127.0.0.1 --port 4103
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from rfq_agent.agent.service import AgentService, ReviewError, RunBusy
from rfq_agent.evals.runner import EvalReport, run_evals
from rfq_agent.gate import ROUTE_LABELS
from rfq_agent.llm import make_provider
from rfq_agent.llm.replay import ReplayProvider
from rfq_agent.models import Allocation, CheckedQuote, LinePlan
from rfq_agent.policy import load_policy, policy_text
from rfq_agent.seed import AS_OF
from rfq_agent.store import Store, data_dir, new_id, now_iso
from rfq_agent.web import views

HERE = Path(__file__).parent
templates = Jinja2Templates(directory=str(HERE / "templates"))


def _money(v: float | None, digits: int = 0) -> str:
    return "-" if v is None else f"${v:,.{digits}f}"


def _ms(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.0f} ms" if v < 1000 else f"{v / 1000:.1f} s"


def _when(iso: str | None) -> str:
    if not iso:
        return "-"
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%d %b %H:%M:%S")
    except ValueError:
        return iso


templates.env.filters.update(
    money=_money,
    ms=_ms,
    when=_when,
    num=lambda v: "-" if v is None else f"{v:,}",
    pct=lambda v: f"{v:.0%}",
    flagclass=views.flag_class,
    route_label=lambda r: ROUTE_LABELS.get(r, r),
    fromjson=lambda s: json.loads(s) if s else None,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with AsyncExitStack() as stack:
        store = Store()
        stack.callback(store.close)
        saver = await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(str(data_dir() / "checkpoints.db")))
        provider = await make_provider(replay_delay_scale=float(os.environ.get("REPLAY_DELAY_SCALE", "0.35")))
        # A run cut off by a restart cannot continue mid-node; mark it so the page does not spin forever.
        store.x("UPDATE runs SET status = 'failed', error = 'server restarted mid-run' WHERE status = 'running'")
        app.state.store = store
        app.state.service = AgentService(store, provider, saver)
        app.state.provider = provider
        yield


app = FastAPI(title="Pharma RFQ Agent", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")


def svc(request: Request) -> AgentService:
    service: AgentService = request.app.state.service
    return service


def page(request: Request, name: str, ctx: dict[str, Any], status_code: int = 200) -> HTMLResponse:
    service = svc(request)
    store = service.store
    base = {
        "llm_mode": service.provider.mode,
        "llm_model": service.provider.model,
        "pending_reviews": len(store.q("SELECT id FROM reviews WHERE status = 'pending'")),
        "nav": ctx.pop("nav", ""),
        "as_of": AS_OF,
        "flash": request.query_params.get("msg"),
        "flash_error": request.query_params.get("err"),
    }
    return templates.TemplateResponse(request, name, {**base, **ctx}, status_code=status_code)


# ---------------------------------------------------------------- RFQs


@app.get("/", response_class=HTMLResponse)
async def desk(request: Request) -> HTMLResponse:
    service = svc(request)
    data = views.rfq_page(service.store, service.deps)
    return page(request, "desk.html", {"nav": "rfqs", **data, "deps": service.deps})


@app.get("/rfq/{rfq_id}", response_class=HTMLResponse)
async def rfq_detail(request: Request, rfq_id: str) -> HTMLResponse:
    service = svc(request)
    if rfq_id != service.rfq_id:
        raise HTTPException(404, "RFQ not found")
    data = views.rfq_page(service.store, service.deps)
    return page(request, "rfq.html", {"nav": "rfqs", **data, "deps": service.deps})


@app.get("/rfq/{rfq_id}/live", response_class=HTMLResponse)
async def rfq_live(request: Request, rfq_id: str) -> HTMLResponse:
    service = svc(request)
    data = views.rfq_page(service.store, service.deps)
    if request.query_params.get("poll") and not data["running"]:
        # The run just ended: reload the whole page so the header, nav counts and buttons update too.
        return HTMLResponse("", headers={"HX-Refresh": "true"})
    return page(request, "_live.html", {**data, "deps": service.deps})


@app.post("/rfq/{rfq_id}/run")
async def rfq_run(request: Request, rfq_id: str) -> Response:
    service = svc(request)
    try:
        service.start_award_run()
    except RunBusy as exc:
        return RedirectResponse(f"/rfq/{rfq_id}?err={exc}", 303)
    return RedirectResponse(f"/rfq/{rfq_id}", 303)


@app.post("/reset")
async def reset(request: Request) -> Response:
    service = svc(request)
    if service.store.one("SELECT id FROM runs WHERE status = 'running'"):
        return RedirectResponse("/?err=Wait for the running run to finish", 303)
    service.store.reset()
    return RedirectResponse("/?msg=Demo data reset", 303)


# ---------------------------------------------------------------- reviews


@app.get("/reviews", response_class=HTMLResponse)
async def reviews(request: Request) -> HTMLResponse:
    service = svc(request)
    deps = service.deps
    cards = []
    for r in service.store.reviews():
        if r["status"] == "superseded":
            continue
        plan = LinePlan.model_validate(r["payload"]["plan"])
        quotes = [CheckedQuote.model_validate(q) for q in r["payload"]["quotes"]]
        line = deps.line(r["line_id"])
        last_error = json.loads(r["error"]) if r["error"] else {}
        attempt = last_error.get("attempt")
        shown = [Allocation.model_validate(a) for a in attempt] if attempt is not None else plan.allocations
        current = {(a.buyer_id, a.quote_ref): a.quantity for a in shown}
        options = []
        for d in line.demands:
            buyer = deps.buyers[d.buyer_id]
            for q in quotes:
                supplier = deps.suppliers[q.supplier_id]
                if not q.usable or buyer.country not in supplier.registrations.get(line.product_id, []):
                    continue
                options.append(
                    {
                        "buyer": buyer,
                        "demand": d.quantity,
                        "quote": q,
                        "supplier": supplier,
                        "qty": current.get((buyer.id, q.ref), 0),
                    }
                )
        resolved_plan = (
            LinePlan.model_validate(r["resolution"]["plan"])
            if r["resolution"] and r["resolution"].get("plan")
            else None
        )
        cards.append(
            {
                "review": r,
                "line": line,
                "product": deps.products[line.product_id],
                "plan": plan,
                "decision": r["payload"]["decision"],
                "options": options,
                "quotes": quotes,
                "errors": last_error.get("errors", []),
                "note": last_error.get("note", ""),
                "resolved_plan": resolved_plan,
            }
        )
    service.sweep_alerts()
    return page(
        request,
        "reviews.html",
        {"nav": "reviews", "cards": cards, "deps": deps, "sla": deps.policy.monitoring.review_sla_minutes},
    )


@app.post("/reviews/{review_id}")
async def decide(request: Request, review_id: str) -> Response:
    service = svc(request)
    form = await request.form()
    action = str(form.get("action", ""))
    note = str(form.get("note", ""))
    answer: dict[str, Any] = {"action": action, "note": note}
    if action not in ("approve", "edit", "reject"):
        return RedirectResponse(f"/reviews?err=Unknown action#{review_id}", 303)
    if action == "edit":
        allocations = []
        for key, value in form.multi_items():
            if not key.startswith("qty:"):
                continue
            _, buyer_id, supplier_id, ref = key.split(":", 3)
            try:
                qty = int(str(value).replace(",", "").strip() or 0)
            except ValueError:
                return RedirectResponse(f"/reviews?err=Quantities must be whole numbers#{review_id}", 303)
            if qty < 0:
                return RedirectResponse(f"/reviews?err=Quantities cannot be negative#{review_id}", 303)
            if qty:
                allocations.append(
                    Allocation(buyer_id=buyer_id, supplier_id=supplier_id, quote_ref=ref, quantity=qty).model_dump()
                )
        answer["allocations"] = allocations
    try:
        result = await service.resume_review(review_id, answer)
    except ReviewError as exc:
        return RedirectResponse(f"/reviews?err={exc}#{review_id}", 303)
    if result["status"] == "pending":
        return RedirectResponse(f"/reviews#{review_id}", 303)  # the card shows why the edit failed
    return RedirectResponse(f"/rfq/{result['rfq_id']}?msg=Review saved, the run resumed from its checkpoint", 303)


# ---------------------------------------------------------------- outbox


@app.get("/outbox", response_class=HTMLResponse)
async def outbox(request: Request) -> HTMLResponse:
    service = svc(request)
    return page(request, "outbox.html", {"nav": "outbox", "messages": service.store.outbox(), "deps": service.deps})


@app.post("/outbox/{message_id}/reply")
async def simulate_reply(request: Request, message_id: str) -> Response:
    service = svc(request)
    try:
        service.simulate_reply(message_id)
    except (ReviewError, RunBusy) as exc:
        return RedirectResponse(f"/outbox?err={exc}", 303)
    return RedirectResponse(f"/rfq/{service.rfq_id}?msg=Supplier reply received, re-check running", 303)


# ---------------------------------------------------------------- runs


@app.get("/runs", response_class=HTMLResponse)
async def runs(request: Request) -> HTMLResponse:
    service = svc(request)
    service.sweep_alerts()
    return page(
        request,
        "runs.html",
        {"nav": "runs", **views.runs_page(service.store), "monitoring": service.deps.policy.monitoring},
    )


@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_detail(request: Request, run_id: str) -> HTMLResponse:
    service = svc(request)
    run = service.store.one("SELECT * FROM runs WHERE id = ?", (run_id,))
    if run is None:
        raise HTTPException(404, "Run not found")
    alerts = service.store.q("SELECT * FROM alerts WHERE run_id = ? ORDER BY ts", (run_id,))
    return page(
        request,
        "run.html",
        {
            "nav": "runs",
            "run": run,
            "segments": views.waterfall(service.store, run_id),
            "alerts": alerts,
            "steps": views.timeline(service.store, run),
        },
    )


# ---------------------------------------------------------------- evals and policy


@app.get("/evals", response_class=HTMLResponse)
async def evals(request: Request) -> HTMLResponse:
    service = svc(request)
    rows = service.store.q("SELECT * FROM eval_runs ORDER BY ts DESC LIMIT 10")
    latest = EvalReport.model_validate_json(rows[0]["metrics"]) if rows else None
    return page(
        request,
        "evals.html",
        {"nav": "evals", "latest": latest, "history": rows, "latest_row": rows[0] if rows else None},
    )


@app.post("/evals/run")
async def evals_run(request: Request) -> Response:
    service = svc(request)
    form = await request.form()
    guard_on = "on" in form.getlist("price_outlier")  # the hidden "off" field is sent either way
    policy = load_policy() if guard_on else load_policy().without("price_outlier")
    report = await run_evals(ReplayProvider(delay_scale=0), policy)
    service.store.x(
        "INSERT INTO eval_runs (id, ts, mode, disabled_rules, metrics, cases, passed) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            new_id("EVAL"),
            now_iso(),
            report.mode,
            json.dumps(report.disabled_rules),
            report.model_dump_json(),
            str(len(report.cases)),
            int(report.passed),
        ),
    )
    return RedirectResponse("/evals", 303)


@app.get("/policy", response_class=HTMLResponse)
async def policy(request: Request) -> HTMLResponse:
    return page(request, "policy.html", {"nav": "policy", "policy": load_policy(), "raw": policy_text()})


@app.get("/healthz")
async def healthz(request: Request) -> dict[str, str]:
    return {"status": "ok", "llm": svc(request).provider.mode}
