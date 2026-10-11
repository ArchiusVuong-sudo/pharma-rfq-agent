"""The LangGraph run: shortlist -> read quotes -> compliance -> price -> plan -> gate -> follow-ups
-> act alone -> human review (interrupt) -> finish.

The model only reads emails and drafts follow-ups. Code decides compliance, money and routing.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from rfq_agent.agent import documents
from rfq_agent.agent.extract import extract_quote
from rfq_agent.agent.followup import draft_follow_up
from rfq_agent.agent.registry import shortlist
from rfq_agent.checks import CheckContext, apply_prices, check_email, merge_reply
from rfq_agent.gate import ROUTE_LABELS, decide_line
from rfq_agent.llm.base import LlmProvider
from rfq_agent.models import (
    Allocation,
    Buyer,
    CheckedQuote,
    ExtractedQuote,
    LineDecision,
    LinePlan,
    Product,
    QuoteEmail,
    Rfq,
    RfqLine,
    Severity,
    Supplier,
)
from rfq_agent.planner import plan_line, validate_plan
from rfq_agent.policy import Policy
from rfq_agent.store import Store
from rfq_agent.tracing import Tracer


class RunState(TypedDict, total=False):
    run_id: str
    rfq_id: str
    kind: str  # "award" or "recheck"
    email_ids: list[str]  # emails this run reads
    reply_to: dict[str, str]  # reply email id -> original email id
    line_ids: list[str]  # lines this run decides
    shortlist: dict[str, list[dict[str, Any]]]
    extractions: dict[str, dict[str, Any]]
    quotes: list[dict[str, Any]]
    plans: dict[str, dict[str, Any]]
    decisions: dict[str, dict[str, Any]]
    follow_ups: list[str]
    reviewed: dict[str, dict[str, Any]]


@dataclass
class Deps:
    store: Store
    provider: LlmProvider
    policy: Policy
    rfq: Rfq
    products: dict[str, Product]
    suppliers: dict[str, Supplier]
    buyers: dict[str, Buyer]
    emails: dict[str, QuoteEmail]  # quote emails and recorded replies
    as_of: Any

    def ctx(self) -> CheckContext:
        return CheckContext(
            as_of=self.as_of,
            policy=self.policy,
            rfq=self.rfq,
            products=self.products,
            suppliers=self.suppliers,
            buyers=self.buyers,
        )

    def line(self, line_id: str) -> RfqLine:
        return next(ln for ln in self.rfq.lines if ln.id == line_id)

    def names(self) -> dict[str, str]:
        return {s.id: s.name for s in self.suppliers.values()}


Node = Callable[[RunState], Awaitable[dict[str, Any]]]


def traced(deps: Deps, name: str, fn: Callable[[RunState, Tracer, Any], Awaitable[dict[str, Any]]]) -> Node:
    """Wrap a node in a trace span and a timeline step."""

    async def node(state: RunState) -> dict[str, Any]:
        run_id = state["run_id"]
        tracer = Tracer(deps.store, run_id)
        async with tracer.span("node", name) as span:
            try:
                return await fn(state, tracer, span)
            except GraphBubbleUp:
                span.status = "waiting"
                raise
            except Exception as exc:
                deps.store.event(run_id, name, f"{name} failed: {type(exc).__name__}: {exc}"[:300], "error")
                raise

    return node


def _quotes(state: RunState) -> list[CheckedQuote]:
    return [CheckedQuote.model_validate(q) for q in state.get("quotes", [])]


def build_graph(deps: Deps) -> StateGraph[RunState]:
    store = deps.store

    # ------------------------------------------------------------ 1. shortlist (sourcing)
    async def shortlist_node(state: RunState, tracer: Tracer, span: Any) -> dict[str, Any]:
        run_id = state["run_id"]
        out: dict[str, list[dict[str, Any]]] = {}
        for line_id in state["line_ids"]:
            line = deps.line(line_id)
            product = deps.products[line.product_id]
            start = tracer_now()
            try:
                entries, errors = await shortlist(line, list(deps.suppliers.values()), deps.buyers)
            except TimeoutError as exc:
                tracer.record("tool", "registry_lookup", span, start, tracer_now(), "error", error=str(exc))
                store.alert(run_id, "tool_error", f"Registry lookup failed for {product.name}: {exc}", "error")
                raise
            for err in errors:
                store.event(run_id, "shortlist", f"Registry lookup retried: {err}", "warn")
                store.alert(
                    run_id,
                    "tool_error",
                    f"Tool error on {product.name}: {err}; recovered on retry",
                    "warn",
                    dedupe_key=f"{run_id}-registry-{line_id}",
                )
            tracer.record(
                "tool",
                "registry_lookup",
                span,
                start,
                tracer_now(),
                "retry" if errors else "ok",
                line=line_id,
                attempts=len(errors) + 1,
                errors=errors,
                found=[e.supplier_id for e in entries],
            )
            out[line_id] = [e.model_dump() for e in entries]
            names = ", ".join(deps.suppliers[e.supplier_id].name for e in entries) or "none"
            store.event(run_id, "shortlist", f"{product.name}: {len(entries)} registry suppliers can serve ({names})")
        return {"shortlist": out}

    # ------------------------------------------------------------ 2. read quotes (LLM)
    async def read_node(state: RunState, tracer: Tracer, span: Any) -> dict[str, Any]:
        run_id, rfq_id = state["run_id"], state["rfq_id"]
        reply_to = state.get("reply_to", {})

        async def read_one(email_id: str) -> tuple[str, ExtractedQuote]:
            email = deps.emails[email_id]
            supplier = deps.suppliers.get(email.supplier_id)
            name = supplier.name if supplier else email.supplier_id
            store.event(run_id, "read_quotes", f"Reading {email_id} from {name}")
            result = await extract_quote(email, supplier, deps.provider, tracer, span)
            if result.attempts > 1:
                store.alert(
                    run_id,
                    "schema_retry",
                    f"{email_id}: model output failed the schema check, retried",
                    "warn",
                    dedupe_key=f"{run_id}-schema-{email_id}",
                )
            quote = result.quote
            tools = sum(len(r.tool_calls) for r in result.responses)
            if quote.items:
                store.event(
                    run_id,
                    "read_quotes",
                    f"{email_id}: read {len(quote.items)} item(s) with {tools} search_catalog call(s)",
                    data={"items": [i.product_text for i in quote.items]},
                )
            else:
                store.event(
                    run_id,
                    "read_quotes",
                    f"{email_id}: could not read a price, follow-up queued ({quote.unreadable_reason})",
                    "warn",
                )
            original = reply_to.get(email_id)
            if original:
                before = store.extraction(rfq_id, original)
                if before:
                    quote = merge_reply(ExtractedQuote.model_validate(before), quote)
                    store.event(run_id, "read_quotes", f"Merged the reply into the original quote {original}")
                return original, quote
            return email_id, quote

        results = await asyncio.gather(*(read_one(e) for e in state["email_ids"]))
        extractions = {}
        for email_id, quote in results:
            extractions[email_id] = quote.model_dump(mode="json")
            store.save_extraction(rfq_id, email_id, run_id, extractions[email_id])
        return {"extractions": extractions}

    # ------------------------------------------------------------ 3. compliance checks (code)
    async def compliance_node(state: RunState, tracer: Tracer, span: Any) -> dict[str, Any]:
        run_id, rfq_id = state["run_id"], state["rfq_id"]
        ctx = deps.ctx()
        fresh: list[CheckedQuote] = []
        for email_id, data in state["extractions"].items():
            email = deps.emails[email_id]
            checked = check_email(email, ExtractedQuote.model_validate(data), ctx)
            fresh.extend(checked)
            for q in checked:
                blocking = [f.label for f in q.flags if f.severity in (Severity.BLOCK, Severity.HOLD)]
                store.event(
                    run_id,
                    "compliance",
                    f"{deps.suppliers[q.supplier_id].name} / {q.item.product_text}: "
                    + (", ".join(blocking) if blocking else "passes registry, GMP, registration and document checks"),
                    "warn" if blocking else "info",
                )
        # Quotes from earlier runs on the same lines stay in play (a re-check reads only the reply).
        read = set(state["extractions"])
        kept = [
            CheckedQuote.model_validate(q)
            for q in store.quotes(rfq_id)
            if q["email_id"] not in read and q.get("line_id") in state["line_ids"]
        ]
        return {"quotes": [q.model_dump(mode="json") for q in kept + fresh]}

    # ------------------------------------------------------------ 4. price check (code)
    async def price_node(state: RunState, tracer: Tracer, span: Any) -> dict[str, Any]:
        run_id, rfq_id = state["run_id"], state["rfq_id"]
        priced = apply_prices(_quotes(state), deps.ctx())
        by_email: dict[str, list[dict[str, Any]]] = {}
        for q in priced:
            by_email.setdefault(q.email_id, []).append(q.model_dump(mode="json"))
            if q.landed_unit_usd is not None:
                outlier = [f.label for f in q.flags if f.check.startswith("price_far")]
                store.event(
                    run_id,
                    "price",
                    f"{deps.suppliers[q.supplier_id].name}: ${q.landed_unit_usd:.4f} landed per unit"
                    + (f" - {outlier[0]}" if outlier else ""),
                    "warn" if outlier else "info",
                )
        for email_id in state["extractions"]:
            store.save_quotes(rfq_id, run_id, email_id, by_email.get(email_id, []))
        return {"quotes": [q.model_dump(mode="json") for q in priced]}

    # ------------------------------------------------------------ 5. award plan (code)
    async def plan_node(state: RunState, tracer: Tracer, span: Any) -> dict[str, Any]:
        run_id = state["run_id"]
        quotes = _quotes(state)
        plans = {}
        for line_id in state["line_ids"]:
            line = deps.line(line_id)
            plan = plan_line(line, quotes, deps.suppliers, deps.buyers)
            plans[line_id] = plan.model_dump(mode="json")
            split = (
                "; ".join(
                    f"{deps.buyers[a.buyer_id].country} {a.quantity:,} from {deps.suppliers[a.supplier_id].name}"
                    for a in plan.allocations
                )
                or "no compliant supplier"
            )
            store.event(run_id, "plan", f"{deps.products[line.product_id].name}: {split} (${plan.total_usd:,.0f})")
        return {"plans": plans}

    # ------------------------------------------------------------ 6. autonomy gate (policy.yaml)
    async def gate_node(state: RunState, tracer: Tracer, span: Any) -> dict[str, Any]:
        run_id, rfq_id = state["run_id"], state["rfq_id"]
        quotes = _quotes(state)
        decisions = {}
        for line_id in state["line_ids"]:
            line = deps.line(line_id)
            plan = LinePlan.model_validate(state["plans"][line_id])
            decision = decide_line(
                line,
                plan,
                [q for q in quotes if q.line_id == line_id],
                deps.products[line.product_id],
                deps.policy,
                deps.names(),
            )
            decisions[line_id] = decision.model_dump(mode="json")
            status = {"act_alone": "acting", "review": "review", "stop": "blocked"}[decision.route]
            store.set_line(rfq_id, line_id, status, run_id, plan.model_dump(mode="json"), decisions[line_id])
            rules = ", ".join(h.rule for h in decision.rules_fired) or "no rule fired"
            store.event(
                run_id,
                "gate",
                f"{deps.products[line.product_id].name}: {ROUTE_LABELS[decision.route]} ({rules})",
                "warn" if decision.route != "act_alone" else "info",
                data={"line_id": line_id, "route": decision.route},
            )
        return {"decisions": decisions}

    # ------------------------------------------------------------ 7. supplier follow-ups (LLM + guard)
    async def follow_up_node(state: RunState, tracer: Tracer, span: Any) -> dict[str, Any]:
        run_id, rfq_id = state["run_id"], state["rfq_id"]
        sent: list[str] = list(state.get("follow_ups", []))
        holds: dict[str, list[CheckedQuote]] = {}
        for q in _quotes(state):
            if q.email_id in state["extractions"] and any(f.severity is Severity.HOLD for f in q.flags):
                if any(f.severity is Severity.BLOCK for f in q.flags):
                    continue  # no point asking a supplier we cannot buy from
                holds.setdefault(q.email_id, []).append(q)

        async def one(email_id: str, qs: list[CheckedQuote]) -> None:
            supplier = deps.suppliers[qs[0].supplier_id]
            checks = sorted({f.check for q in qs for f in q.flags if f.severity is Severity.HOLD})
            missing = list(dict.fromkeys(f.detail for q in qs for f in q.flags if f.severity is Severity.HOLD))
            product_id = qs[0].item.product_id
            product = deps.products[product_id].name if product_id else "your offer"
            result = await draft_follow_up(
                contact=supplier.contact_name,
                supplier=supplier.name,
                rfq_id=rfq_id,
                product=product,
                missing_checks=checks,
                missing_items=missing,
                provider=deps.provider,
                max_words=deps.policy.follow_up.max_words,
                tracer=tracer,
                parent=span,
            )
            msg_id = f"{run_id}-FU-{email_id}"
            store.add_outbox(
                {
                    "id": msg_id,
                    "run_id": run_id,
                    "rfq_id": rfq_id,
                    "line_id": qs[0].line_id,
                    "kind": "follow_up",
                    "supplier_id": supplier.id,
                    "subject": result.draft.subject,
                    "body": result.draft.body,
                    "drafted_by": result.drafted_by,
                    "guard_notes": result.notes,
                    "reply_email_id": deps_reply_for(supplier.id, email_id),
                    "source_email_id": email_id,
                }
            )
            sent.append(msg_id)
            if result.drafted_by == "template":
                store.alert(
                    run_id,
                    "follow_up_template",
                    f"Follow-up to {supplier.name} fell back to the template: {'; '.join(result.notes)}"[:300],
                    "warn",
                    dedupe_key=f"{msg_id}-template",
                )
            store.event(
                run_id,
                "follow_up",
                f"Follow-up to {supplier.name} asking for {len(missing)} item(s), "
                f"{'drafted by the model, guard passed' if result.drafted_by == 'model' else 'template'}"
                " - sent alone (simulated)",
            )

        def deps_reply_for(supplier_id: str, email_id: str) -> str | None:
            replies = {e.supplier_id: e for e in deps.emails.values() if e.id.startswith("R")}
            reply = replies.get(supplier_id)
            return reply.id if reply and state.get("kind") == "award" else None

        await asyncio.gather(*(one(e, qs) for e, qs in holds.items()))
        return {"follow_ups": sent}

    # ------------------------------------------------------------ 8. act alone
    def write_award(run_id: str, rfq_id: str, line_id: str, plan: LinePlan, quotes: list[CheckedQuote]) -> int:
        line = deps.line(line_id)
        product = deps.products[line.product_id]
        n = 0
        for sid, subject, body in documents.po_drafts(deps.rfq, line, plan, product, deps.suppliers, deps.buyers):
            store.add_outbox(
                {
                    "id": f"{run_id}-PO-{line_id}-{sid}",
                    "run_id": run_id,
                    "rfq_id": rfq_id,
                    "line_id": line_id,
                    "kind": "po_draft",
                    "supplier_id": sid,
                    "subject": subject,
                    "body": body,
                    "drafted_by": "code",
                }
            )
            store.add_outbox(
                {
                    "id": f"{run_id}-AW-{line_id}-{sid}",
                    "run_id": run_id,
                    "rfq_id": rfq_id,
                    "line_id": line_id,
                    "kind": "award_notice",
                    "supplier_id": sid,
                    "subject": f"Award: {product.name} on {rfq_id}",
                    "body": documents.award_notice(deps.rfq, line, plan, product, deps.suppliers[sid]),
                    "drafted_by": "code",
                }
            )
            n += 2
        awarded = {a.supplier_id for a in plan.allocations if a.quantity}
        regret = {q.supplier_id for q in quotes if q.line_id == line_id and q.usable} - awarded
        for sid in sorted(regret):
            store.add_outbox(
                {
                    "id": f"{run_id}-RG-{line_id}-{sid}",
                    "run_id": run_id,
                    "rfq_id": rfq_id,
                    "line_id": line_id,
                    "kind": "regret_notice",
                    "supplier_id": sid,
                    "subject": f"{rfq_id}: {product.name} outcome",
                    "body": documents.regret_notice(deps.rfq, product, deps.suppliers[sid]),
                    "drafted_by": "code",
                }
            )
            n += 1
        return n

    async def act_node(state: RunState, tracer: Tracer, span: Any) -> dict[str, Any]:
        run_id, rfq_id = state["run_id"], state["rfq_id"]
        quotes = _quotes(state)
        for line_id, d in state["decisions"].items():
            if d["route"] != "act_alone":
                continue
            plan = LinePlan.model_validate(state["plans"][line_id])
            n = write_award(run_id, rfq_id, line_id, plan, quotes)
            store.set_line(
                rfq_id, line_id, "awarded", run_id, plan.model_dump(mode="json"), d, note="Awarded by the agent alone"
            )
            store.event(
                run_id,
                "act",
                f"{deps.products[deps.line(line_id).product_id].name}: awarded alone, "
                f"{n} documents in the outbox (simulated sends)",
            )
        return {}

    # ------------------------------------------------------------ 9. human review (interrupt / resume)
    async def review_node(state: RunState, tracer: Tracer, span: Any) -> dict[str, Any]:
        run_id, rfq_id = state["run_id"], state["rfq_id"]
        quotes = _quotes(state)
        reviewed = dict(state.get("reviewed", {}))
        for line_id, d in state["decisions"].items():
            if d["route"] != "review" or line_id in reviewed:
                continue
            line = deps.line(line_id)
            plan = LinePlan.model_validate(state["plans"][line_id])
            line_quotes = [q for q in quotes if q.line_id == line_id]
            payload = {
                "plan": plan.model_dump(mode="json"),
                "decision": d,
                "quotes": [q.model_dump(mode="json") for q in line_quotes],
            }
            review_id = store.open_review(run_id, rfq_id, line_id, payload)
            errors: list[str] = []
            earlier_errors: list[str] = []
            while True:
                answer = interrupt({"review_id": review_id, "line_id": line_id, "errors": errors})
                action = answer.get("action")
                note = str(answer.get("note") or "")[:500]
                if action == "edit":
                    edited = LinePlan(
                        line_id=line_id,
                        allocations=[Allocation.model_validate(a) for a in answer.get("allocations", [])],
                    )
                    by_ref = {q.ref: q for q in line_quotes}
                    edited.allocations = [
                        a.model_copy(update={"landed_unit_usd": by_ref[a.quote_ref].landed_unit_usd or 0.0})
                        if a.quote_ref in by_ref
                        else a
                        for a in edited.allocations
                        if a.quantity
                    ]
                    covered: dict[str, int] = {}
                    for a in edited.allocations:
                        covered[a.buyer_id] = covered.get(a.buyer_id, 0) + a.quantity
                    edited.unmet = {
                        dm.buyer_id: dm.quantity - covered.get(dm.buyer_id, 0)
                        for dm in line.demands
                        if covered.get(dm.buyer_id, 0) < dm.quantity
                    }
                    errors = validate_plan(edited, line, line_quotes, deps.suppliers, deps.buyers)
                    if errors:
                        store.x(
                            "UPDATE reviews SET error = ? WHERE id = ?",
                            (
                                json.dumps({"errors": errors, "attempt": answer.get("allocations", []), "note": note}),
                                review_id,
                            ),
                        )
                        message = f"Edited split rejected by the re-check: {'; '.join(errors)}"
                        seen = store.q(
                            "SELECT COUNT(*) AS n FROM events WHERE run_id = ? AND node = 'review' AND message = ?",
                            (run_id, message),
                        )[0]["n"]
                        rejected_before = sum(1 for e in earlier_errors if e == message)
                        earlier_errors.append(message)
                        if seen <= rejected_before:  # a resumed node replays earlier answers: log each once
                            store.event(run_id, "review", message, "warn")
                        continue
                    plan = edited
                break
            resolved = {"action": action, "note": note, "plan": plan.model_dump(mode="json")}
            reviewed[line_id] = resolved
            status = {"approve": "approved", "edit": "changes", "reject": "rejected"}.get(str(action), "rejected")
            store.x(
                "UPDATE reviews SET status = ?, resolved_at = strftime('%Y-%m-%dT%H:%M:%fZ','now'), "
                "resolution = ?, error = NULL WHERE id = ?",
                (status, json.dumps(resolved), review_id),
            )
            product = deps.products[line.product_id].name
            if action in ("approve", "edit"):
                n = write_award(run_id, rfq_id, line_id, plan, quotes)
                store.set_line(
                    rfq_id,
                    line_id,
                    "awarded",
                    run_id,
                    plan.model_dump(mode="json"),
                    d,
                    note=f"Approved by a person{' with changes' if action == 'edit' else ''}"
                    + (f": {note}" if note else ""),
                )
                store.event(
                    run_id,
                    "review",
                    f"{product}: approved{' with changes' if action == 'edit' else ''}, "
                    f"plan re-checked, {n} documents in the outbox",
                )
            else:
                store.set_line(
                    rfq_id,
                    line_id,
                    "rejected",
                    run_id,
                    plan.model_dump(mode="json"),
                    d,
                    note="Rejected by a person" + (f": {note}" if note else ""),
                )
                store.event(run_id, "review", f"{product}: rejected by the reviewer, nothing sent", "warn")
        return {"reviewed": reviewed}

    # ------------------------------------------------------------ 10. finish
    async def finish_node(state: RunState, tracer: Tracer, span: Any) -> dict[str, Any]:
        store.event(state["run_id"], "finish", "Run finished")
        return {}

    graph: StateGraph[RunState] = StateGraph(RunState)
    steps: list[tuple[str, Callable[[RunState, Tracer, Any], Awaitable[dict[str, Any]]]]] = [
        ("shortlist", shortlist_node),
        ("read_quotes", read_node),
        ("compliance", compliance_node),
        ("price", price_node),
        ("plan", plan_node),
        ("gate", gate_node),
        ("follow_up", follow_up_node),
        ("act", act_node),
        ("review", review_node),
        ("finish", finish_node),
    ]
    previous = START
    for name, fn in steps:
        graph.add_node(name, traced(deps, name, fn))  # type: ignore[call-overload]
        graph.add_edge(previous, name)
        previous = name
    graph.add_edge(previous, END)
    return graph


STEP_LABELS = {
    "shortlist": "Shortlist suppliers",
    "read_quotes": "Read quotes",
    "compliance": "Compliance checks",
    "price": "Price check",
    "plan": "Award plan",
    "gate": "Autonomy gate",
    "follow_up": "Supplier follow-ups",
    "act": "Act alone",
    "review": "Human review",
    "finish": "Finish",
}


def tracer_now() -> float:
    import time

    return time.time() * 1000


def decisions_of(state: RunState) -> dict[str, LineDecision]:
    return {k: LineDecision.model_validate(v) for k, v in state.get("decisions", {}).items()}
