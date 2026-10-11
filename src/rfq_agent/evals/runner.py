"""Eval suite: extraction field accuracy, flag recall, routing accuracy and unsafe auto-approvals.

The gate fails when any case that should stop or go to a person is awarded by the agent alone.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from rfq_agent.agent.extract import extract_quote
from rfq_agent.checks import CheckContext, check_quotes
from rfq_agent.gate import decide_line
from rfq_agent.llm.base import LlmProvider
from rfq_agent.models import Demand, QuoteEmail, QuoteItem, Rfq, RfqLine, Route
from rfq_agent.planner import plan_line
from rfq_agent.policy import Policy
from rfq_agent.seed import AS_OF, buyers_by_id, catalog_by_id, demo_rfq, suppliers_by_id

CASES_PATH = Path(__file__).with_name("cases.yaml")
NEEDED_BY_DAYS = 65
MIN_ROUTING_ACCURACY = 0.9


class CaseEmail(BaseModel):
    subject: str
    body: str


class CaseDemand(BaseModel):
    buyer_id: str
    quantity: int


class Expected(BaseModel):
    fields: dict[str, Any]
    flags: list[str]
    route: Route


class EvalCase(BaseModel):
    id: str
    title: str
    rule: str
    product_id: str
    demands: list[CaseDemand]
    expected: Expected
    email_id: str | None = None
    supplier_id: str | None = None
    email: CaseEmail | None = None


class CaseResult(BaseModel):
    id: str
    title: str
    rule: str
    expected_route: Route
    route: Route | None
    rules_fired: list[str] = Field(default_factory=list)
    field_hits: int = 0
    field_total: int = 0
    field_misses: list[str] = Field(default_factory=list)
    flags_expected: list[str] = Field(default_factory=list)
    flags_found: list[str] = Field(default_factory=list)
    unsafe: bool = False
    error: str | None = None
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class EvalReport(BaseModel):
    mode: str
    disabled_rules: list[str]
    cases: list[CaseResult]
    field_accuracy: float
    flag_recall: float
    routing_accuracy: float
    unsafe_auto_approvals: int
    errors: int
    passed: bool
    cost_usd: float


def load_cases(path: Path = CASES_PATH) -> list[EvalCase]:
    return [EvalCase.model_validate(c) for c in yaml.safe_load(path.read_text(encoding="utf-8"))]


def case_email(case: EvalCase) -> QuoteEmail:
    if case.email_id:
        return next(e for e in demo_rfq().emails if e.id == case.email_id)
    assert case.email is not None and case.supplier_id is not None
    return QuoteEmail(
        id=f"EV-{case.id}",
        supplier_id=case.supplier_id,
        received_at=datetime.combine(AS_OF, datetime.min.time()),
        subject=case.email.subject,
        body=case.email.body.strip(),
    )


def _same(expected: Any, actual: Any) -> bool:
    if expected is None or actual is None:
        return expected is actual
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, int | float) and isinstance(actual, int | float):
        return math.isclose(float(expected), float(actual), rel_tol=0.005)
    return str(expected).strip().lower() == str(actual).strip().lower()


def score_fields(expected: dict[str, Any], item: QuoteItem | None) -> tuple[int, list[str]]:
    misses = []
    for key, want in expected.items():
        got = getattr(item, key) if item is not None else None
        if not _same(want, got):
            misses.append(f"{key}: expected {want!r}, got {got!r}")
    return len(expected) - len(misses), misses


async def run_case(case: EvalCase, provider: LlmProvider, policy: Policy) -> CaseResult:
    result = CaseResult(
        id=case.id,
        title=case.title,
        rule=case.rule,
        expected_route=case.expected.route,
        route=None,
        field_total=len(case.expected.fields),
        flags_expected=case.expected.flags,
    )
    suppliers = suppliers_by_id()
    try:
        email = case_email(case)
        extracted = await extract_quote(email, suppliers.get(email.supplier_id), provider)
        result.cost_usd = sum(r.cost_usd for r in extracted.responses)
        result.latency_ms = sum(r.latency_ms for r in extracted.responses)
        line = RfqLine(
            id="L1",
            product_id=case.product_id,
            demands=[
                Demand(buyer_id=d.buyer_id, quantity=d.quantity, needed_by=AS_OF + timedelta(days=NEEDED_BY_DAYS))
                for d in case.demands
            ],
        )
        rfq = Rfq(
            id=f"RFQ-EVAL-{case.id}", title=case.title, issued_on=AS_OF, closes_on=AS_OF, lines=[line], emails=[email]
        )
        ctx = CheckContext(
            as_of=AS_OF, policy=policy, rfq=rfq, products=catalog_by_id(), suppliers=suppliers, buyers=buyers_by_id()
        )
        quotes = check_quotes(email, extracted.quote, ctx)
        item = next((i for i in extracted.quote.items if i.product_id == case.product_id), None)
        result.field_hits, result.field_misses = score_fields(case.expected.fields, item)
        found = {f.check for q in quotes for f in q.flags}
        result.flags_found = sorted(found)
        plan = plan_line(line, quotes, suppliers, buyers_by_id())
        decision = decide_line(line, plan, quotes, catalog_by_id()[case.product_id], policy)
        result.route = decision.route
        result.rules_fired = [h.rule for h in decision.rules_fired]
        result.unsafe = decision.route == "act_alone" and case.expected.route != "act_alone"
    except Exception as exc:  # a failing case is scored as wrong, never hidden
        result.error = f"{type(exc).__name__}: {exc}"[:300]
    return result


async def run_evals(
    provider: LlmProvider, policy: Policy, cases: list[EvalCase] | None = None, concurrency: int = 4
) -> EvalReport:
    cases = cases or load_cases()
    sem = asyncio.Semaphore(concurrency)

    async def one(c: EvalCase) -> CaseResult:
        async with sem:
            return await run_case(c, provider, policy)

    results = list(await asyncio.gather(*(one(c) for c in cases)))
    field_total = sum(r.field_total for r in results)
    flag_total = sum(len(r.flags_expected) for r in results)
    flag_hits = sum(len(set(r.flags_expected) & set(r.flags_found)) for r in results)
    routing = sum(1 for r in results if r.route == r.expected_route) / len(results)
    unsafe = sum(1 for r in results if r.unsafe)
    errors = sum(1 for r in results if r.error)
    return EvalReport(
        mode=provider.mode,
        disabled_rules=sorted(policy.disabled_rules),
        cases=results,
        field_accuracy=round(sum(r.field_hits for r in results) / field_total, 4) if field_total else 1.0,
        flag_recall=round(flag_hits / flag_total, 4) if flag_total else 1.0,
        routing_accuracy=round(routing, 4),
        unsafe_auto_approvals=unsafe,
        errors=errors,
        passed=unsafe == 0 and errors == 0 and routing >= MIN_ROUTING_ACCURACY,
        cost_usd=round(sum(r.cost_usd for r in results), 4),
    )
