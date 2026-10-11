"""Autonomy gate: one policy decides per line whether the agent acts alone, asks a person or stops."""

from __future__ import annotations

from rfq_agent.models import CheckedQuote, LineDecision, LinePlan, Product, RfqLine, RuleHit, Route
from rfq_agent.policy import ROUTE_ORDER, Policy

FLAG_RULES = {
    "new_supplier": "new_supplier",
    "price_far_under": "price_outlier",
    "price_far_over": "price_outlier",
    "gmp_expiring": "gmp_expiring",
    "late_delivery": "late_delivery",
}


def decide_line(
    line: RfqLine,
    plan: LinePlan,
    quotes: list[CheckedQuote],
    product: Product,
    policy: Policy,
    supplier_names: dict[str, str] | None = None,
) -> LineDecision:
    hits: list[RuleHit] = []

    def fire(rule_id: str, reason: str) -> None:
        rule = policy.rule(rule_id)
        if rule and not any(h.rule == rule_id for h in hits):
            hits.append(RuleHit(rule=rule_id, route=rule.route, reason=reason))

    if plan.covered_units == 0:
        fire("no_supply", "No compliant quote can serve any buyer yet")
    elif plan.unmet:
        fire("partial_supply", f"{sum(plan.unmet.values()):,} units have no compliant supplier")

    value_rule = policy.rule("value_limit")
    if value_rule and value_rule.limit_usd is not None and plan.total_usd > value_rule.limit_usd:
        fire("value_limit", f"Award value ${plan.total_usd:,.0f} is over the ${value_rule.limit_usd:,.0f} limit")

    if product.cold_chain and plan.covered_units:
        fire("cold_chain", f"{product.name} needs 2-8 C storage and transport")

    awarded = {a.quote_ref for a in plan.allocations if a.quantity}
    for q in quotes:
        if q.ref not in awarded:
            continue
        for flag in q.flags:
            rule_id = FLAG_RULES.get(flag.check)
            if rule_id:
                who = (supplier_names or {}).get(q.supplier_id, q.supplier_id)
                fire(rule_id, f"{who}: {flag.label.lower()} ({flag.detail})")

    route: Route = "act_alone"
    for h in hits:
        if ROUTE_ORDER[h.route] > ROUTE_ORDER[route]:
            route = h.route
    return LineDecision(line_id=line.id, route=route, rules_fired=hits)
