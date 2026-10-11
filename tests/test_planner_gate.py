from __future__ import annotations

from datetime import date

import pytest

from rfq_agent.gate import decide_line
from rfq_agent.models import Allocation, CheckedQuote, Demand, Flag, LinePlan, RfqLine, Severity
from rfq_agent.planner import plan_line, validate_plan
from rfq_agent.policy import Policy
from rfq_agent.seed import buyers_by_id, catalog_by_id, suppliers_by_id
from tests.conftest import item

NEEDED = date(2026, 12, 15)
AMOX = RfqLine(
    id="L2",
    product_id="P-AMOX-250-DT",
    demands=[
        Demand(buyer_id="B-UG-01", quantity=250_000, needed_by=NEEDED),
        Demand(buyer_id="B-KE-01", quantity=180_000, needed_by=NEEDED),
        Demand(buyer_id="B-MW-01", quantity=150_000, needed_by=NEEDED),
    ],
)


def quote(ref: str, supplier: str, landed: float, flags: list[Flag] | None = None, **overrides: object) -> CheckedQuote:
    return CheckedQuote(
        ref=ref,
        email_id=ref.split("#")[0],
        supplier_id=supplier,
        line_id="L2",
        item=item(product_id="P-AMOX-250-DT", **overrides),
        landed_unit_usd=landed,
        flags=flags or [],
    )


KES = quote("E1#1", "S-KES", 0.0488, moq_units=50_000, max_units=600_000)
BRW = quote("E2#0", "S-BRW", 0.0419, moq_units=100_000, max_units=200_000)


def plan(*quotes: CheckedQuote, line: RfqLine = AMOX) -> LinePlan:
    return plan_line(line, list(quotes), suppliers_by_id(), buyers_by_id())


def split(p: LinePlan) -> dict[tuple[str, str], int]:
    return {(a.buyer_id, a.supplier_id): a.quantity for a in p.allocations}


def test_cheapest_registered_supplier_wins_each_buyer() -> None:
    assert split(plan(KES, BRW)) == {
        ("B-UG-01", "S-KES"): 250_000,
        ("B-KE-01", "S-BRW"): 180_000,
        ("B-MW-01", "S-KES"): 150_000,
    }


def test_capacity_spills_to_next_supplier() -> None:
    small = quote("E2#0", "S-BRW", 0.0419, moq_units=10_000, max_units=100_000)
    assert split(plan(KES, small))[("B-KE-01", "S-BRW")] == 100_000
    assert split(plan(KES, small))[("B-KE-01", "S-KES")] == 80_000


def test_supplier_under_moq_is_dropped() -> None:
    big_moq = quote("E2#0", "S-BRW", 0.0419, moq_units=500_000, max_units=600_000)
    assert ("B-KE-01", "S-BRW") not in split(plan(KES, big_moq))


def test_unmet_demand_when_no_registered_supplier() -> None:
    p = plan(BRW)
    assert p.unmet == {"B-UG-01": 250_000, "B-MW-01": 150_000}


def test_unusable_quotes_are_never_planned() -> None:
    held = quote("E4#0", "S-KES", 0.01, flags=[Flag(check="coa_missing", severity=Severity.HOLD, label="", detail="")])
    assert plan(held).covered_units == 0


def alloc(buyer: str, supplier: str, ref: str, qty: int) -> Allocation:
    return Allocation(buyer_id=buyer, supplier_id=supplier, quote_ref=ref, quantity=qty)


def errors(*allocations: Allocation) -> list[str]:
    return validate_plan(
        LinePlan(line_id="L2", allocations=list(allocations)), AMOX, [KES, BRW], suppliers_by_id(), buyers_by_id()
    )


def test_valid_edit_passes() -> None:
    assert errors(alloc("B-KE-01", "S-BRW", "E2#0", 100_000), alloc("B-KE-01", "S-KES", "E1#1", 80_000)) == []


def test_edit_under_moq_is_rejected() -> None:
    assert errors(alloc("B-KE-01", "S-BRW", "E2#0", 50_000)) == [
        "Brightwater Pharma: 50,000 units is under its MOQ of 100,000"
    ]


def test_edit_over_capacity_is_rejected() -> None:
    assert "over its capacity" in errors(alloc("B-KE-01", "S-BRW", "E2#0", 250_000))[0]


def test_edit_to_unregistered_country_is_rejected() -> None:
    assert "Brightwater Pharma is not registered in Uganda" in errors(alloc("B-UG-01", "S-BRW", "E2#0", 150_000))


def test_edit_over_demand_and_unknown_quote_are_rejected() -> None:
    assert any("more than its demand" in e for e in errors(alloc("B-KE-01", "S-KES", "E1#1", 200_000)))
    assert any("Unknown quote" in e for e in errors(alloc("B-KE-01", "S-KES", "E9#0", 1_000)))


def decide(p: LinePlan, quotes: list[CheckedQuote], policy: Policy, product: str = "P-AMOX-250-DT") -> list[str]:
    d = decide_line(AMOX, p, quotes, catalog_by_id()[product], policy)
    return [d.route, *sorted(h.rule for h in d.rules_fired)]


def test_small_clean_award_acts_alone(policy: Policy) -> None:
    line = RfqLine(
        id="L2", product_id="P-AMOX-250-DT", demands=[Demand(buyer_id="B-UG-01", quantity=100_000, needed_by=NEEDED)]
    )
    p = plan(KES, line=line)
    assert decide_line(line, p, [KES], catalog_by_id()["P-AMOX-250-DT"], policy).route == "act_alone"


def test_value_over_limit_needs_review(policy: Policy) -> None:
    assert decide(plan(KES), [KES], policy) == ["review", "value_limit"]


def test_flags_on_awarded_quotes_fire_rules(policy: Policy) -> None:
    outlier = quote(
        "E1#1",
        "S-KES",
        0.0488,
        flags=[Flag(check="price_far_under", severity=Severity.REVIEW, label="Far under", detail="x")],
        max_units=600_000,
    )
    assert "price_outlier" in decide(plan(outlier), [outlier], policy)


def test_no_supply_stops_and_strictest_route_wins(policy: Policy) -> None:
    assert decide(LinePlan(line_id="L2", allocations=[]), [], policy)[0] == "stop"
    assert decide(plan(BRW), [BRW], policy)[0] == "review"  # partial supply + new supplier, no stop


def test_cold_chain_product_needs_review(policy: Policy) -> None:
    p = LinePlan(line_id="L2", allocations=[alloc("B-UG-01", "S-HAL", "E4#0", 100)])
    assert "cold_chain" in decide(p, [], policy, product="P-OXY-10-AMP")


def test_switching_a_guardrail_off(policy: Policy) -> None:
    outlier = quote(
        "E1#1",
        "S-KES",
        0.0488,
        flags=[Flag(check="price_far_under", severity=Severity.REVIEW, label="Far under", detail="x")],
        max_units=600_000,
    )
    line = RfqLine(
        id="L2", product_id="P-AMOX-250-DT", demands=[Demand(buyer_id="B-UG-01", quantity=30_000, needed_by=NEEDED)]
    )
    p = plan(outlier, line=line)
    product = catalog_by_id()["P-AMOX-250-DT"]
    assert decide_line(line, p, [outlier], product, policy).route == "review"
    assert decide_line(line, p, [outlier], product, policy.without("price_outlier")).route == "act_alone"
    with pytest.raises(ValueError):
        policy.without("no_such_rule")
