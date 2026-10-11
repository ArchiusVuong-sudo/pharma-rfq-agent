"""Award planner (order matching) and the validator a reviewer's edited split goes through.

Each buyer's demand goes to compliant suppliers registered in that buyer's country, cheapest
landed unit cost first, within each supplier's capacity. A supplier whose pooled total ends up
under its MOQ is dropped and its share goes to the next supplier.
"""

from __future__ import annotations

from collections import defaultdict

from rfq_agent.models import Allocation, Buyer, CheckedQuote, LinePlan, RfqLine, Supplier


def _eligible(
    quotes: list[CheckedQuote], line: RfqLine, buyer: Buyer, suppliers: dict[str, Supplier]
) -> list[CheckedQuote]:
    out = []
    for q in quotes:
        if q.line_id != line.id or not q.usable or q.landed_unit_usd is None:
            continue
        supplier = suppliers.get(q.supplier_id)
        if supplier and buyer.country in supplier.registrations.get(line.product_id, []):
            out.append(q)
    return sorted(out, key=lambda q: (q.landed_unit_usd or 0.0, q.supplier_id))


def plan_line(
    line: RfqLine,
    quotes: list[CheckedQuote],
    suppliers: dict[str, Supplier],
    buyers: dict[str, Buyer],
) -> LinePlan:
    excluded: set[str] = set()
    for _ in range(len(quotes) + 1):
        allocations: list[Allocation] = []
        unmet: dict[str, int] = {}
        used: dict[str, int] = defaultdict(int)
        for demand in line.demands:
            buyer = buyers[demand.buyer_id]
            need = demand.quantity
            for q in _eligible(quotes, line, buyer, suppliers):
                if q.ref in excluded or need == 0:
                    continue
                cap = q.item.max_units if q.item.max_units is not None else need
                take = min(need, max(cap - used[q.ref], 0))
                if take <= 0:
                    continue
                used[q.ref] += take
                need -= take
                allocations.append(
                    Allocation(
                        buyer_id=buyer.id,
                        supplier_id=q.supplier_id,
                        quote_ref=q.ref,
                        quantity=take,
                        landed_unit_usd=q.landed_unit_usd or 0.0,
                    )
                )
            if need:
                unmet[buyer.id] = need
        under_moq = [
            q.ref
            for q in quotes
            if used.get(q.ref) and q.item.moq_units and used[q.ref] < q.item.moq_units
        ]
        if not under_moq:
            return LinePlan(line_id=line.id, allocations=allocations, unmet=unmet)
        excluded.add(under_moq[0])
    raise RuntimeError("planner did not converge")  # pragma: no cover


def validate_plan(
    plan: LinePlan,
    line: RfqLine,
    quotes: list[CheckedQuote],
    suppliers: dict[str, Supplier],
    buyers: dict[str, Buyer],
) -> list[str]:
    """Problems with a (possibly hand-edited) plan. Empty means it can run."""
    errors: list[str] = []
    by_ref = {q.ref: q for q in quotes}
    totals: dict[str, int] = defaultdict(int)
    per_buyer: dict[str, int] = defaultdict(int)
    demand = {d.buyer_id: d.quantity for d in line.demands}
    for a in plan.allocations:
        if a.quantity == 0:
            continue
        q = by_ref.get(a.quote_ref)
        supplier = suppliers.get(a.supplier_id)
        buyer = buyers.get(a.buyer_id)
        if q is None or supplier is None or buyer is None or q.supplier_id != a.supplier_id:
            errors.append(f"Unknown quote, supplier or buyer in allocation {a.quote_ref}")
            continue
        if a.buyer_id not in demand:
            errors.append(f"{buyer.name} has no demand on this line")
        if not q.usable or q.line_id != line.id:
            errors.append(f"{supplier.name}'s quote is not compliant for this line")
        if buyer.country not in supplier.registrations.get(line.product_id, []):
            errors.append(f"{supplier.name} is not registered in {buyer.country_name}")
        totals[a.quote_ref] += a.quantity
        per_buyer[a.buyer_id] += a.quantity
    for ref, total in totals.items():
        q = by_ref[ref]
        name = suppliers[q.supplier_id].name
        if q.item.max_units is not None and total > q.item.max_units:
            errors.append(f"{name}: {total:,} units is over its capacity of {q.item.max_units:,}")
        if q.item.moq_units and total < q.item.moq_units:
            errors.append(f"{name}: {total:,} units is under its MOQ of {q.item.moq_units:,}")
    for buyer_id, qty in demand.items():
        got = per_buyer.get(buyer_id, 0)
        if got > qty:
            errors.append(f"{buyers[buyer_id].name}: {got:,} units is more than its demand of {qty:,}")
    return errors
