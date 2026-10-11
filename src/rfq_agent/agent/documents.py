"""PO drafts, award and regret notices. Written by code from the approved plan, never by the model."""

from __future__ import annotations

from collections import defaultdict

from rfq_agent.models import Buyer, LinePlan, Product, Rfq, RfqLine, Supplier


def po_drafts(rfq: Rfq, line: RfqLine, plan: LinePlan, product: Product,
              suppliers: dict[str, Supplier], buyers: dict[str, Buyer]) -> list[tuple[str, str, str]]:
    """One PO draft per supplier: (supplier id, subject, body)."""
    by_supplier: dict[str, list[str]] = defaultdict(list)
    totals: dict[str, float] = defaultdict(float)
    needed = {d.buyer_id: d.needed_by for d in line.demands}
    for a in plan.allocations:
        if not a.quantity:
            continue
        buyer = buyers[a.buyer_id]
        by_supplier[a.supplier_id].append(
            f"- Ship to {buyer.name} ({buyer.country_name}): {a.quantity:,} {product.unit}s "
            f"at ${a.landed_unit_usd:.4f} landed = ${a.total_usd:,.2f}, needed by {needed[a.buyer_id]}"
        )
        totals[a.supplier_id] += a.total_usd
    out = []
    for sid, rows in by_supplier.items():
        body = (
            f"PURCHASE ORDER DRAFT - {rfq.id} line {line.id}\n"
            f"Supplier: {suppliers[sid].name}\nProduct: {product.name}\n\n"
            + "\n".join(rows)
            + f"\n\nTotal (landed, USD): ${totals[sid]:,.2f}\n"
            "Landed prices use the demo FX rates and Incoterm uplifts in policy.yaml.\n"
            "Draft for the buyers' signature. Simulated send, synthetic data."
        )
        out.append((sid, f"PO draft {rfq.id}-{line.id}: {product.name}", body))
    return out


def award_notice(rfq: Rfq, line: RfqLine, plan: LinePlan, product: Product, supplier: Supplier) -> str:
    units = sum(a.quantity for a in plan.allocations if a.supplier_id == supplier.id)
    return (
        f"Dear {supplier.contact_name},\n\nWe are pleased to award {units:,} {product.unit}s of "
        f"{product.name} on {rfq.id} to {supplier.name}. The PO draft follows separately for signature.\n\n"
        "Kind regards,\nProcurement team"
    )


def regret_notice(rfq: Rfq, product: Product, supplier: Supplier) -> str:
    return (
        f"Dear {supplier.contact_name},\n\nThank you for your quote for {product.name} on {rfq.id}. "
        "This time the line was awarded to other suppliers. We hope to work with you on a future RFQ.\n\n"
        "Kind regards,\nProcurement team"
    )
