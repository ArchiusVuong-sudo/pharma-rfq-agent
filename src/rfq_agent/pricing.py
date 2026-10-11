"""Price normalisation: quoted price -> USD per unit -> landed USD per unit (demo FX and uplifts)."""

from __future__ import annotations

from rfq_agent.models import QuoteItem
from rfq_agent.policy import Policy


def unit_price_usd(item: QuoteItem, policy: Policy) -> float | None:
    """USD per single unit (tablet, ampoule), or None when the price cannot be computed."""
    if item.price_amount is None or item.currency is None or not item.price_per_quantity:
        return None
    rate = policy.fx_to_usd.get(item.currency)
    if rate is None or item.price_amount <= 0 or item.price_per_quantity <= 0:
        return None
    return item.price_amount / item.price_per_quantity * rate


def landed_unit_usd(item: QuoteItem, policy: Policy) -> float | None:
    """Unit price plus the Incoterm freight and insurance uplift."""
    base = unit_price_usd(item, policy)
    if base is None or item.incoterm is None:
        return None
    uplift = policy.incoterm_uplift.get(item.incoterm)
    if uplift is None:
        return None
    return round(base * (1 + uplift), 5)
