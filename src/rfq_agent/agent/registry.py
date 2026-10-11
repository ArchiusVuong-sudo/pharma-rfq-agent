"""Supplier registry lookup (sourcing), with a timeout, retries and an injectable fault for demos.

FAULT_ONCE=registry_timeout makes the first lookup in this process time out, so the trace shows
the retry and the alert.
"""

from __future__ import annotations

import asyncio
import os
import threading

from pydantic import BaseModel, Field

from rfq_agent.models import Buyer, RfqLine, Supplier

LOOKUP_TIMEOUT_S = 1.0
MAX_ATTEMPTS = 3

_fault_lock = threading.Lock()
_fault_used: set[str] = set()


def take_fault(name: str) -> bool:
    """True once per process when FAULT_ONCE names this fault."""
    wanted = {f.strip() for f in os.environ.get("FAULT_ONCE", "").split(",") if f.strip()}
    with _fault_lock:
        if name in wanted and name not in _fault_used:
            _fault_used.add(name)
            return True
    return False


def reset_faults() -> None:
    with _fault_lock:
        _fault_used.clear()


class ShortlistEntry(BaseModel):
    supplier_id: str
    countries: list[str] = Field(default_factory=list)


async def _lookup(line: RfqLine, suppliers: list[Supplier], buyers: dict[str, Buyer]) -> list[ShortlistEntry]:
    if take_fault("registry_timeout"):
        await asyncio.sleep(LOOKUP_TIMEOUT_S * 3)  # the lookup hangs; the timeout below cuts it off
    countries = {buyers[d.buyer_id].country for d in line.demands}
    out = []
    for s in suppliers:
        if not s.in_registry:
            continue
        served = sorted(set(s.registrations.get(line.product_id, [])) & countries)
        if served:
            out.append(ShortlistEntry(supplier_id=s.id, countries=served))
    return out


async def shortlist(
    line: RfqLine, suppliers: list[Supplier], buyers: dict[str, Buyer]
) -> tuple[list[ShortlistEntry], list[str]]:
    """Registry suppliers registered for the product in at least one buyer country.

    Returns the shortlist and the errors of attempts that failed before it succeeded.
    """
    errors: list[str] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return await asyncio.wait_for(_lookup(line, suppliers, buyers), LOOKUP_TIMEOUT_S), errors
        except TimeoutError:
            errors.append(f"registry lookup timed out after {LOOKUP_TIMEOUT_S:.0f}s (attempt {attempt})")
            await asyncio.sleep(0.2 * attempt)
    raise TimeoutError("; ".join(errors))
