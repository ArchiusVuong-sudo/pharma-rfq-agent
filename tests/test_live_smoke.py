"""Opt-in: one real call through the Claude Agent SDK. Run with LIVE=1 (needs the Claude Code login)."""

from __future__ import annotations

import os

import pytest

from rfq_agent.agent.extract import extract_quote
from rfq_agent.llm.live import LiveProvider
from rfq_agent.seed import demo_rfq, suppliers_by_id

pytestmark = pytest.mark.skipif(os.environ.get("LIVE") != "1", reason="set LIVE=1 to call the real model")


async def test_live_extraction_reads_the_new_supplier_quote() -> None:
    email = next(e for e in demo_rfq().emails if e.id == "E2")
    result = await extract_quote(email, suppliers_by_id()[email.supplier_id], LiveProvider())
    [item] = result.quote.items
    assert item.product_id == "P-AMOX-250-DT"
    assert item.currency == "EUR" and item.price_per_quantity == 1000 and item.incoterm == "CIP"
    response = result.responses[-1]
    assert response.source == "live" and response.tool_calls and response.cost_usd > 0
