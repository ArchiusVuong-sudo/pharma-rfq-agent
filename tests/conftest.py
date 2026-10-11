"""Tests run in replay mode: recorded model outputs, no login, no network."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ["DEMO_REPLAY"] = "1"
os.environ["REPLAY_DELAY_SCALE"] = "0"
os.environ.pop("FAULT_ONCE", None)

from rfq_agent.agent.registry import reset_faults
from rfq_agent.checks import CheckContext
from rfq_agent.models import QuoteItem
from rfq_agent.policy import Policy, load_policy
from rfq_agent.seed import AS_OF, buyers_by_id, catalog_by_id, demo_rfq, suppliers_by_id


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("RFQ_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("FAULT_ONCE", raising=False)
    reset_faults()
    yield
    reset_faults()


@pytest.fixture
def policy() -> Policy:
    return load_policy()


@pytest.fixture
def ctx(policy: Policy) -> CheckContext:
    return CheckContext(
        as_of=AS_OF,
        policy=policy,
        rfq=demo_rfq(),
        products=catalog_by_id(),
        suppliers=suppliers_by_id(),
        buyers=buyers_by_id(),
    )


def item(**overrides: object) -> QuoteItem:
    """A complete, compliant zinc quote item; override fields to break one thing at a time."""
    base: dict[str, object] = {
        "product_text": "zinc 20mg DT",
        "product_id": "P-ZINC-20-DT",
        "price_amount": 2.40,
        "currency": "USD",
        "price_per_quantity": 100,
        "incoterm": "FCA",
        "moq_units": 20_000,
        "max_units": 300_000,
        "lead_time_days": 30,
        "shelf_life_months": 30,
        "coa_provided": True,
        "storage_condition": None,
        "gmp_certificate_number": "SYN-GMP-4471",
    }
    base.update(overrides)
    return QuoteItem.model_validate(base)
