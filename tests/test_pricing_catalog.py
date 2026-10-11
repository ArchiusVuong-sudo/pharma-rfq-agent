from __future__ import annotations

import pytest

from rfq_agent.catalog import CatalogQuery, search
from rfq_agent.policy import Policy
from rfq_agent.pricing import landed_unit_usd, unit_price_usd
from tests.conftest import item


def test_unit_price_per_box(policy: Policy) -> None:
    assert unit_price_usd(item(price_amount=2.40, price_per_quantity=100), policy) == pytest.approx(0.024)


def test_unit_price_converts_currency(policy: Policy) -> None:
    got = unit_price_usd(item(price_amount=38.0, currency="EUR", price_per_quantity=1000), policy)
    assert got == pytest.approx(0.038 * 1.08)


def test_landed_adds_incoterm_uplift(policy: Policy) -> None:
    assert landed_unit_usd(item(incoterm="EXW"), policy) == pytest.approx(0.024 * 1.10, rel=1e-4)
    assert landed_unit_usd(item(incoterm="DDP"), policy) == pytest.approx(0.024, rel=1e-4)


@pytest.mark.parametrize(
    "overrides",
    [
        {"price_amount": None},
        {"currency": None},
        {"price_per_quantity": None},
        {"price_per_quantity": 0},
        {"price_amount": -1.0},
    ],
)
def test_no_price_when_any_part_is_missing(policy: Policy, overrides: dict[str, object]) -> None:
    assert unit_price_usd(item(**overrides), policy) is None
    assert landed_unit_usd(item(**overrides), policy) is None


def test_no_landed_cost_without_incoterm(policy: Policy) -> None:
    assert unit_price_usd(item(incoterm=None), policy) is not None
    assert landed_unit_usd(item(incoterm=None), policy) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Amoxycillin 250mg dispersible", "P-AMOX-250-DT"),
        ("oxytocin 10iu/ml amp", "P-OXY-10-AMP"),
        ("zinc sulphate dispersible", "P-ZINC-20-DT"),
        ("acetaminophen 500mg", "P-PARA-500-TAB"),
    ],
)
def test_search_finds_synonyms(text: str, expected: str) -> None:
    assert search(CatalogQuery(query=text))[0].product_id == expected


def test_search_penalises_strength_mismatch() -> None:
    right = search(CatalogQuery(query="amoxicillin 250 mg"))[0]
    wrong = search(CatalogQuery(query="amoxicillin 500 mg"))[0]
    assert right.score > wrong.score


def test_search_returns_nothing_for_unknown_product() -> None:
    assert search(CatalogQuery(query="insulin glargine pen")) == []


def test_query_input_is_validated() -> None:
    with pytest.raises(ValueError):
        CatalogQuery(query="x")
    with pytest.raises(ValueError):
        CatalogQuery(query="zinc", limit=50)
