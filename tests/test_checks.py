from __future__ import annotations

from datetime import datetime

from rfq_agent.checks import CheckContext, apply_prices, check_email, check_quotes, merge_reply
from rfq_agent.models import ExtractedQuote, QuoteEmail, Severity
from tests.conftest import item


def email(supplier_id: str) -> QuoteEmail:
    return QuoteEmail(id="T1", supplier_id=supplier_id, received_at=datetime(2026, 10, 1), subject="s", body="b")


def checks(ctx: CheckContext, supplier_id: str = "S-KES", **overrides: object) -> dict[str, Severity]:
    quote = ExtractedQuote(supplier_name=None, items=[item(**overrides)], unreadable_reason=None)
    [cq] = check_quotes(email(supplier_id), quote, ctx)
    return {f.check: f.severity for f in cq.flags}


def test_clean_quote_has_no_flags(ctx: CheckContext) -> None:
    assert checks(ctx) == {}


def test_supplier_not_in_registry_is_blocked(ctx: CheckContext) -> None:
    assert checks(ctx, "S-TAM")["registry"] is Severity.BLOCK


def test_expired_gmp_is_blocked(ctx: CheckContext) -> None:
    got = checks(
        ctx, "S-CRV", product_id="P-OXY-10-AMP", storage_condition="2-8 C", price_amount=0.25, price_per_quantity=1
    )
    assert got["gmp_expired"] is Severity.BLOCK


def test_gmp_expiring_and_new_supplier_need_review(ctx: CheckContext) -> None:
    got = checks(ctx, "S-BRW", product_id="P-AMOX-250-DT", price_amount=4.4)
    assert got["gmp_expiring"] is Severity.REVIEW
    assert got["new_supplier"] is Severity.REVIEW


def test_partial_registration_is_info(ctx: CheckContext) -> None:
    got = checks(ctx, "S-BRW", product_id="P-AMOX-250-DT", price_amount=4.4)
    assert got["registration"] is Severity.INFO


def test_missing_documents_hold_the_quote(ctx: CheckContext) -> None:
    got = checks(ctx, coa_provided=None, shelf_life_months=None, lead_time_days=None, incoterm=None)
    assert {k for k, v in got.items() if v is Severity.HOLD} == {
        "coa_missing",
        "shelf_life_missing",
        "lead_time_missing",
        "incoterm_missing",
    }


def test_cold_chain_needs_stated_storage(ctx: CheckContext) -> None:
    base = {"product_id": "P-OXY-10-AMP", "price_amount": 0.27, "price_per_quantity": 1}
    assert "storage_missing" in checks(ctx, "S-HAL", **base)
    assert "storage_missing" not in checks(ctx, "S-HAL", storage_condition="store at 2 to 8 °C", **base)


def test_short_shelf_life_on_arrival_is_blocked(ctx: CheckContext) -> None:
    assert checks(ctx, shelf_life_months=20, lead_time_days=60)["shelf_life_short"] is Severity.BLOCK


def test_late_delivery_needs_review(ctx: CheckContext) -> None:
    assert checks(ctx, lead_time_days=90)["late_delivery"] is Severity.REVIEW


def test_price_far_under_and_over_reference(ctx: CheckContext) -> None:
    assert checks(ctx, price_amount=1.0)["price_far_under"] is Severity.REVIEW
    assert checks(ctx, price_amount=5.0)["price_far_over"] is Severity.REVIEW
    assert "price_far_under" not in checks(ctx, price_amount=2.2)


def test_unreadable_email_holds_with_no_price(ctx: CheckContext) -> None:
    quote = ExtractedQuote(supplier_name=None, items=[], unreadable_reason="attachment missing")
    [cq] = check_quotes(email("S-ORR"), quote, ctx)
    assert cq.line_id is None and cq.unit_price_usd is None and not cq.usable
    assert [f.check for f in cq.flags] == ["price_unreadable"]


def test_product_not_on_rfq_is_info(ctx: CheckContext) -> None:
    [cq] = check_quotes(
        email("S-KES"),
        ExtractedQuote(supplier_name=None, items=[item(product_id="P-PARA-500-TAB")], unreadable_reason=None),
        ctx,
    )
    assert cq.line_id is None and cq.flags[-1].check == "not_on_rfq"


def test_compliance_does_not_price_and_prices_replace_old_price_flags(ctx: CheckContext) -> None:
    quote = ExtractedQuote(supplier_name=None, items=[item(price_amount=1.0)], unreadable_reason=None)
    [raw] = check_email(email("S-KES"), quote, ctx)
    assert raw.landed_unit_usd is None
    [priced] = apply_prices(apply_prices([raw], ctx), ctx)
    assert [f.check for f in priced.flags].count("price_far_under") == 1


def test_merge_reply_fills_only_missing_fields() -> None:
    original = ExtractedQuote(
        supplier_name="Halden",
        items=[item(coa_provided=None, shelf_life_months=None, price_amount=0.27)],
        unreadable_reason=None,
    )
    reply = ExtractedQuote(
        supplier_name=None,
        items=[item(coa_provided=True, shelf_life_months=30, price_amount=None, storage_condition="2-8 C")],
        unreadable_reason=None,
    )
    [merged] = merge_reply(original, reply).items
    assert merged.coa_provided is True and merged.shelf_life_months == 30
    assert merged.price_amount == 0.27  # the reply did not restate a price, so the original stays
    assert merged.storage_condition == "2-8 C"
