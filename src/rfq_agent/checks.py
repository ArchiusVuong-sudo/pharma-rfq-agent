"""Compliance and price checks. Every check runs on every quote item, in code, and names itself."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from rfq_agent.models import (
    Buyer,
    CheckedQuote,
    ExtractedQuote,
    Flag,
    Product,
    QuoteEmail,
    QuoteItem,
    Rfq,
    RfqLine,
    Severity,
    Supplier,
)
from rfq_agent.policy import Policy
from rfq_agent.pricing import landed_unit_usd, unit_price_usd


@dataclass(frozen=True)
class CheckContext:
    as_of: date
    policy: Policy
    rfq: Rfq
    products: dict[str, Product]
    suppliers: dict[str, Supplier]
    buyers: dict[str, Buyer]

    def line_for(self, product_id: str | None) -> RfqLine | None:
        return next((ln for ln in self.rfq.lines if ln.product_id == product_id), None)


COLD_CHAIN_RE = re.compile(r"\b2\s*(?:-|to|–)\s*8\s*°?\s*c\b", re.IGNORECASE)

EMPTY_ITEM = QuoteItem(
    product_text="(no readable quote)",
    product_id=None,
    price_amount=None,
    currency=None,
    price_per_quantity=None,
    incoterm=None,
    moq_units=None,
    max_units=None,
    lead_time_days=None,
    shelf_life_months=None,
    coa_provided=None,
    storage_condition=None,
    gmp_certificate_number=None,
)


def _flag(check: str, severity: Severity, label: str, detail: str) -> Flag:
    return Flag(check=check, severity=severity, label=label, detail=detail)


def supplier_flags(supplier: Supplier | None, ctx: CheckContext) -> list[Flag]:
    if supplier is None or not supplier.in_registry:
        return [_flag("registry", Severity.BLOCK, "Not in supplier registry", "Supplier is not in the vetted registry")]
    flags: list[Flag] = []
    if supplier.gmp is None:
        flags.append(_flag("gmp_expired", Severity.BLOCK, "No GMP certificate", "No GMP certificate on file"))
    else:
        days_left = (supplier.gmp.expires_on - ctx.as_of).days
        if days_left < 0:
            flags.append(
                _flag(
                    "gmp_expired",
                    Severity.BLOCK,
                    "GMP certificate expired",
                    f"{supplier.gmp.number} expired {-days_left} days ago ({supplier.gmp.expires_on})",
                )
            )
        elif days_left <= ctx.policy.checks.gmp_expiry_warn_days:
            flags.append(
                _flag(
                    "gmp_expiring",
                    Severity.REVIEW,
                    f"GMP expires in {days_left} days",
                    f"{supplier.gmp.number} expires {supplier.gmp.expires_on}",
                )
            )
    if supplier.completed_orders < ctx.policy.checks.new_supplier_min_orders:
        flags.append(
            _flag(
                "new_supplier",
                Severity.REVIEW,
                "New supplier",
                f"{supplier.completed_orders} completed orders on the marketplace",
            )
        )
    return flags


def registration_flags(supplier: Supplier | None, line: RfqLine, ctx: CheckContext) -> list[Flag]:
    if supplier is None or not supplier.in_registry:
        return []
    countries = {ctx.buyers[d.buyer_id].country for d in line.demands}
    registered = set(supplier.registrations.get(line.product_id, [])) & countries
    if not registered:
        return [
            _flag(
                "registration",
                Severity.BLOCK,
                "Not registered in buyer countries",
                f"No product registration in {', '.join(sorted(countries))}",
            )
        ]
    if registered != countries:
        return [
            _flag(
                "registration",
                Severity.INFO,
                f"Registered in {', '.join(sorted(registered))} only",
                f"Can serve buyers in {', '.join(sorted(registered))}",
            )
        ]
    return []


def document_flags(item: QuoteItem, product: Product) -> list[Flag]:
    """Missing information: the quote waits on the supplier and gets a follow-up."""
    flags: list[Flag] = []
    if item.price_amount is None or item.currency is None or not item.price_per_quantity:
        flags.append(
            _flag("price_unreadable", Severity.HOLD, "Could not read a price", "a price with currency and pack basis")
        )
    if item.incoterm is None:
        flags.append(_flag("incoterm_missing", Severity.HOLD, "No Incoterm", "the Incoterm for the price"))
    if item.coa_provided is not True:
        flags.append(_flag("coa_missing", Severity.HOLD, "No CoA", "a certificate of analysis (CoA)"))
    if product.cold_chain and not (item.storage_condition and COLD_CHAIN_RE.search(item.storage_condition)):
        flags.append(
            _flag(
                "storage_missing",
                Severity.HOLD,
                "No 2-8 C storage confirmed",
                "confirmation of 2-8 C storage and transport",
            )
        )
    if item.shelf_life_months is None:
        flags.append(
            _flag("shelf_life_missing", Severity.HOLD, "No shelf life", "the remaining shelf life at dispatch")
        )
    if item.lead_time_days is None:
        flags.append(_flag("lead_time_missing", Severity.HOLD, "No lead time", "the lead time from PO"))
    return flags


def timing_flags(item: QuoteItem, product: Product, line: RfqLine, ctx: CheckContext) -> list[Flag]:
    flags: list[Flag] = []
    if item.shelf_life_months is not None and item.lead_time_days is not None:
        on_arrival = item.shelf_life_months - item.lead_time_days / 30
        share = on_arrival / product.total_shelf_life_months
        if share < ctx.policy.checks.shelf_life_min_share_on_arrival:
            flags.append(
                _flag(
                    "shelf_life_short",
                    Severity.BLOCK,
                    "Short shelf life on arrival",
                    f"{on_arrival:.0f} of {product.total_shelf_life_months} months left on arrival ({share:.0%})",
                )
            )
    if item.lead_time_days is not None:
        arrives = ctx.as_of + timedelta(days=item.lead_time_days)
        earliest = min(d.needed_by for d in line.demands)
        if arrives > earliest:
            flags.append(
                _flag(
                    "late_delivery",
                    Severity.REVIEW,
                    "Arrives after needed-by",
                    f"Arrives about {arrives}, first needed-by is {earliest}",
                )
            )
    return flags


def price_flags(landed: float | None, product: Product, ctx: CheckContext) -> list[Flag]:
    if landed is None:
        return []
    share = landed / product.reference_unit_price_usd
    if share < ctx.policy.checks.price_far_under_share:
        return [
            _flag(
                "price_far_under",
                Severity.REVIEW,
                "Far under reference price",
                f"${landed:.4f} landed vs ${product.reference_unit_price_usd:.4f} reference ({share - 1:+.0%})",
            )
        ]
    if share > ctx.policy.checks.price_far_over_share:
        return [
            _flag(
                "price_far_over",
                Severity.REVIEW,
                "Far over reference price",
                f"${landed:.4f} landed vs ${product.reference_unit_price_usd:.4f} reference ({share - 1:+.0%})",
            )
        ]
    return []


def check_email(email: QuoteEmail, extracted: ExtractedQuote, ctx: CheckContext) -> list[CheckedQuote]:
    """Run every compliance check on every item the model read from one email."""
    supplier = ctx.suppliers.get(email.supplier_id)
    items = extracted.items or [EMPTY_ITEM]
    out: list[CheckedQuote] = []
    for i, item in enumerate(items):
        product = ctx.products.get(item.product_id or "")
        line = ctx.line_for(item.product_id)
        cq = CheckedQuote(
            ref=f"{email.id}#{i}",
            email_id=email.id,
            supplier_id=email.supplier_id,
            line_id=line.id if line else None,
            item=item,
        )
        flags = supplier_flags(supplier, ctx)
        if product is None or line is None:
            flags.append(
                _flag(
                    "price_unreadable" if product is None else "not_on_rfq",
                    Severity.HOLD if product is None else Severity.INFO,
                    "Could not read a price" if product is None else "Not on this RFQ",
                    "a readable quote in the email body: product, price, currency and pack basis"
                    if product is None
                    else f"{product.name} is not requested",
                )
            )
            cq.flags = flags
            out.append(cq)
            continue
        flags += registration_flags(supplier, line, ctx)
        flags += document_flags(item, product)
        flags += timing_flags(item, product, line, ctx)
        cq.flags = flags
        out.append(cq)
    return out


def apply_prices(quotes: list[CheckedQuote], ctx: CheckContext) -> list[CheckedQuote]:
    """Normalise each price to USD per unit, add the landed cost and the reference-price check."""
    out: list[CheckedQuote] = []
    for q in quotes:
        product = ctx.products.get(q.item.product_id or "")
        if product is None or q.line_id is None:
            out.append(q)
            continue
        unit = unit_price_usd(q.item, ctx.policy)
        landed = landed_unit_usd(q.item, ctx.policy)
        flags = [f for f in q.flags if f.check not in ("price_far_under", "price_far_over")]
        flags += price_flags(landed, product, ctx)
        out.append(q.model_copy(update={"unit_price_usd": unit, "landed_unit_usd": landed, "flags": flags}))
    return out


def check_quotes(email: QuoteEmail, extracted: ExtractedQuote, ctx: CheckContext) -> list[CheckedQuote]:
    """Compliance checks then price checks: every check on every item."""
    return apply_prices(check_email(email, extracted, ctx), ctx)


def merge_reply(original: ExtractedQuote, reply: ExtractedQuote) -> ExtractedQuote:
    """Fill the original quote's empty fields from a supplier reply, item by item (matched on product)."""
    merged: list[QuoteItem] = []
    by_product = {i.product_id: i for i in reply.items if i.product_id}
    for item in original.items:
        update = by_product.pop(item.product_id, None) if item.product_id else None
        if update is None:
            merged.append(item)
            continue
        data = item.model_dump()
        for key, value in update.model_dump().items():
            if value is not None and key != "product_text":
                data[key] = value
        merged.append(QuoteItem.model_validate(data))
    merged.extend(by_product.values())
    return ExtractedQuote(
        supplier_name=original.supplier_name or reply.supplier_name,
        items=merged,
        unreadable_reason=None if merged else reply.unreadable_reason,
    )
