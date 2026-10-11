"""Domain model: catalog, buyers, suppliers, the pooled RFQ, quotes, checks, plans and decisions.

Everything the agent reads or writes is one of these Pydantic models. Code (not the model)
decides money and compliance, so the LLM only ever produces an ``ExtractedQuote``.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Country = Literal["UG", "KE", "MW"]
Currency = Literal["USD", "EUR", "GBP", "INR"]
Incoterm = Literal["EXW", "FCA", "FOB", "CIF", "CIP", "DAP", "DDP"]


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------- reference data


class Product(Frozen):
    id: str
    inn: str
    strength: str
    form: str
    unit: str  # what one priced unit is: tablet, ampoule, sachet, vial
    pack_sizes: list[int]
    cold_chain: bool
    synonyms: list[str]
    reference_unit_price_usd: float  # synthetic reference, demo assumption
    total_shelf_life_months: int

    @property
    def name(self) -> str:
        return f"{self.inn} {self.strength} {self.form}"


class Buyer(Frozen):
    id: str
    name: str
    kind: str
    country: Country
    country_name: str


class GmpCertificate(Frozen):
    number: str
    expires_on: date


class Supplier(Frozen):
    id: str
    name: str
    kind: Literal["manufacturer", "distributor", "trader"]
    in_registry: bool
    completed_orders: int
    on_time_rate: float
    audit_score: int
    gmp: GmpCertificate | None
    registrations: dict[str, list[Country]] = Field(default_factory=dict)  # product id -> countries
    contact_name: str
    contact_email_alias: str  # shown as a handle, never a real address


class Demand(Frozen):
    buyer_id: str
    quantity: int
    needed_by: date


class RfqLine(Frozen):
    id: str
    product_id: str
    demands: list[Demand]

    @property
    def total_quantity(self) -> int:
        return sum(d.quantity for d in self.demands)


class QuoteEmail(Frozen):
    id: str
    supplier_id: str
    received_at: datetime
    subject: str
    body: str
    in_reply_to: str | None = None  # outbox message id when this is a supplier reply


class Rfq(Frozen):
    id: str
    title: str
    issued_on: date
    closes_on: date
    lines: list[RfqLine]
    emails: list[QuoteEmail]


# ---------------------------------------------------------------- what the model extracts


class QuoteItem(BaseModel):
    """One product line in a supplier email, as written. Unknown means null, never a guess."""

    model_config = ConfigDict(extra="forbid")

    product_text: str = Field(description="The product exactly as the email names it")
    product_id: str | None = Field(description="Catalog id from search_catalog, null if no match")
    price_amount: float | None = Field(description="Price number as written, null if not stated")
    currency: Currency | None = Field(description="Currency of the price, null if not stated")
    price_per_quantity: int | None = Field(
        description="How many units (tablets, ampoules) the price covers, e.g. 100 for a box of 100"
    )
    incoterm: Incoterm | None = Field(description="Incoterm as written, null if not stated")
    moq_units: int | None = Field(description="Minimum order in units, null if not stated")
    max_units: int | None = Field(description="Most units the supplier can supply, null if not stated")
    lead_time_days: int | None = Field(description="Lead time in days, null if not stated")
    shelf_life_months: int | None = Field(
        description="Remaining shelf life at dispatch in months, null if not stated"
    )
    coa_provided: bool | None = Field(
        description="True if a certificate of analysis is attached or included, false if said missing, null if not mentioned"
    )
    storage_condition: str | None = Field(description="Storage condition as written, e.g. '2-8 C'")
    gmp_certificate_number: str | None = Field(description="GMP certificate number as written")


class ExtractedQuote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supplier_name: str | None = Field(description="Supplier company name as signed in the email")
    items: list[QuoteItem]
    unreadable_reason: str | None = Field(
        description="Set when the email has no readable quote (e.g. points to a missing attachment)"
    )


# ---------------------------------------------------------------- what code derives


class Severity(StrEnum):
    BLOCK = "block"  # quote cannot be used
    HOLD = "hold"  # quote cannot be used until the supplier sends something: follow-up
    REVIEW = "review"  # quote can be used, but the line needs a person
    INFO = "info"


class Flag(Frozen):
    check: str  # stable id of the check that raised it
    severity: Severity
    label: str  # short chip text
    detail: str  # one line for the tooltip


class CheckedQuote(BaseModel):
    """A quote item after every check ran on it, with prices normalised to USD per unit."""

    ref: str  # email id + item index
    email_id: str
    supplier_id: str
    line_id: str | None
    item: QuoteItem
    unit_price_usd: float | None = None
    landed_unit_usd: float | None = None
    flags: list[Flag] = Field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.line_id is not None and not any(
            f.severity in (Severity.BLOCK, Severity.HOLD) for f in self.flags
        )

    @property
    def missing_items(self) -> list[str]:
        return [f.detail for f in self.flags if f.severity is Severity.HOLD]


class Allocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    buyer_id: str
    supplier_id: str
    quote_ref: str
    quantity: int = Field(ge=0)
    landed_unit_usd: float = 0.0

    @property
    def total_usd(self) -> float:
        return round(self.quantity * self.landed_unit_usd, 2)


class LinePlan(BaseModel):
    line_id: str
    allocations: list[Allocation]
    unmet: dict[str, int] = Field(default_factory=dict)  # buyer id -> units with no supplier

    @property
    def total_usd(self) -> float:
        return round(sum(a.total_usd for a in self.allocations), 2)

    @property
    def covered_units(self) -> int:
        return sum(a.quantity for a in self.allocations)


Route = Literal["act_alone", "review", "stop"]


class RuleHit(Frozen):
    rule: str
    route: Route
    reason: str


class LineDecision(BaseModel):
    line_id: str
    route: Route
    rules_fired: list[RuleHit]


class OutboxKind(StrEnum):
    PO_DRAFT = "po_draft"
    AWARD = "award_notice"
    REGRET = "regret_notice"
    FOLLOW_UP = "follow_up"


class OutboxMessage(BaseModel):
    id: str
    run_id: str
    rfq_id: str
    line_id: str | None
    kind: OutboxKind
    to_supplier_id: str
    subject: str
    body: str
    created_at: datetime
    drafted_by: Literal["model", "template", "code"]
    guard_notes: list[str] = Field(default_factory=list)
    reply_email_id: str | None = None  # recorded reply that can be simulated
