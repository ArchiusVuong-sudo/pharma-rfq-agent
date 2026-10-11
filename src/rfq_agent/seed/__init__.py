"""Synthetic demo data. Every name, number and certificate here is invented for this demo.

Dates are set relative to ``AS_OF`` so the scenario is the same on any day the app runs.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import cache
from pathlib import Path

from rfq_agent.models import (
    Buyer,
    Demand,
    GmpCertificate,
    Product,
    QuoteEmail,
    Rfq,
    RfqLine,
    Supplier,
)

AS_OF = date(2026, 10, 11)  # seed date: every check is scored at this date
EMAIL_DIR = Path(__file__).parent / "emails"


def _d(days: int) -> date:
    return AS_OF + timedelta(days=days)


def _t(days: int, hour: int) -> datetime:
    return datetime.combine(_d(days), datetime.min.time()).replace(hour=hour)


CATALOG: list[Product] = [
    Product(
        id="P-ZINC-20-DT",
        inn="Zinc sulfate",
        strength="20 mg",
        form="dispersible tablet",
        unit="tablet",
        pack_sizes=[100, 1000],
        cold_chain=False,
        synonyms=["zinc 20mg DT", "zinc sulphate dispersible", "zinc dispersible tabs"],
        reference_unit_price_usd=0.025,
        total_shelf_life_months=36,
    ),
    Product(
        id="P-AMOX-250-DT",
        inn="Amoxicillin",
        strength="250 mg",
        form="dispersible tablet",
        unit="tablet",
        pack_sizes=[100, 1000],
        cold_chain=False,
        synonyms=["amoxycillin 250 DT", "amox 250 dispersible", "amoxicillin trihydrate 250mg"],
        reference_unit_price_usd=0.045,
        total_shelf_life_months=24,
    ),
    Product(
        id="P-OXY-10-AMP",
        inn="Oxytocin",
        strength="10 IU/mL",
        form="1 mL ampoule",
        unit="ampoule",
        pack_sizes=[10, 50],
        cold_chain=True,
        synonyms=["oxytocin injection 10 IU", "oxytocin 10iu/ml amp", "oxytocin 10 units"],
        reference_unit_price_usd=0.30,
        total_shelf_life_months=36,
    ),
    Product(
        id="P-ORS-1L-SACHET",
        inn="Oral rehydration salts",
        strength="1 L",
        form="sachet",
        unit="sachet",
        pack_sizes=[50, 100],
        cold_chain=False,
        synonyms=["ORS sachets", "low osmolarity ORS", "oral rehydration sachet"],
        reference_unit_price_usd=0.08,
        total_shelf_life_months=36,
    ),
    Product(
        id="P-PARA-500-TAB",
        inn="Paracetamol",
        strength="500 mg",
        form="tablet",
        unit="tablet",
        pack_sizes=[100, 1000],
        cold_chain=False,
        synonyms=["acetaminophen 500mg", "paracetamol 500 tabs", "PCM 500"],
        reference_unit_price_usd=0.008,
        total_shelf_life_months=48,
    ),
    Product(
        id="P-MISO-200-TAB",
        inn="Misoprostol",
        strength="200 mcg",
        form="tablet",
        unit="tablet",
        pack_sizes=[28, 100],
        cold_chain=False,
        synonyms=["misoprostol 200 microgram", "miso 200mcg"],
        reference_unit_price_usd=0.12,
        total_shelf_life_months=24,
    ),
    Product(
        id="P-MGSO4-50-AMP",
        inn="Magnesium sulfate",
        strength="50% 10 mL",
        form="ampoule",
        unit="ampoule",
        pack_sizes=[10, 50],
        cold_chain=False,
        synonyms=["magnesium sulphate injection 50%", "MgSO4 5g/10ml"],
        reference_unit_price_usd=0.40,
        total_shelf_life_months=36,
    ),
    Product(
        id="P-AL-20-120-TAB",
        inn="Artemether/lumefantrine",
        strength="20/120 mg",
        form="tablet",
        unit="tablet",
        pack_sizes=[24, 360],
        cold_chain=False,
        synonyms=["AL 20/120", "artemether lumefantrine tabs", "ACT 20/120"],
        reference_unit_price_usd=0.035,
        total_shelf_life_months=24,
    ),
]

BUYERS: list[Buyer] = [
    Buyer(
        id="B-UG-01",
        name="Rwenzo District Hospital",
        kind="District hospital",
        country="UG",
        country_name="Uganda",
    ),
    Buyer(
        id="B-KE-01",
        name="Kivuli Clinic Network",
        kind="Faith-based clinic network",
        country="KE",
        country_name="Kenya",
    ),
    Buyer(
        id="B-MW-01",
        name="Bright Start Maternal Program",
        kind="NGO maternal health program",
        country="MW",
        country_name="Malawi",
    ),
]

ALL: list[str] = ["UG", "KE", "MW"]

SUPPLIERS: list[Supplier] = [
    Supplier(
        id="S-KES",
        name="Kestrel Generics",
        kind="manufacturer",
        in_registry=True,
        completed_orders=42,
        on_time_rate=0.95,
        audit_score=88,
        gmp=GmpCertificate(number="SYN-GMP-4471", expires_on=_d(410)),
        registrations={
            "P-ZINC-20-DT": ["UG", "KE", "MW"],
            "P-AMOX-250-DT": ["UG", "KE", "MW"],
            "P-PARA-500-TAB": ["UG", "KE", "MW"],
        },
        contact_name="Ravi Menon",
        contact_email_alias="sales desk",
    ),
    Supplier(
        id="S-BRW",
        name="Brightwater Pharma",
        kind="manufacturer",
        in_registry=True,
        completed_orders=0,
        on_time_rate=0.0,
        audit_score=79,
        gmp=GmpCertificate(number="SYN-GMP-9013", expires_on=_d(41)),
        registrations={"P-AMOX-250-DT": ["KE"]},
        contact_name="Ilse Moreau",
        contact_email_alias="export team",
    ),
    Supplier(
        id="S-CRV",
        name="Corvane Biologics",
        kind="manufacturer",
        in_registry=True,
        completed_orders=17,
        on_time_rate=0.9,
        audit_score=84,
        gmp=GmpCertificate(number="SYN-GMP-2208", expires_on=_d(-12)),
        registrations={"P-OXY-10-AMP": ["UG", "KE", "MW"], "P-MGSO4-50-AMP": ["UG", "MW"]},
        contact_name="Tomas Ferreira",
        contact_email_alias="tenders",
    ),
    Supplier(
        id="S-HAL",
        name="Halden Medical Supply",
        kind="distributor",
        in_registry=True,
        completed_orders=14,
        on_time_rate=0.93,
        audit_score=86,
        gmp=GmpCertificate(number="SYN-GDP-5530", expires_on=_d(300)),
        registrations={
            "P-OXY-10-AMP": ["UG", "KE", "MW"],
            "P-ORS-1L-SACHET": ["UG", "KE", "MW"],
            "P-MISO-200-TAB": ["UG", "MW"],
        },
        contact_name="Grace Atieno",
        contact_email_alias="quotes",
    ),
    Supplier(
        id="S-TAM",
        name="Tamsin Trading",
        kind="trader",
        in_registry=False,
        completed_orders=0,
        on_time_rate=0.0,
        audit_score=0,
        gmp=None,
        registrations={},
        contact_name="Samir Haddad",
        contact_email_alias="info",
    ),
    Supplier(
        id="S-ORR",
        name="Orrin Distribution",
        kind="distributor",
        in_registry=True,
        completed_orders=23,
        on_time_rate=0.91,
        audit_score=82,
        gmp=GmpCertificate(number="SYN-GDP-7764", expires_on=_d(520)),
        registrations={"P-ZINC-20-DT": ["UG", "KE", "MW"], "P-AL-20-120-TAB": ["UG", "KE", "MW"]},
        contact_name="Nadia Lindqvist",
        contact_email_alias="bids",
    ),
    # The two suppliers below appear only in the labelled eval cases, not on the demo RFQ.
    Supplier(
        id="S-VEL",
        name="Velmora Labs",
        kind="manufacturer",
        in_registry=True,
        completed_orders=30,
        on_time_rate=0.92,
        audit_score=85,
        gmp=GmpCertificate(number="SYN-GMP-3150", expires_on=_d(25)),
        registrations={"P-ORS-1L-SACHET": ["UG", "KE", "MW"], "P-PARA-500-TAB": ["UG", "KE", "MW"]},
        contact_name="Jonas Albrecht",
        contact_email_alias="sales",
    ),
    Supplier(
        id="S-PIN",
        name="Pinecrest Pharma",
        kind="manufacturer",
        in_registry=True,
        completed_orders=19,
        on_time_rate=0.94,
        audit_score=90,
        gmp=GmpCertificate(number="SYN-GMP-6402", expires_on=_d(380)),
        registrations={
            "P-MISO-200-TAB": ["UG", "MW"],
            "P-MGSO4-50-AMP": ["UG", "KE", "MW"],
            "P-AL-20-120-TAB": ["UG", "KE", "MW"],
        },
        contact_name="Mei Tanaka",
        contact_email_alias="tenders",
    ),
]

DEMO_SUPPLIER_IDS = ("S-KES", "S-BRW", "S-CRV", "S-HAL", "S-TAM", "S-ORR")


def _email(id: str, supplier_id: str, day: int, hour: int, subject: str) -> QuoteEmail:
    body = (EMAIL_DIR / f"{id}.txt").read_text(encoding="utf-8").strip()
    return QuoteEmail(
        id=id, supplier_id=supplier_id, received_at=_t(day, hour), subject=subject, body=body
    )


@cache
def demo_rfq() -> Rfq:
    needed = _d(65)
    return Rfq(
        id="RFQ-SYN-0107",
        title="Pooled Q4 order: child health and maternal care",
        issued_on=_d(-9),
        closes_on=_d(-1),
        lines=[
            RfqLine(
                id="L1",
                product_id="P-ZINC-20-DT",
                demands=[
                    Demand(buyer_id="B-UG-01", quantity=60_000, needed_by=needed),
                    Demand(buyer_id="B-KE-01", quantity=40_000, needed_by=needed),
                    Demand(buyer_id="B-MW-01", quantity=30_000, needed_by=needed),
                ],
            ),
            RfqLine(
                id="L2",
                product_id="P-AMOX-250-DT",
                demands=[
                    Demand(buyer_id="B-UG-01", quantity=250_000, needed_by=needed),
                    Demand(buyer_id="B-KE-01", quantity=180_000, needed_by=needed),
                    Demand(buyer_id="B-MW-01", quantity=150_000, needed_by=needed),
                ],
            ),
            RfqLine(
                id="L3",
                product_id="P-OXY-10-AMP",
                demands=[
                    Demand(buyer_id="B-UG-01", quantity=20_000, needed_by=needed),
                    Demand(buyer_id="B-KE-01", quantity=15_000, needed_by=needed),
                    Demand(buyer_id="B-MW-01", quantity=25_000, needed_by=needed),
                ],
            ),
        ],
        emails=[
            _email("E1", "S-KES", -6, 9, "RE: RFQ-SYN-0107 quotation - zinc and amoxicillin"),
            _email("E2", "S-BRW", -5, 14, "Offer amoxicillin 250 DT / RFQ-SYN-0107"),
            _email("E3", "S-CRV", -4, 11, "Oxytocin 10 IU tender response"),
            _email("E4", "S-HAL", -3, 16, "Quote: oxytocin ampoules"),
            _email("E5", "S-TAM", -2, 8, "BEST PRICE amoxicillin!!"),
            _email("E6", "S-ORR", -2, 17, "Fwd: zinc offer"),
        ],
    )


@cache
def recorded_replies() -> dict[str, QuoteEmail]:
    """Recorded supplier replies, keyed by supplier id. Loaded when a reply is simulated."""
    return {
        "S-HAL": _email("R4", "S-HAL", 0, 10, "RE: Oxytocin ampoules - documents"),
    }


def catalog_by_id() -> dict[str, Product]:
    return {p.id: p for p in CATALOG}


def suppliers_by_id() -> dict[str, Supplier]:
    return {s.id: s for s in SUPPLIERS}


def buyers_by_id() -> dict[str, Buyer]:
    return {b.id: b for b in BUYERS}
