"""Product catalog retrieval used by the search_catalog tool: fuzzy match over names and synonyms."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from rfq_agent.models import Product
from rfq_agent.seed import CATALOG


class CatalogQuery(BaseModel):
    query: str = Field(min_length=2, max_length=200)
    limit: int = Field(default=3, ge=1, le=5)


class CatalogHit(BaseModel):
    product_id: str
    name: str
    unit: str
    pack_sizes: list[int]
    cold_chain: bool
    score: float


def _norm(text: str) -> str:
    text = text.lower().replace("sulphate", "sulfate").replace("amoxycillin", "amoxicillin")
    text = re.sub(r"(\d)\s*(mg|mcg|iu|ml)\b", r"\1 \2", text)
    return re.sub(r"[^a-z0-9/%. ]+", " ", text).strip()


def _strength_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def search(q: CatalogQuery, catalog: list[Product] = CATALOG) -> list[CatalogHit]:
    """Best catalog matches for free text. A strength mismatch (250 vs 500 mg) costs points."""
    needle = _norm(q.query)
    numbers = _strength_numbers(needle)
    hits: list[CatalogHit] = []
    for p in catalog:
        names = [p.name, f"{p.inn} {p.strength}", *p.synonyms]
        best = max(fuzz.token_set_ratio(needle, _norm(n)) for n in names)
        product_numbers = _strength_numbers(p.strength)
        if numbers and product_numbers and not (numbers & product_numbers):
            best -= 25
        hits.append(
            CatalogHit(
                product_id=p.id,
                name=p.name,
                unit=p.unit,
                pack_sizes=p.pack_sizes,
                cold_chain=p.cold_chain,
                score=round(best, 1),
            )
        )
    hits.sort(key=lambda h: h.score, reverse=True)
    return [h for h in hits[: q.limit] if h.score >= 45]
