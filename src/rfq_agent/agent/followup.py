"""Supplier follow-ups: the model drafts, a guard checks, one redraft, else a fixed template."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rfq_agent.llm.base import LlmError, LlmProvider, LlmRequest
from rfq_agent.tracing import Span, Tracer

MAX_DRAFTS = 2


class FollowUpDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: str = Field(description="Short subject line that names the RFQ id")
    body: str = Field(description="Plain-text email body")


# What the guard looks for so every missing item is named in the email.
MISSING_KEYWORDS: dict[str, tuple[str, ...]] = {
    "price_unreadable": ("price",),
    "incoterm_missing": ("incoterm",),
    "coa_missing": ("certificate of analysis", "coa"),
    "storage_missing": ("2-8", "2 to 8", "2–8"),
    "shelf_life_missing": ("shelf life", "shelf-life"),
    "lead_time_missing": ("lead time", "lead-time"),
}

PROMISE_RE = re.compile(
    r"\b(award(ed|ing)?|you (have )?won|selected|will (place|issue) (an? |the )?(order|po)|"
    r"purchase order|guarantee[ds]?|confirm(ed)? (the|your) order|contract)\b",
    re.IGNORECASE,
)
AMOUNT_RE = re.compile(r"(\$|€|£|\b(usd|eur|gbp|inr)\b)\s*\d|\d\s*(usd|eur|gbp|inr)\b", re.IGNORECASE)

FOLLOW_UP_SYSTEM = """You write short follow-up emails to pharmaceutical suppliers on behalf of a procurement team.
Ask only for the missing items you are given, and name each one. Be polite and plain.
Never mention prices or amounts, never promise or hint at an award, an order or a contract.
Under 100 words. Sign off as "Procurement team". Return only the structured output."""


@dataclass
class GuardResult:
    ok: bool
    problems: list[str] = field(default_factory=list)


def guard(draft: FollowUpDraft, missing_checks: list[str], max_words: int) -> GuardResult:
    text = f"{draft.subject}\n{draft.body}".lower()
    problems: list[str] = []
    for check in missing_checks:
        words = MISSING_KEYWORDS.get(check, ())
        if words and not any(w in text for w in words):
            problems.append(f"does not ask for {check.replace('_', ' ')}")
    if PROMISE_RE.search(text):
        problems.append("contains an award or order promise")
    if AMOUNT_RE.search(text):
        problems.append("mentions a price or amount")
    count = len(draft.body.split())
    if count > max_words:
        problems.append(f"{count} words, over the {max_words} word limit")
    return GuardResult(ok=not problems, problems=problems)


def template(contact: str, rfq_id: str, product: str, missing: list[str]) -> FollowUpDraft:
    items = "\n".join(f"- {m}" for m in missing)
    return FollowUpDraft(
        subject=f"{rfq_id}: information needed for your {product} quote",
        body=(
            f"Dear {contact},\n\nThank you for your quote on {rfq_id} for {product}. "
            f"Before we can evaluate it, please send:\n{items}\n\nKind regards,\nProcurement team"
        ),
    )


def follow_up_request(
    contact: str, supplier: str, rfq_id: str, product: str, missing: list[str], feedback: list[str] | None = None
) -> LlmRequest:
    prompt = (
        f"RFQ: {rfq_id}\nSupplier: {supplier}\nContact: {contact}\nProduct quoted: {product}\n"
        "Missing items to ask for:\n" + "\n".join(f"- {m}" for m in missing)
    )
    if feedback:
        prompt += "\n\nYour previous draft was rejected: " + "; ".join(feedback) + ". Write a corrected draft."
    return LlmRequest(
        task="draft_follow_up",
        system=FOLLOW_UP_SYSTEM,
        prompt=prompt,
        output_schema=FollowUpDraft.model_json_schema(),
        tools=[],
        max_turns=2,
    )


@dataclass
class FollowUpResult:
    draft: FollowUpDraft
    drafted_by: str  # "model" or "template"
    notes: list[str]


async def draft_follow_up(
    *,
    contact: str,
    supplier: str,
    rfq_id: str,
    product: str,
    missing_checks: list[str],
    missing_items: list[str],
    provider: LlmProvider,
    max_words: int,
    tracer: Tracer | None = None,
    parent: Span | None = None,
) -> FollowUpResult:
    notes: list[str] = []
    feedback: list[str] | None = None
    for attempt in range(1, MAX_DRAFTS + 1):
        request = follow_up_request(contact, supplier, rfq_id, product, missing_items, feedback)
        draft: FollowUpDraft | None = None
        try:
            if tracer:
                async with tracer.span("llm", f"draft follow-up to {supplier}", parent, task=request.task) as span:
                    response = await provider.complete(request)
                    tracer.llm_done(span, response, attempt)
                    draft = FollowUpDraft.model_validate(response.output or {})
                    result = guard(draft, missing_checks, max_words)
                    span.attrs["guard"] = "pass" if result.ok else "; ".join(result.problems)
                    if not result.ok:
                        span.status = "retry"
            else:
                response = await provider.complete(request)
                draft = FollowUpDraft.model_validate(response.output or {})
                result = guard(draft, missing_checks, max_words)
        except ValidationError:
            result = GuardResult(ok=False, problems=["output failed the schema check"])
        except LlmError as exc:  # the model being down must not stop a follow-up: fall back to the template
            notes.append(f"model unavailable ({type(exc).__name__}: {str(exc)[:120]})")
            break
        if draft is not None and result.ok:
            notes.append(f"guard passed on draft {attempt}")
            return FollowUpResult(draft=draft, drafted_by="model", notes=notes)
        notes.append(f"draft {attempt} rejected: {'; '.join(result.problems)}")
        feedback = result.problems
    notes.append("sent the fixed template instead")
    return FollowUpResult(draft=template(contact, rfq_id, product, missing_items), drafted_by="template", notes=notes)
