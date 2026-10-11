from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from rfq_agent.agent.followup import FollowUpDraft, draft_follow_up, guard, template
from rfq_agent.agent.registry import shortlist
from rfq_agent.llm.base import LlmError, LlmRequest, LlmResponse
from rfq_agent.models import Demand, RfqLine
from rfq_agent.policy import load_policy
from rfq_agent.seed import SUPPLIERS, buyers_by_id

MISSING = ["coa_missing", "storage_missing", "shelf_life_missing"]
GOOD = FollowUpDraft(
    subject="RFQ-SYN-0107: documents for your oxytocin quote",
    body="Dear Grace, please send the certificate of analysis (CoA), confirmation of 2-8 C storage and transport, "
    "and the remaining shelf life at dispatch. Kind regards, Procurement team",
)


def test_guard_passes_a_complete_polite_draft() -> None:
    assert guard(GOOD, MISSING, 120).ok


def test_guard_needs_every_missing_item_named() -> None:
    draft = GOOD.model_copy(update={"body": "Dear Grace, please send the CoA. Thanks"})
    result = guard(draft, MISSING, 120)
    assert not result.ok and len(result.problems) == 2


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        (" We will award you the line.", "award or order promise"),
        (" We will issue a purchase order next week.", "award or order promise"),
        (" Your price of USD 0.27 is fine.", "price or amount"),
        (" Your $0.27 is fine.", "price or amount"),
    ],
)
def test_guard_blocks_promises_and_amounts(text: str, problem: str) -> None:
    draft = GOOD.model_copy(update={"body": GOOD.body + text})
    assert any(problem in p for p in guard(draft, MISSING, 120).problems)


def test_guard_enforces_word_limit() -> None:
    draft = GOOD.model_copy(update={"body": GOOD.body + " please" * 120})
    assert any("word limit" in p for p in guard(draft, MISSING, 120).problems)


class Scripted:
    """A provider that returns scripted outputs in order (or raises)."""

    mode = "replay"
    model = "scripted"

    def __init__(self, *outputs: object) -> None:
        self.outputs = list(outputs)
        self.requests: list[LlmRequest] = []

    async def complete(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        assert isinstance(out, dict)
        return LlmResponse(output=out, model=self.model)


async def run_draft(provider: Scripted) -> tuple[str, list[str], FollowUpDraft]:
    items = ["a certificate of analysis (CoA)", "confirmation of 2-8 C storage", "the remaining shelf life"]
    result = await draft_follow_up(
        contact="Grace",
        supplier="Halden",
        rfq_id="RFQ-1",
        product="Oxytocin",
        missing_checks=MISSING,
        missing_items=items,
        provider=provider,
        max_words=120,
    )
    return result.drafted_by, result.notes, result.draft


async def test_one_redraft_with_the_guard_feedback() -> None:
    bad = {"subject": "x", "body": "We will award you. USD 1 each."}
    provider = Scripted(bad, GOOD.model_dump())
    by, _notes, draft = await run_draft(provider)
    assert by == "model" and draft == GOOD
    assert "previous draft was rejected" in provider.requests[1].prompt


async def test_template_after_two_bad_drafts() -> None:
    bad = {"subject": "x", "body": "We will award you."}
    by, notes, draft = await run_draft(Scripted(bad, bad))
    assert by == "template" and guard(draft, MISSING, 120).ok
    assert notes[-1] == "sent the fixed template instead"


async def test_template_when_the_model_is_down() -> None:
    by, notes, _ = await run_draft(Scripted(LlmError("overloaded", retryable=True)))
    assert by == "template" and "model unavailable" in notes[0]


def test_template_names_every_item() -> None:
    draft = template("Grace", "RFQ-1", "Oxytocin", ["a certificate of analysis (CoA)", "the remaining shelf life"])
    assert "certificate of analysis" in draft.body and "shelf life" in draft.body


LINE = RfqLine(
    id="L3", product_id="P-OXY-10-AMP", demands=[Demand(buyer_id="B-UG-01", quantity=1, needed_by=date(2026, 12, 1))]
)


async def test_shortlist_keeps_registry_suppliers_registered_in_buyer_countries() -> None:
    entries, errors = await shortlist(LINE, SUPPLIERS, buyers_by_id())
    assert {e.supplier_id for e in entries} == {"S-CRV", "S-HAL"} and errors == []


async def test_injected_registry_timeout_is_retried_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAULT_ONCE", "registry_timeout")
    entries, errors = await shortlist(LINE, SUPPLIERS, buyers_by_id())
    assert len(errors) == 1 and "timed out" in errors[0] and entries
    _, again = await shortlist(LINE, SUPPLIERS, buyers_by_id())
    assert again == []  # the fault fires once per process


def test_policy_file_is_validated(tmp_path: Path) -> None:
    policy = load_policy()
    assert policy.rule("value_limit") is not None and policy.rule("value_limit").limit_usd == 20000  # type: ignore[union-attr]
    bad = tmp_path / "policy.yaml"
    bad.write_text("checks: {}\n")
    with pytest.raises(ValidationError):
        load_policy.__wrapped__(bad)
