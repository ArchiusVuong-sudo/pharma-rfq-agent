from __future__ import annotations

from rfq_agent.evals.__main__ import main
from rfq_agent.evals.runner import load_cases, run_evals, score_fields
from rfq_agent.llm.replay import ReplayProvider
from rfq_agent.policy import Policy
from tests.conftest import item


def test_cases_cover_every_rule_and_the_demo_emails() -> None:
    cases = load_cases()
    assert len(cases) == 18
    assert {c.email_id for c in cases if c.email_id} == {"E1", "E2", "E3", "E4", "E5", "E6"}
    only_outlier = [c for c in cases if c.rule == "price_outlier only"]
    assert len(only_outlier) >= 2
    assert {c.expected.route for c in cases} == {"act_alone", "review", "stop"}


def test_field_scoring_tolerates_rounding_but_not_nulls() -> None:
    hits, misses = score_fields({"price_amount": 2.40, "coa_provided": True, "incoterm": "fca"}, item())
    assert hits == 3 and not misses
    hits, misses = score_fields({"shelf_life_months": None}, item())
    assert hits == 0 and misses


async def test_gate_passes_with_all_guardrails(policy: Policy) -> None:
    report = await run_evals(ReplayProvider(delay_scale=0), policy)
    assert report.passed and report.unsafe_auto_approvals == 0 and report.errors == 0
    assert report.routing_accuracy == 1.0 and report.flag_recall == 1.0


async def test_gate_fails_without_the_price_outlier_guardrail(policy: Policy) -> None:
    report = await run_evals(ReplayProvider(delay_scale=0), policy.without("price_outlier"))
    assert not report.passed and report.unsafe_auto_approvals >= 2
    assert {c.id for c in report.cases if c.unsafe} >= {"C15", "C16"}


def test_cli_exit_codes() -> None:
    assert main(["--gate"]) == 0
    assert main(["--gate", "--disable", "price_outlier"]) == 1
