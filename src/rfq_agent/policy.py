"""Loads policy.yaml into a typed Policy and decides the route for each line."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from rfq_agent.models import Route

POLICY_PATH = Path(__file__).parent / "policy.yaml"

ROUTE_ORDER: dict[Route, int] = {"act_alone": 0, "review": 1, "stop": 2}


class CheckSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gmp_expiry_warn_days: int
    shelf_life_min_share_on_arrival: float
    price_far_under_share: float
    price_far_over_share: float
    new_supplier_min_orders: int


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    route: Route
    description: str
    limit_usd: float | None = None


class FollowUpSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    route: Route
    max_words: int


class MonitoringSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_cost_budget_usd: float
    run_latency_budget_s: float
    review_sla_minutes: int


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checks: CheckSettings
    fx_to_usd: dict[str, float]
    incoterm_uplift: dict[str, float]
    rules: list[Rule]
    follow_up: FollowUpSettings
    monitoring: MonitoringSettings
    disabled_rules: frozenset[str] = frozenset()

    def rule(self, rule_id: str) -> Rule | None:
        """The rule, or None when it is missing or switched off."""
        if rule_id in self.disabled_rules:
            return None
        return next((r for r in self.rules if r.id == rule_id), None)

    def without(self, *rule_ids: str) -> Policy:
        """A copy with guardrails switched off (used by the eval toggle)."""
        unknown = set(rule_ids) - {r.id for r in self.rules}
        if unknown:
            raise ValueError(f"unknown rules: {sorted(unknown)}")
        return self.model_copy(update={"disabled_rules": self.disabled_rules | set(rule_ids)})


@cache
def load_policy(path: Path = POLICY_PATH) -> Policy:
    return Policy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def policy_text(path: Path = POLICY_PATH) -> str:
    return path.read_text(encoding="utf-8")
