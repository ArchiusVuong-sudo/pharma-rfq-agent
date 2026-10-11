"""python -m rfq_agent.evals [--gate] [--mode replay|live] [--record] [--disable RULE]

Replay mode (default) needs no login or network. --gate exits 1 when the gate fails.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from rfq_agent.evals.runner import EvalReport, run_evals
from rfq_agent.llm.base import LlmProvider
from rfq_agent.llm.live import LiveProvider
from rfq_agent.llm.replay import RecordingProvider, ReplayProvider
from rfq_agent.policy import load_policy


def print_report(report: EvalReport) -> None:
    print(f"Eval suite ({report.mode}), guardrails off: {', '.join(report.disabled_rules) or 'none'}")
    print(f"{'case':5} {'expected':10} {'got':10} {'fields':7} {'flags':6} note")
    for r in report.cases:
        flags = f"{len(set(r.flags_expected) & set(r.flags_found))}/{len(r.flags_expected)}"
        note = "UNSAFE auto-approval" if r.unsafe else (r.error or "; ".join(r.field_misses[:2]))
        print(f"{r.id:5} {r.expected_route:10} {r.route or '-':10} {r.field_hits}/{r.field_total:<5} {flags:6} {note}")
    print(
        f"\nfield accuracy {report.field_accuracy:.1%}  flag recall {report.flag_recall:.1%}  "
        f"routing accuracy {report.routing_accuracy:.1%}  unsafe auto-approvals {report.unsafe_auto_approvals}  "
        f"errors {report.errors}"
    )
    print(f"GATE {'PASS' if report.passed else 'FAIL'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rfq_agent.evals")
    parser.add_argument("--mode", choices=["replay", "live"], default="replay")
    parser.add_argument("--record", action="store_true", help="live mode: re-record the model outputs")
    parser.add_argument("--gate", action="store_true", help="exit 1 when the gate fails")
    parser.add_argument("--disable", action="append", default=[], help="switch a guardrail off, e.g. price_outlier")
    args = parser.parse_args(argv)

    provider: LlmProvider = ReplayProvider(delay_scale=0)
    if args.mode == "live":
        provider = RecordingProvider(LiveProvider()) if args.record else LiveProvider()
    policy = load_policy().without(*args.disable) if args.disable else load_policy()
    report = asyncio.run(run_evals(provider, policy))
    print_report(report)
    return 1 if args.gate and not report.passed else 0


if __name__ == "__main__":
    sys.exit(main())
