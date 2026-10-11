"""Re-record every model response the app and the evals use, from real live runs.

    uv run python scripts/record_fixtures.py

Needs the Claude Code login (or ANTHROPIC_API_KEY). It clears fixtures/llm, records the 18 eval
cases, then walks the demo flow (award run, reviewer edit, simulated supplier reply, re-check) so
every request the replay mode will make has a recorded answer. Run it after changing a prompt,
a schema description, a seed email or a check's wording.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from rfq_agent.agent.service import AgentService
from rfq_agent.evals.runner import run_evals
from rfq_agent.llm.live import LiveProvider, live_available
from rfq_agent.llm.replay import FIXTURE_DIR, RecordingProvider
from rfq_agent.policy import load_policy
from rfq_agent.store import Store


async def main() -> int:
    if not await live_available():
        print("No Claude Code login or ANTHROPIC_API_KEY found: cannot record.", file=sys.stderr)
        return 2
    shutil.rmtree(FIXTURE_DIR, ignore_errors=True)
    FIXTURE_DIR.mkdir(parents=True)
    recorder = RecordingProvider(LiveProvider())

    report = await run_evals(recorder, load_policy())
    print(f"evals: {len(report.cases)} cases, gate {'PASS' if report.passed else 'FAIL'}")

    with tempfile.TemporaryDirectory(prefix="rfq-record-") as tmp:
        store = Store(Path(tmp) / "app.db")
        async with AsyncSqliteSaver.from_conn_string(str(Path(tmp) / "checkpoints.db")) as saver:
            service = AgentService(store, recorder, saver)
            run_id = service.start_award_run()
            await service.wait(run_id)
            for review in store.reviews("pending"):
                await service.resume_review(review["id"], {"action": "approve", "note": "recording"})
            for msg in store.outbox():
                if msg["reply_email_id"] and not msg["reply_simulated"]:
                    recheck = service.simulate_reply(msg["id"])
                    await service.wait(recheck)
            failed = store.q("SELECT id, error FROM runs WHERE status = 'failed'")
        store.close()
    if failed:
        print(f"failed runs: {failed}", file=sys.stderr)
        return 1
    print(f"recorded {len(set(recorder.written))} fixtures into {FIXTURE_DIR}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
