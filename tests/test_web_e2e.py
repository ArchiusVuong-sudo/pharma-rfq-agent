"""Main flow through the web app in replay mode: run, review with an edit, reply, re-check, evals."""

from __future__ import annotations

import re
import time

from fastapi.testclient import TestClient

from rfq_agent.web.app import app

RFQ = "/rfq/RFQ-SYN-0107"


def wait_for_run(client: TestClient) -> str:
    for _ in range(200):
        html = client.get(f"{RFQ}/live").text
        if "hx-trigger" not in html:  # the live panel stops polling once the run is not running
            return html
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_main_flow() -> None:
    with TestClient(app) as client:
        for path in ("/", RFQ, "/reviews", "/outbox", "/runs", "/evals", "/policy"):
            page = client.get(path)
            assert page.status_code == 200, path
            assert "Synthetic data" in page.text and "Replay mode" in page.text
        assert "No run yet" in client.get(RFQ).text

        client.post(f"{RFQ}/run")
        html = wait_for_run(client)
        assert "Awarded" in html and "Needs review" in html and "Blocked" in html
        assert "Could not read a price" in html and "GMP certificate expired" in html
        assert "Far under reference price" in html

        reviews = client.get("/reviews").text
        review_id = re.search(r'id="(REV-[0-9a-f]+)"', reviews).group(1)  # type: ignore[union-attr]
        fields = dict(re.findall(r'name="(qty:[^"]+)" value="(\d+)"', reviews))
        brw = next(k for k in fields if ":S-BRW:" in k)
        kes_ke = next(k for k in fields if k.startswith("qty:B-KE-01:S-KES"))

        bad = {**fields, brw: "50000", kes_ke: "130000", "action": "edit", "note": "too low"}
        client.post(f"/reviews/{review_id}", data=bad)
        assert "under its MOQ of 100,000" in client.get("/reviews").text

        good = {**fields, brw: "100000", kes_ke: "80000", "action": "edit", "note": "split"}
        done = client.post(f"/reviews/{review_id}", data=good)
        assert "resumed from its checkpoint" in done.text
        assert "Approved by a person with changes: split" in done.text

        outbox = client.get("/outbox").text
        assert outbox.count("PO draft") >= 3 and "Drafted by the model, guard passed" in outbox
        message_id = re.search(r'action="/outbox/([^"]+)/reply"', outbox).group(1)  # type: ignore[union-attr]
        client.post(f"/outbox/{message_id}/reply")
        wait_for_run(client)
        assert "cold_chain" in client.get("/reviews").text

        runs = client.get("/runs").text
        assert "recheck" in runs and "award" in runs
        run_id = re.search(r'href="/runs/(RUN-[0-9a-f]+)"', runs).group(1)  # type: ignore[union-attr]
        assert "Trace waterfall" in client.get(f"/runs/{run_id}").text

        client.post("/evals/run", data={"price_outlier": ["off", "on"]})
        assert "GATE PASS" in client.get("/evals").text
        client.post("/evals/run", data={"price_outlier": "off"})
        evals = client.get("/evals").text
        assert "GATE FAIL" in evals and "Unsafe auto-approval" in evals

        assert client.post("/reset").status_code == 200
        assert "No run yet" in client.get(RFQ).text
