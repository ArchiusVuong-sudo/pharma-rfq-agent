"""SQLite app store: runs, timeline events, spans, alerts, quotes, line state, reviews, outbox, evals.

LangGraph checkpoints live in a second file (checkpoints.db) owned by the checkpointer.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, rfq_id TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL,
  mode TEXT NOT NULL, model TEXT, started_at TEXT NOT NULL, ended_at TEXT,
  active_ms REAL DEFAULT 0, cost_usd REAL DEFAULT 0, input_tokens INTEGER DEFAULT 0,
  output_tokens INTEGER DEFAULT 0, llm_calls INTEGER DEFAULT 0, error TEXT, parent_run_id TEXT,
  summary TEXT
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, ts TEXT NOT NULL,
  node TEXT NOT NULL, level TEXT NOT NULL, message TEXT NOT NULL, data TEXT
);
CREATE TABLE IF NOT EXISTS spans (
  id TEXT PRIMARY KEY, run_id TEXT NOT NULL, parent_id TEXT, kind TEXT NOT NULL, name TEXT NOT NULL,
  start_ms REAL NOT NULL, end_ms REAL, status TEXT NOT NULL, attrs TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts TEXT NOT NULL, kind TEXT NOT NULL,
  severity TEXT NOT NULL, message TEXT NOT NULL, dedupe_key TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS extractions (
  rfq_id TEXT NOT NULL, email_id TEXT NOT NULL, run_id TEXT NOT NULL, data TEXT NOT NULL,
  PRIMARY KEY (rfq_id, email_id)
);
CREATE TABLE IF NOT EXISTS quotes (
  rfq_id TEXT NOT NULL, ref TEXT NOT NULL, email_id TEXT NOT NULL, line_id TEXT,
  run_id TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY (rfq_id, ref)
);
CREATE TABLE IF NOT EXISTS line_state (
  rfq_id TEXT NOT NULL, line_id TEXT NOT NULL, status TEXT NOT NULL, run_id TEXT,
  plan TEXT, decision TEXT, note TEXT, updated_at TEXT NOT NULL, PRIMARY KEY (rfq_id, line_id)
);
CREATE TABLE IF NOT EXISTS reviews (
  id TEXT PRIMARY KEY, run_id TEXT NOT NULL, rfq_id TEXT NOT NULL, line_id TEXT NOT NULL,
  status TEXT NOT NULL, created_at TEXT NOT NULL, resolved_at TEXT, payload TEXT NOT NULL,
  resolution TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS outbox (
  id TEXT PRIMARY KEY, run_id TEXT NOT NULL, rfq_id TEXT NOT NULL, line_id TEXT, kind TEXT NOT NULL,
  supplier_id TEXT NOT NULL, subject TEXT NOT NULL, body TEXT NOT NULL, created_at TEXT NOT NULL,
  drafted_by TEXT NOT NULL, guard_notes TEXT, reply_email_id TEXT, reply_simulated INTEGER DEFAULT 0,
  source_email_id TEXT
);
CREATE TABLE IF NOT EXISTS eval_runs (
  id TEXT PRIMARY KEY, ts TEXT NOT NULL, mode TEXT NOT NULL, disabled_rules TEXT NOT NULL,
  metrics TEXT NOT NULL, cases TEXT NOT NULL, passed INTEGER NOT NULL
);
"""


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def data_dir() -> Path:
    path = Path(os.environ.get("RFQ_DATA_DIR", Path.cwd() / "data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _dumps(value: Any) -> str | None:
    return None if value is None else json.dumps(value, default=str)


class Store:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or data_dir() / "app.db"
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def q(self, sql: str, args: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    def one(self, sql: str, args: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        rows = self.q(sql, args)
        return rows[0] if rows else None

    def x(self, sql: str, args: tuple[Any, ...] = ()) -> None:
        with self._lock:
            self._conn.execute(sql, args)

    def reset(self) -> None:
        with self.tx() as c:
            for table in (
                "runs",
                "events",
                "spans",
                "alerts",
                "extractions",
                "quotes",
                "line_state",
                "reviews",
                "outbox",
                "eval_runs",
            ):
                c.execute(f"DELETE FROM {table}")

    # ------------------------------------------------------------ runs and timeline

    def create_run(
        self, run_id: str, rfq_id: str, kind: str, mode: str, model: str, parent_run_id: str | None = None
    ) -> None:
        self.x(
            "INSERT INTO runs (id, rfq_id, kind, status, mode, model, started_at, parent_run_id) "
            "VALUES (?, ?, ?, 'running', ?, ?, ?, ?)",
            (run_id, rfq_id, kind, mode, model, now_iso(), parent_run_id),
        )

    def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        self.x(f"UPDATE runs SET {cols} WHERE id = ?", (*fields.values(), run_id))

    def add_run_usage(self, run_id: str, cost: float, tin: int, tout: int) -> None:
        self.x(
            "UPDATE runs SET cost_usd = cost_usd + ?, input_tokens = input_tokens + ?, "
            "output_tokens = output_tokens + ?, llm_calls = llm_calls + 1 WHERE id = ?",
            (cost, tin, tout, run_id),
        )

    def event(self, run_id: str, node: str, message: str, level: str = "info", data: Any = None) -> None:
        self.x(
            "INSERT INTO events (run_id, ts, node, level, message, data) VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, now_iso(), node, level, message, _dumps(data)),
        )

    def alert(
        self, run_id: str | None, kind: str, message: str, severity: str = "warn", dedupe_key: str | None = None
    ) -> None:
        self.x(
            "INSERT OR IGNORE INTO alerts (run_id, ts, kind, severity, message, dedupe_key) VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, now_iso(), kind, severity, message, dedupe_key),
        )

    # ------------------------------------------------------------ quotes and lines

    def save_extraction(self, rfq_id: str, email_id: str, run_id: str, data: dict[str, Any]) -> None:
        self.x(
            "INSERT OR REPLACE INTO extractions (rfq_id, email_id, run_id, data) VALUES (?, ?, ?, ?)",
            (rfq_id, email_id, run_id, json.dumps(data)),
        )

    def extraction(self, rfq_id: str, email_id: str) -> dict[str, Any] | None:
        row = self.one("SELECT data FROM extractions WHERE rfq_id = ? AND email_id = ?", (rfq_id, email_id))
        return json.loads(row["data"]) if row else None

    def save_quotes(self, rfq_id: str, run_id: str, email_id: str, quotes: list[dict[str, Any]]) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM quotes WHERE rfq_id = ? AND email_id = ?", (rfq_id, email_id))
            for qd in quotes:
                c.execute(
                    "INSERT INTO quotes (rfq_id, ref, email_id, line_id, run_id, data) VALUES (?, ?, ?, ?, ?, ?)",
                    (rfq_id, qd["ref"], email_id, qd.get("line_id"), run_id, json.dumps(qd)),
                )

    def quotes(self, rfq_id: str) -> list[dict[str, Any]]:
        return [
            json.loads(r["data"]) for r in self.q("SELECT data FROM quotes WHERE rfq_id = ? ORDER BY ref", (rfq_id,))
        ]

    def set_line(
        self,
        rfq_id: str,
        line_id: str,
        status: str,
        run_id: str | None,
        plan: Any = None,
        decision: Any = None,
        note: str | None = None,
    ) -> None:
        self.x(
            "INSERT OR REPLACE INTO line_state (rfq_id, line_id, status, run_id, plan, decision, note, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (rfq_id, line_id, status, run_id, _dumps(plan), _dumps(decision), note, now_iso()),
        )

    def lines(self, rfq_id: str) -> dict[str, dict[str, Any]]:
        out = {}
        for r in self.q("SELECT * FROM line_state WHERE rfq_id = ?", (rfq_id,)):
            r["plan"] = json.loads(r["plan"]) if r["plan"] else None
            r["decision"] = json.loads(r["decision"]) if r["decision"] else None
            out[r["line_id"]] = r
        return out

    # ------------------------------------------------------------ reviews

    def open_review(self, run_id: str, rfq_id: str, line_id: str, payload: dict[str, Any]) -> str:
        # Idempotent: a resumed graph re-runs the review node, and must find the same card.
        existing = self.one("SELECT id FROM reviews WHERE run_id = ? AND line_id = ?", (run_id, line_id))
        if existing:
            return str(existing["id"])
        review_id = new_id("REV")
        self.x(
            "INSERT INTO reviews (id, run_id, rfq_id, line_id, status, created_at, payload) "
            "VALUES (?, ?, ?, ?, 'pending', ?, ?)",
            (review_id, run_id, rfq_id, line_id, now_iso(), json.dumps(payload)),
        )
        return review_id

    def review(self, review_id: str) -> dict[str, Any] | None:
        row = self.one("SELECT * FROM reviews WHERE id = ?", (review_id,))
        if row:
            row["payload"] = json.loads(row["payload"])
            row["resolution"] = json.loads(row["resolution"]) if row["resolution"] else None
        return row

    def reviews(self, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT id FROM reviews" + (" WHERE status = ?" if status else "") + " ORDER BY created_at DESC"
        return [r for r in (self.review(x["id"]) for x in self.q(sql, (status,) if status else ())) if r]

    # ------------------------------------------------------------ outbox

    def add_outbox(self, msg: dict[str, Any]) -> None:
        self.x(
            "INSERT OR IGNORE INTO outbox (id, run_id, rfq_id, line_id, kind, supplier_id, subject, body, created_at, "
            "drafted_by, guard_notes, reply_email_id, source_email_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                msg["id"],
                msg["run_id"],
                msg["rfq_id"],
                msg.get("line_id"),
                msg["kind"],
                msg["supplier_id"],
                msg["subject"],
                msg["body"],
                now_iso(),
                msg["drafted_by"],
                json.dumps(msg.get("guard_notes", [])),
                msg.get("reply_email_id"),
                msg.get("source_email_id"),
            ),
        )

    def outbox(self) -> list[dict[str, Any]]:
        rows = self.q("SELECT * FROM outbox ORDER BY created_at DESC, id")
        for r in rows:
            r["guard_notes"] = json.loads(r["guard_notes"] or "[]")
        return rows
