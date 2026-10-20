"""Tiny SQLite layer. One connection per call (simple + thread-safe with WAL).

LEARNING NOTE: runs, sub-tasks, events, reports, watchlist and alerts all live in
SQLite. The SSE endpoint just *reads the events table*, which means progress
streaming works even though the pipeline runs in a different thread, and a
browser can reconnect and replay history.
"""
import json
import sqlite3
import time
from typing import Any

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  id TEXT PRIMARY KEY, brief TEXT, prefs TEXT, status TEXT, error TEXT,
  report_id TEXT, watch_id INTEGER, task_graph TEXT, metrics TEXT DEFAULT '{}',
  clarifying_question TEXT, created_at REAL, updated_at REAL
);
CREATE TABLE IF NOT EXISTS subtasks(
  run_id TEXT, id TEXT, title TEXT, query TEXT, source_type TEXT, time_horizon TEXT,
  priority INTEGER, status TEXT DEFAULT 'Queued', detail TEXT DEFAULT '',
  PRIMARY KEY(run_id, id)
);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts REAL,
  agent TEXT, message TEXT, kind TEXT DEFAULT 'log', level TEXT DEFAULT 'info'
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, id);
CREATE TABLE IF NOT EXISTS reports(
  id TEXT PRIMARY KEY, run_id TEXT, watch_id INTEGER, title TEXT, brief TEXT,
  markdown TEXT, bundle TEXT, citations TEXT, factcheck TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS watchlist(
  id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT, cadence TEXT, webhook_url TEXT,
  last_run REAL, next_run REAL, active INTEGER DEFAULT 1, last_report_id TEXT,
  created_at REAL
);
CREATE TABLE IF NOT EXISTS alerts(
  id INTEGER PRIMARY KEY AUTOINCREMENT, watch_id INTEGER, topic TEXT, run_id TEXT,
  summary TEXT, webhook_status TEXT DEFAULT '', created_at REAL, read INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS summaries(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, text TEXT, created_at REAL
);
"""


def conn() -> sqlite3.Connection:
    c = sqlite3.connect(settings.db_path, timeout=30, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db() -> None:
    c = conn()
    c.executescript(SCHEMA)
    c.commit()
    c.close()


def q(sql: str, args: tuple = ()) -> list[dict]:
    c = conn()
    try:
        return [dict(r) for r in c.execute(sql, args).fetchall()]
    finally:
        c.close()


def q1(sql: str, args: tuple = ()) -> dict | None:
    rows = q(sql, args)
    return rows[0] if rows else None


def x(sql: str, args: tuple = ()) -> int:
    c = conn()
    try:
        cur = c.execute(sql, args)
        c.commit()
        return cur.lastrowid or 0
    finally:
        c.close()


def jdump(obj: Any) -> str:
    if hasattr(obj, "model_dump_json"):
        return obj.model_dump_json()
    return json.dumps(obj, default=str)


def jload(s: str | None, default: Any = None) -> Any:
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def now() -> float:
    return time.time()
