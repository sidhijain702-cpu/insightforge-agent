"""Watchlist scheduler: APScheduler runs the SAME pipeline as a manual brief, then the Diff Agent.

    cron-like job per item -> execute_run -> new report -> diff vs previous report
        -> material changes? -> in-app alert (+ optional webhook POST, Slack-compatible)

Set DEMO_INTERVAL_SECONDS=120 to see an alert fire within minutes during your demo video.
"""
import logging

import httpx
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app import db
from app.agents.diff import diff_reports
from app.config import settings
from app.pipeline import create_run, execute_run
from app.schemas import ChangeSummary, SourcePrefs

log = logging.getLogger("insightforge.scheduler")
scheduler = BackgroundScheduler(timezone="UTC")


def _trigger(cadence: str) -> IntervalTrigger:
    if settings.demo_interval_seconds > 0:
        return IntervalTrigger(seconds=settings.demo_interval_seconds)
    return IntervalTrigger(weeks=1) if cadence == "weekly" else IntervalTrigger(days=1)


def _job_id(watch_id: int) -> str:
    return f"watch_{watch_id}"


def _sync_next_run(watch_id: int) -> None:
    job = scheduler.get_job(_job_id(watch_id))
    nxt = job.next_run_time.timestamp() if job and job.next_run_time else None
    db.x("UPDATE watchlist SET next_run=? WHERE id=?", (nxt, watch_id))


def schedule_item(watch_id: int) -> None:
    item = db.q1("SELECT * FROM watchlist WHERE id=?", (watch_id,))
    if not item or not item["active"]:
        unschedule_item(watch_id)
        return
    scheduler.add_job(
        run_watch, _trigger(item["cadence"]), args=[watch_id], id=_job_id(watch_id),
        replace_existing=True, max_instances=1, coalesce=True, misfire_grace_time=3600,
    )
    _sync_next_run(watch_id)


def unschedule_item(watch_id: int) -> None:
    if scheduler.get_job(_job_id(watch_id)):
        scheduler.remove_job(_job_id(watch_id))
    db.x("UPDATE watchlist SET next_run=NULL WHERE id=?", (watch_id,))


def start_scheduler() -> None:
    if not scheduler.running:
        scheduler.start()
    for row in db.q("SELECT id FROM watchlist WHERE active=1"):
        schedule_item(row["id"])
    log.info("scheduler started with %d job(s)", len(scheduler.get_jobs()))


def stop_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)


# ------------------------------------------------------------------ the job
def slack_text(topic: str, ch: ChangeSummary) -> str:
    lines = [f"*InsightForge alert: {topic}*"]
    for label, items in (("New", ch.added_findings), ("Changed", ch.changed_findings), ("No longer found", ch.removed_findings)):
        lines += [f"• *{label}:* {x[:300]}" for x in items[:5]]
    return "\n".join(lines)


def send_webhook(url: str, topic: str, ch: ChangeSummary) -> str:
    try:
        r = httpx.post(url, json={"text": slack_text(topic, ch), "topic": topic, "changes": ch.model_dump()}, timeout=10)
        return f"HTTP {r.status_code}"
    except Exception as e:  # noqa: BLE001
        return f"error: {type(e).__name__}"


def run_watch(watch_id: int) -> int | None:
    """One monitoring cycle. Returns the alert id if a material change was found."""
    item = db.q1("SELECT * FROM watchlist WHERE id=?", (watch_id,))
    if not item or not item["active"]:
        return None
    topic = item["topic"]
    brief = (f"Latest developments for: {topic}. Cover funding, pricing, product launches, partnerships, "
             "leadership changes and risks, with specific figures and dates.")[:500]
    prefs = SourcePrefs(web=True, docs=False, prior=False)   # prior=False: diff must reflect the world, not our memory
    run_id = create_run(brief, prefs, watch_id)
    state = execute_run(run_id, brief, prefs, watch_id=watch_id, allow_clarification=False)
    if not state or not state.report_id:
        log.warning("watch %s run failed", watch_id)
        db.x("UPDATE watchlist SET last_run=? WHERE id=?", (db.now(), watch_id))
        _sync_next_run(watch_id)
        return None

    prev_id = item["last_report_id"]
    db.x("UPDATE watchlist SET last_run=?, last_report_id=? WHERE id=?", (db.now(), state.report_id, watch_id))
    _sync_next_run(watch_id)
    if not prev_id:
        log.info("watch %s: baseline report stored, nothing to diff yet", watch_id)
        return None

    old = db.q1("SELECT citations FROM reports WHERE id=?", (prev_id,))
    old_items = list((db.jload(old["citations"], {}) if old else {}).values())
    changes = diff_reports(old_items, list(state.citations.values()))
    if changes.is_empty():
        return None
    status = ""
    if item["webhook_url"]:
        status = send_webhook(item["webhook_url"], topic, changes)
    return db.x(
        "INSERT INTO alerts(watch_id, topic, run_id, summary, webhook_status, created_at) VALUES (?,?,?,?,?,?)",
        (watch_id, topic, run_id, changes.model_dump_json(), status, db.now()),
    )
