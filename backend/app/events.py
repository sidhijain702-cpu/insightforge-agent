"""Run events, sub-task status and metrics. Everything the UI streams comes from here."""
import logging
import threading

from app import db

log = logging.getLogger("insightforge")
_mlock = threading.Lock()

# overall-progress weights per pipeline stage (used by the progress bar)
STAGE_PROGRESS = {
    "PLANNING": 5, "RESEARCHING": 15, "SYNTHESIZING": 65, "WRITING": 75,
    "FACT_CHECKING": 92, "COMPLETE": 100, "NEEDS_CLARIFICATION": 100, "FAILED": 100,
}


def emit(run_id: str, agent: str, message: str, kind: str = "log", level: str = "info") -> None:
    """Append one line to the per-agent activity log (shown live in the UI)."""
    log.info("[%s] %s: %s", run_id[:6], agent, message)
    db.x(
        "INSERT INTO events(run_id, ts, agent, message, kind, level) VALUES (?,?,?,?,?,?)",
        (run_id, db.now(), agent, message, kind, level),
    )


def set_status(run_id: str, status: str, error: str | None = None) -> None:
    db.x("UPDATE runs SET status=?, error=?, updated_at=? WHERE id=?", (status, error, db.now(), run_id))
    emit(run_id, "Orchestrator", f"state -> {status}", kind="state")


def set_subtask(run_id: str, sub_id: str, status: str, detail: str = "") -> None:
    db.x(
        "UPDATE subtasks SET status=?, detail=? WHERE run_id=? AND id=?",
        (status, detail, run_id, sub_id),
    )
    emit(run_id, "Planner", f"sub-task {sub_id}: {status} {detail}".strip(), kind="subtask")


def add_metric(run_id: str, key: str, value, mode: str = "set") -> None:
    """mode: set | incr | append. Used later to write the context-engineering evidence."""
    with _mlock:
        row = db.q1("SELECT metrics FROM runs WHERE id=?", (run_id,))
        m = db.jload(row["metrics"] if row else None, {}) or {}
        if mode == "incr":
            m[key] = m.get(key, 0) + value
        elif mode == "append":
            m.setdefault(key, []).append(value)
        else:
            m[key] = value
        db.x("UPDATE runs SET metrics=? WHERE id=?", (db.jdump(m), run_id))


def get_metrics(run_id: str) -> dict:
    row = db.q1("SELECT metrics FROM runs WHERE id=?", (run_id,))
    return db.jload(row["metrics"] if row else None, {}) or {}


def snapshot(run_id: str) -> dict:
    """Current run state for the progress panel (status, sub-task rows, progress %, elapsed)."""
    run = db.q1("SELECT * FROM runs WHERE id=?", (run_id,))
    if not run:
        return {}
    subs = db.q("SELECT id,title,query,source_type,status,detail FROM subtasks WHERE run_id=? ORDER BY priority DESC, id", (run_id,))
    base = STAGE_PROGRESS.get(run["status"], 0)
    if run["status"] == "RESEARCHING" and subs:
        done = sum(1 for s in subs if s["status"] in ("Done", "Failed"))
        base = 15 + int(50 * done / len(subs))
    end = run["updated_at"] if run["status"] in ("COMPLETE", "FAILED", "NEEDS_CLARIFICATION") else db.now()
    return {
        "run_id": run_id,
        "status": run["status"],
        "progress": base,
        "elapsed": round(end - run["created_at"], 1),
        "subtasks": subs,
        "report_id": run["report_id"],
        "error": run["error"],
        "clarifying_question": run["clarifying_question"],
    }
