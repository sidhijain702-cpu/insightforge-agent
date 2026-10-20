"""FastAPI backend.

    POST /research/run              start a brief (multipart: brief, source toggles, up to 10 PDFs)
    GET  /research/status/{run_id}  SSE stream: log lines + state snapshots (sub-task rows, progress, elapsed)
    GET  /research/{run_id}         same snapshot as plain JSON
    GET/POST/PATCH/DELETE /watchlist ... CRUD, POST /watchlist/{id}/run (run now), GET /watchlist/{id}/diff
    GET  /alerts                    in-app alert feed
    GET  /reports, /reports/{id}, /reports/{id}/evidence/{ev}, /reports/{id}/export?format=pdf|md
    POST /reports/{id}/share        JWT link (7 days)   ->  GET /share/{token}
"""
import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from typing import Literal

import jwt
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app import db, export, pipeline, scheduler as sched
from app.config import settings
from app.events import get_metrics, snapshot
from app.schemas import SourcePrefs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
TERMINAL = {"COMPLETE", "FAILED", "NEEDS_CLARIFICATION"}
MAX_FILES, MAX_FILE_BYTES, MAX_BRIEF = 10, 20 * 1024 * 1024, 500


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    sched.start_scheduler()
    from app import embeddings

    def warm():  # load the embedding model in the background and report which backend is active
        logging.getLogger("insightforge").info("embedding backend: %s", embeddings.backend_name())

    threading.Thread(target=warm, daemon=True, name="embedding-warmup").start()
    yield
    sched.stop_scheduler()


app = FastAPI(title="InsightForge AI", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=list({settings.frontend_url, "http://localhost:3000"}),
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


# =============================================================== research
@app.post("/research/run")
async def research_run(
    brief: str = Form(...), web: bool = Form(True), docs: bool = Form(True), prior: bool = Form(True),
    files: list[UploadFile] = File(default=[]),
):
    brief = brief.strip()
    if not brief:
        raise HTTPException(400, "brief is empty")
    if len(brief) > MAX_BRIEF:
        raise HTTPException(400, f"brief is {len(brief)} characters; the limit is {MAX_BRIEF}")
    files = [f for f in files if f.filename]
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"at most {MAX_FILES} PDFs per run")
    payload: list[tuple[str, bytes]] = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(400, f"{f.filename}: only PDF files are supported")
        data = await f.read()
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(413, f"{f.filename} is larger than 20 MB")
        payload.append((f.filename, data))
    prefs = SourcePrefs(web=web, docs=docs and bool(payload), prior=prior)
    if not (prefs.web or prefs.docs or prefs.prior):
        raise HTTPException(400, "enable at least one source (web search, uploaded docs or prior reports)")
    run_id = pipeline.create_run(brief, prefs)
    if payload:
        pipeline.save_uploads(run_id, payload)
    pipeline.start_run_async(run_id, brief, prefs)
    return {"run_id": run_id}


@app.get("/research/runs")
def list_runs(limit: int = 20):
    return db.q("SELECT id, brief, status, report_id, created_at FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))


@app.get("/research/status/{run_id}")
async def research_status(run_id: str, request: Request):
    if not await asyncio.to_thread(db.q1, "SELECT id FROM runs WHERE id=?", (run_id,)):
        raise HTTPException(404, "unknown run")
    last = int(request.headers.get("last-event-id") or 0)

    async def gen():
        nonlocal last
        while True:
            if await request.is_disconnected():
                return
            rows = await asyncio.to_thread(
                db.q, "SELECT id, ts, agent, message, kind, level FROM events WHERE run_id=? AND id>? ORDER BY id LIMIT 200",
                (run_id, last),
            )
            for r in rows:
                last = r["id"]
                yield {"event": "log", "id": str(last), "data": json.dumps(r)}
            snap = await asyncio.to_thread(snapshot, run_id)
            yield {"event": "state", "data": json.dumps(snap)}
            if snap.get("status") in TERMINAL and not rows:
                yield {"event": "done", "data": json.dumps(snap)}
                return
            await asyncio.sleep(1.0)

    return EventSourceResponse(gen())


@app.get("/research/{run_id}")
def research_snapshot(run_id: str):
    snap = snapshot(run_id)
    if not snap:
        raise HTTPException(404, "unknown run")
    snap["metrics"] = get_metrics(run_id)
    return snap


# =============================================================== watchlist
class WatchIn(BaseModel):
    topic: str
    cadence: Literal["daily", "weekly"] = "daily"
    webhook_url: str | None = None


class WatchPatch(BaseModel):
    topic: str | None = None
    cadence: Literal["daily", "weekly"] | None = None
    webhook_url: str | None = None
    active: bool | None = None


@app.get("/watchlist")
def watch_list():
    return db.q("SELECT * FROM watchlist ORDER BY id DESC")


@app.post("/watchlist")
def watch_create(w: WatchIn):
    if not w.topic.strip():
        raise HTTPException(400, "topic is empty")
    wid = db.x(
        "INSERT INTO watchlist(topic, cadence, webhook_url, active, created_at) VALUES (?,?,?,1,?)",
        (w.topic.strip(), w.cadence, w.webhook_url or None, db.now()),
    )
    sched.schedule_item(wid)
    return db.q1("SELECT * FROM watchlist WHERE id=?", (wid,))


@app.patch("/watchlist/{wid}")
def watch_update(wid: int, p: WatchPatch):
    if not db.q1("SELECT id FROM watchlist WHERE id=?", (wid,)):
        raise HTTPException(404, "not found")
    for col, val in p.model_dump(exclude_unset=True).items():
        db.x(f"UPDATE watchlist SET {col}=? WHERE id=?", (int(val) if col == "active" else val, wid))
    sched.schedule_item(wid)
    return db.q1("SELECT * FROM watchlist WHERE id=?", (wid,))


@app.delete("/watchlist/{wid}")
def watch_delete(wid: int):
    sched.unschedule_item(wid)
    db.x("DELETE FROM watchlist WHERE id=?", (wid,))
    return {"deleted": wid}


@app.post("/watchlist/{wid}/run")
def watch_run_now(wid: int):
    if not db.q1("SELECT id FROM watchlist WHERE id=?", (wid,)):
        raise HTTPException(404, "not found")
    threading.Thread(target=sched.run_watch, args=(wid,), daemon=True).start()
    return {"started": True}


def _alert(row: dict) -> dict:
    row["summary"] = db.jload(row["summary"], {})
    return row


@app.get("/watchlist/{wid}/diff")
def watch_diff(wid: int):
    row = db.q1("SELECT * FROM alerts WHERE watch_id=? ORDER BY id DESC LIMIT 1", (wid,))
    return _alert(row) if row else {"summary": {"added_findings": [], "removed_findings": [], "changed_findings": []}}


@app.get("/alerts")
def alerts(limit: int = 50):
    return [_alert(r) for r in db.q("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))]


@app.post("/alerts/{aid}/read")
def alert_read(aid: int):
    db.x("UPDATE alerts SET read=1 WHERE id=?", (aid,))
    return {"ok": True}


# =============================================================== reports
def _report_row(rid: str) -> dict:
    row = db.q1("SELECT * FROM reports WHERE id=?", (rid,))
    if not row:
        raise HTTPException(404, "report not found")
    return row


def _report_payload(row: dict) -> dict:
    bundle = db.jload(row["bundle"], {}) or {}
    return {
        "id": row["id"], "title": row["title"], "brief": row["brief"], "markdown": row["markdown"],
        "created_at": row["created_at"], "factcheck": db.jload(row["factcheck"], {}),
        "compression_log": bundle.get("compression_log", []),
        "token_count": bundle.get("token_count"), "token_budget": bundle.get("token_budget"),
        "metrics": get_metrics(row["run_id"]),
    }


def _evidence(row: dict, ev_id: str) -> dict:
    c = (db.jload(row["citations"], {}) or {}).get(ev_id)
    if not c:
        raise HTTPException(404, "citation not found")
    return c


@app.get("/reports")
def reports_list():
    rows = db.q("SELECT id, title, watch_id, factcheck, created_at FROM reports ORDER BY created_at DESC LIMIT 100")
    for r in rows:
        r["pass_rate"] = (db.jload(r.pop("factcheck"), {}) or {}).get("pass_rate")
    return rows


@app.get("/reports/{rid}")
def report_get(rid: str):
    return _report_payload(_report_row(rid))


@app.get("/reports/{rid}/evidence/{ev_id}")
def report_evidence(rid: str, ev_id: str):
    return _evidence(_report_row(rid), ev_id)


@app.get("/reports/{rid}/export")
def report_export(rid: str, format: Literal["pdf", "md"] = Query("pdf")):
    row = _report_row(rid)
    name = f"insightforge-{rid}"
    if format == "md":
        return Response(export.export_markdown(row["markdown"]), media_type="text/markdown",
                        headers={"Content-Disposition": f'attachment; filename="{name}.md"'})
    pdf = export.to_pdf(row["markdown"], row["title"])
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{name}.pdf"'})


@app.post("/reports/{rid}/share")
def report_share(rid: str):
    _report_row(rid)
    token, exp = export.make_share_token(rid)
    return {"token": token, "url": f"{settings.frontend_url}/share/{token}", "expires_at": exp.isoformat()}


def _shared_row(token: str) -> dict:
    try:
        return _report_row(export.verify_share_token(token))
    except jwt.ExpiredSignatureError:
        raise HTTPException(410, "this share link has expired")
    except jwt.PyJWTError:
        raise HTTPException(403, "invalid share link")


@app.get("/share/{token}")
def share_get(token: str):
    p = _report_payload(_shared_row(token))
    p.pop("metrics", None)   # read-only view: no run internals
    return p


@app.get("/share/{token}/evidence/{ev_id}")
def share_evidence(token: str, ev_id: str):
    return _evidence(_shared_row(token), ev_id)
