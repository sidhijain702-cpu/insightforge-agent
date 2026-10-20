"""Run lifecycle: create -> (ingest PDFs) -> LangGraph -> result.

Used by the API (manual briefs) and by the scheduler (watchlist runs) so both go
through exactly the same pipeline.
"""
import logging
import re
import threading
import uuid
from pathlib import Path

from app import db, vector
from app.config import settings
from app.embeddings import embed
from app.events import emit, set_status
from app.graph import build_graph
from app.schemas import RunState, SourcePrefs

log = logging.getLogger("insightforge.pipeline")
DOC = "Doc Reader Agent"


# ------------------------------------------------------------------ run records
def create_run(brief: str, prefs: SourcePrefs, watch_id: int | None = None) -> str:
    run_id = uuid.uuid4().hex[:12]
    db.x(
        "INSERT INTO runs(id, brief, prefs, status, watch_id, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
        (run_id, brief, prefs.model_dump_json(), "PLANNING", watch_id, db.now(), db.now()),
    )
    emit(run_id, "Orchestrator", "run created")
    return run_id


def save_uploads(run_id: str, files: list[tuple[str, bytes]]) -> list[Path]:
    d = Path(settings.uploads_root) / run_id
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for name, data in files:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name) or "doc.pdf"
        p = d / safe
        p.write_bytes(data)
        out.append(p)
    return out


# ------------------------------------------------------------------ PDF ingestion
def _chunks_docling(path: Path) -> list[dict]:
    from docling.chunking import HybridChunker
    from docling.document_converter import DocumentConverter

    doc = DocumentConverter().convert(str(path)).document
    chunks = []
    for ch in HybridChunker().chunk(dl_doc=doc):
        pages = sorted({p.page_no for it in ch.meta.doc_items for p in getattr(it, "prov", [])})
        chunks.append({"text": ch.text, "page": pages[0] if pages else 0, "heading": " > ".join(ch.meta.headings or [])})
    return chunks


def _chunks_pypdf(path: Path) -> list[dict]:
    from pypdf import PdfReader

    chunks = []
    for pno, page in enumerate(PdfReader(str(path)).pages, 1):
        buf = ""
        for para in re.split(r"\n\s*\n", page.extract_text() or ""):
            if len(buf) + len(para) > 1200 and buf:
                chunks.append({"text": buf.strip(), "page": pno, "heading": ""})
                buf = ""
            buf += para + "\n\n"
        if len(buf.strip()) > 80:
            chunks.append({"text": buf.strip(), "page": pno, "heading": ""})
    return chunks


def ingest_documents(run_id: str) -> int:
    """Chunk uploaded PDFs (Docling, with a pypdf fallback) and store them in Qdrant."""
    d = Path(settings.uploads_root) / run_id
    pdfs = sorted(d.glob("*.pdf")) if d.exists() else []
    total = 0
    for pdf in pdfs:
        emit(run_id, DOC, f"parsing {pdf.name} with Docling")
        try:
            chunks = _chunks_docling(pdf)
        except Exception as e:  # noqa: BLE001
            emit(run_id, DOC, f"Docling unavailable/failed ({type(e).__name__}); using pypdf fallback", level="warn")
            try:
                chunks = _chunks_pypdf(pdf)
            except Exception as e2:  # noqa: BLE001
                emit(run_id, DOC, f"could not read {pdf.name}: {e2}", level="error")
                continue
        chunks = [c for c in chunks if len(c["text"]) > 40]
        if not chunks:
            emit(run_id, DOC, f"{pdf.name}: no extractable text (scanned PDF?)", level="warn")
            continue
        vecs = embed([c["text"][:2000] for c in chunks])
        payloads = [{**c, "run_id": run_id, "filename": pdf.name, "uploaded_date": None} for c in chunks]
        vector.upsert("doc_chunks", vecs, payloads)
        total += len(chunks)
        emit(run_id, DOC, f"{pdf.name}: {len(chunks)} chunks indexed in Qdrant")
    return total


# ------------------------------------------------------------------ execution
def execute_run(run_id: str, brief: str, prefs: SourcePrefs, watch_id: int | None = None,
                allow_clarification: bool = True) -> RunState | None:
    """Blocking. Runs the whole LangGraph pipeline; never raises (errors become status=FAILED)."""
    try:
        ingest_documents(run_id)
        state = RunState(run_id=run_id, brief=brief, prefs=prefs, watch_id=watch_id,
                         allow_clarification=allow_clarification)
        out = build_graph().invoke(state)
        return out if isinstance(out, RunState) else RunState(**out)
    except Exception as e:  # noqa: BLE001
        log.exception("run %s failed", run_id)
        row = db.q1("SELECT status FROM runs WHERE id=?", (run_id,))
        if not row or row["status"] != "FAILED":
            set_status(run_id, "FAILED", f"{type(e).__name__}: {str(e)[:300]}")
        return None


def start_run_async(run_id: str, brief: str, prefs: SourcePrefs) -> threading.Thread:
    t = threading.Thread(target=execute_run, args=(run_id, brief, prefs), daemon=True, name=f"run-{run_id}")
    t.start()
    return t
