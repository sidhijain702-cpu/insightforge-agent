"""Keyed scratch store for RAW content (web pages, doc chunks, extraction results).

THE central context-engineering idea of this project:

    researchers WRITE raw text here and return only {source_id, one-line summary};
    downstream agents READ specific sources on demand with `read_source(id)`.

Layout on disk (one folder per run):
    scratch/<run_id>/src_ab12cd34.json       full Source record (heavy)
    scratch/<run_id>/src_ab12cd34.ref.json   {sub_task_id, source_id, summary}  (light)
    scratch/<run_id>/extraction_<sub>.json   claims + entity graph for one sub-task
    scratch/<run_id>/skipped.json            paywalled / failed URLs (for the Gaps section)

Random ids (not a counter) because parallel subagents write at the same time.
"""
import json
import threading
import uuid
from pathlib import Path
from urllib.parse import urlparse

from app.config import settings
from app.schemas import ExtractionResult, Source, SourceRef
from app.tokens import count_tokens

_skip_lock = threading.Lock()


def domain_of(url: str) -> str:
    if url.startswith("doc://"):
        return url[6:].split("#")[0]
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


class ScratchStore:
    def __init__(self, run_id: str, root: str | None = None):
        self.run_id = run_id
        self.dir = Path(root or settings.scratch_root) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- sources
    def write_source(self, sub_task_id: str, url: str, summary: str, **fields) -> SourceRef:
        source_id = f"src_{uuid.uuid4().hex[:8]}"
        src = Source(
            id=source_id, run_id=self.run_id, sub_task_id=sub_task_id, url=url,
            domain=domain_of(url), **fields,
        )
        (self.dir / f"{source_id}.json").write_text(src.model_dump_json(indent=1), encoding="utf-8")
        ref = SourceRef(source_id=source_id, summary=summary)
        sidecar = {"sub_task_id": sub_task_id, **ref.model_dump()}
        (self.dir / f"{source_id}.ref.json").write_text(json.dumps(sidecar), encoding="utf-8")
        return ref

    def read(self, source_id: str) -> Source:
        """Full record, with doc-chunk text resolved from Qdrant (Synthesis uses this)."""
        path = self.dir / f"{source_id}.json"
        if not path.exists():
            raise KeyError(f"unknown source id {source_id}")
        src = Source.model_validate_json(path.read_text(encoding="utf-8"))
        if src.kind == "doc_chunk" and src.chunk_id and not src.body_text:
            from app import vector

            payload = vector.get_payload("doc_chunks", src.chunk_id) or {}
            src.body_text = payload.get("text", "")
        return src

    def read_source(self, source_id: str, max_chars: int = 4000) -> str:
        """The on-demand reader exposed to agents as a tool. Always size-capped."""
        src = self.read(source_id)
        text = src.body_text[:max_chars]
        more = "" if len(src.body_text) <= max_chars else f"\n[... truncated, {len(src.body_text)} chars total]"
        return f"{src.title} ({src.url}, {src.published_date or 'n.d.'})\n{text}{more}"

    def approx_tokens(self, source_id: str) -> int:
        return count_tokens(self.read(source_id).body_text)

    def refs_for(self, sub_task_id: str) -> list[SourceRef]:
        out = []
        for p in sorted(self.dir.glob("*.ref.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            if d["sub_task_id"] == sub_task_id:
                out.append(SourceRef(source_id=d["source_id"], summary=d["summary"]))
        return out

    # ---------------------------------------------------------- extraction results
    def write_extraction(self, res: ExtractionResult) -> None:
        (self.dir / f"extraction_{res.sub_task_id}.json").write_text(res.model_dump_json(indent=1), encoding="utf-8")

    def read_extraction(self, sub_task_id: str) -> ExtractionResult | None:
        p = self.dir / f"extraction_{sub_task_id}.json"
        return ExtractionResult.model_validate_json(p.read_text(encoding="utf-8")) if p.exists() else None

    # ---------------------------------------------------------- skipped urls
    def add_skipped(self, url: str, reason: str) -> None:
        with _skip_lock:
            p = self.dir / "skipped.json"
            items = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
            items.append({"url": url, "reason": reason})
            p.write_text(json.dumps(items), encoding="utf-8")

    def skipped(self) -> list[dict]:
        p = self.dir / "skipped.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
