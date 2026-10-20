"""Memory Store: compressed session summaries + cross-run entity / claim recall.

LEARNING NOTE - "memory compaction":
After every run we keep a ~80 word summary, NOT the transcript. At startup the
Planner loads only the last 3 summaries (settings.session_summaries_to_load). That
bounds the Planner's memory cost forever, no matter how many runs have happened.
"""
import logging

from app import db, vector
from app.config import settings
from app.embeddings import embed
from app.llm import call_llm

log = logging.getLogger("insightforge.memory")


def recent_summaries(n: int | None = None) -> list[str]:
    n = n or settings.session_summaries_to_load
    rows = db.q("SELECT text FROM summaries ORDER BY id DESC LIMIT ?", (n,))
    return [r["text"] for r in reversed(rows)]


def save_session_summary(run_id: str, brief: str, report_markdown: str) -> str:
    """Compress a finished run into <=80 words (LLM, with a deterministic fallback)."""
    try:
        text = call_llm(
            f"Brief: {brief}\n\nReport (excerpt):\n{report_markdown[:3500]}\n\n"
            "Write a <=80 word memory note: what was researched, the 2-3 key findings, and open gaps. "
            "No preamble.",
            model=settings.fast_model,
        ).strip()
    except Exception as e:  # noqa: BLE001
        log.warning("summary LLM failed (%s); using fallback", e)
        text = f"Researched: {brief[:200]}"
    db.x("INSERT INTO summaries(run_id, text, created_at) VALUES (?,?,?)", (run_id, text, db.now()))
    return text


def related_entities(brief: str, k: int = 8) -> list[str]:
    """Entity names from PAST runs that are semantically close to this brief."""
    try:
        hits = vector.search("entities", embed([brief])[0], limit=k)
    except Exception as e:  # noqa: BLE001
        log.warning("entity recall failed: %s", e)
        return []
    return [h["payload"]["name"] for h in hits if h["score"] > 0.4]


def index_report_claims(report_id: str, citations: dict[str, dict]) -> int:
    """Store a finished report's cited claims so later runs can use 'prior_reports' as a source."""
    items = [c for c in citations.values() if c.get("claim")]
    if not items:
        return 0
    vecs = embed([c["claim"] for c in items])
    payloads = [
        {"report_id": report_id, "claim": c["claim"], "url": c.get("url", ""), "title": c.get("title", ""),
         "date": c.get("date"), "credibility": c.get("credibility", 0.5), "paragraph": c.get("paragraph", "")}
        for c in items
    ]
    vector.upsert("report_claims", vecs, payloads)
    return len(items)
