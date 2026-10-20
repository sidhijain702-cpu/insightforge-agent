"""Fact-Checker Agent: the last gate before COMPLETE.

For a random 20% sample of the report's claims (min 5):
  1. re-fetch the cited source URL (falls back to the cached copy if the live fetch fails)
  2. cut the page into sentence windows, embed them, load into a TEMPORARY Qdrant collection
  3. query Qdrant with the claim's embedding; top cosine score >= threshold (0.85) => verified
Failed claims get ` [UNVERIFIED]` inserted right after their citation in the report text.

LEARNING NOTE: the threshold is a precision/recall dial. 0.85 is strict for paraphrased
sentences, so the result details log every best score. Look at the distribution for
your first runs, then decide whether to tune FACTCHECK_THRESHOLD or the Writer's style.
"""
import math
import random
import re

from app import vector
from app.citations import FINAL_CITE, split_units, strip_citations
from app.config import settings
from app.embeddings import embed
from app.events import add_metric, emit
from app.schemas import FactCheckResult
from app.scratch_store import ScratchStore
from app.tools.fetch import fetch_article

AGENT = "Fact-Checker Agent"
UNVERIFIED = " [UNVERIFIED]"


def _windows(text: str, max_windows: int = 500) -> list[str]:
    """Sentence windows (1 and 2 sentences) so a claim can match single or merged statements."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 25]
    wins = list(sents) + [f"{a} {b}" for a, b in zip(sents, sents[1:])]
    return wins[:max_windows]


def best_similarity(claim: str, page_text: str) -> float:
    """Top cosine score of `claim` against the page, computed through Qdrant."""
    wins = _windows(page_text)
    if not wins:
        return 0.0
    vecs = embed(wins)
    with vector.temp_collection(vecs, [{"i": i} for i in range(len(wins))]) as name:
        hits = vector.search(name, embed([claim])[0], limit=1)
    return hits[0]["score"] if hits else 0.0


def _page_text(url: str, source_id: str, store: ScratchStore, cache: dict) -> tuple[str, str]:
    """(text, how) where how = 'live' | 'cached'."""
    if url in cache:
        return cache[url]
    text, how = "", "cached"
    if url.startswith("http"):
        f = fetch_article(url)
        if f.ok:
            text, how = f.text, "live"
    if not text and source_id:
        try:
            text = store.read(source_id).body_text  # doc chunks / memory / failed live fetch
        except KeyError:
            text = ""
    cache[url] = (text, how)
    return text, how


def fact_check(run_id: str, markdown: str, citations: dict[str, dict], seed: str | None = None,
               store: ScratchStore | None = None, similarity=best_similarity) -> tuple[str, FactCheckResult]:
    store = store or ScratchStore(run_id)
    units = [u for u in split_units(markdown, FINAL_CITE) if u.cites and all(c in citations for c in u.cites)]
    if not units:
        emit(run_id, AGENT, "no cited claims found to check", level="warn")
        return markdown, FactCheckResult()

    n = min(len(units), max(settings.factcheck_min_claims, math.ceil(settings.factcheck_sample_rate * len(units))))
    sample = random.Random(seed or run_id).sample(units, n)
    emit(run_id, AGENT, f"sampling {n} of {len(units)} claims (>= {settings.factcheck_threshold} cosine to pass)")

    cache: dict = {}
    details, failed_units = [], []
    for u in sample:
        best, how, best_cite = 0.0, "cached", u.cites[0]
        for cid in u.cites:                       # a merged sentence passes if ANY cited source supports it
            c = citations[cid]
            text, h = _page_text(c["url"], c.get("source_id", ""), store, cache)
            s = similarity(strip_citations(u.text), text)
            if s > best:
                best, how, best_cite = s, h, cid
        ok = best >= settings.factcheck_threshold
        details.append({"claim": u.text, "cite": best_cite, "score": round(best, 3), "passed": ok, "checked_against": how})
        emit(run_id, AGENT, f"{'PASS' if ok else 'FAIL'} {best:.2f} ({how}): {u.text[:70]}…", level="info" if ok else "warn")
        if not ok:
            failed_units.append((u, best_cite, best))

    # mark failures inline (edit from the end so earlier offsets stay valid)
    out = markdown
    for u, _, _ in sorted(failed_units, key=lambda t: t[0].end, reverse=True):
        out = out[: u.end] + UNVERIFIED + out[u.end :]

    passed = len(sample) - len(failed_units)
    res = FactCheckResult(
        total_checked=len(sample), passed=passed, failed=len(failed_units),
        pass_rate=round(passed / len(sample), 3),
        unverified_claims=[
            {"claim": u.text, "cite": c, "url": citations[c]["url"], "score": round(s, 3)} for u, c, s in failed_units
        ],
        details=details,
    )
    add_metric(run_id, "factcheck_pass_rate", res.pass_rate)
    emit(run_id, AGENT, f"fact-check: {passed}/{len(sample)} verified ({res.pass_rate:.0%})")
    return out, res

