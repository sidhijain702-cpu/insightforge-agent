"""Synthesis Agent: raw extraction results -> ranked, de-duplicated, BUDGETED evidence bundle.

This is the ONE agent allowed to resolve full content from the scratch store / Qdrant,
because its whole job is compression. What leaves it is small and bounded:

    sections[ { theme, confidence, evidence[ {id, claim, domain, date, credibility, confidence} ] } ]
    token_count <= EVIDENCE_TOKEN_BUDGET   (default 6000)

Pipeline:  load claims -> de-duplicate -> flag conflicts -> confidence -> rank
           -> if over budget: COMPRESS lowest-ranked items (never blind truncation) + log why.
"""
import re
from dataclasses import dataclass, field

import numpy as np

from app.config import settings
from app.embeddings import embed
from app.entities import conflict_descriptions, flag_conflicts, norm
from app.events import add_metric, emit
from app.schemas import (
    CompressionLogEntry, EvidenceBundle, EvidenceItem, EvidenceSection, Relationship, TaskGraph,
)
from app.scratch_store import ScratchStore
from app.tokens import count_tokens
from app.tools.credibility import norm_date, parse_date

AGENT = "Synthesis Agent"
CONF_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}


@dataclass
class _Cand:
    claim_id: str
    text: str
    source_id: str
    sub_task_id: str
    url: str
    domain: str
    date: str | None
    cred: float
    entities: list[str] = field(default_factory=list)
    paragraph: str = ""


# ------------------------------------------------------------------ rendering (what the Writer sees)
def render_item(it: EvidenceItem) -> str:
    flag = " [CONFLICT]" if it.conflict else ""
    return f"[{it.id}] {it.claim} ({it.domain}, {it.date or 'n.d.'}, cred {it.credibility:.2f}, {it.confidence}){flag}"


def render_section(sec: EvidenceSection) -> str:
    head = f"## {sec.theme} [section confidence: {sec.confidence}]"
    return "\n".join([head] + [render_item(i) for i in sec.evidence])


def render_bundle(b: EvidenceBundle) -> str:
    return "\n\n".join(render_section(s) for s in b.sections)


def bundle_tokens(b: EvidenceBundle) -> int:
    return count_tokens(render_bundle(b))


# ------------------------------------------------------------------ ranking helpers
def _date_ord(d: str | None) -> int:
    p = parse_date(d)
    return p.toordinal() if p else 0


def rank_key(it: EvidenceItem):
    """Recency first, then credibility, then relevance to the brief (higher = better)."""
    return (_date_ord(it.date), it.credibility, it.relevance)


def _confidence_for(n_domains: int, cred: float, conflict: bool) -> str:
    if conflict:
        return "LOW"
    if n_domains >= 2:
        return "HIGH"       # 2+ independent sources agree
    if cred >= 0.6:
        return "MEDIUM"     # single high-credibility source
    return "LOW"            # single low-credibility source


def _section_confidence(items: list[EvidenceItem]) -> str:
    if not items:
        return "LOW"
    mean = sum(CONF_RANK[i.confidence] for i in items) / len(items)
    return "HIGH" if mean >= 2.5 else "MEDIUM" if mean >= 1.75 else "LOW"


# ------------------------------------------------------------------ compression
_STOP = {"The", "A", "An", "In", "On", "At", "It", "This", "These", "And", "But", "Or", "Of", "To", "For", "With", "As", "By"}


def _nums(s: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in re.findall(r"\d[\d,]*\.?\d*", s)} - {""}


def summary_is_grounded(summary: str, items: list[EvidenceItem]) -> bool:
    """A compressed summary may only rephrase/shorten its inputs: no new numbers, no new proper nouns,
    and most of its content words must come from the source claims. Otherwise it is rejected."""
    src = " ".join(i.claim for i in items)
    if _nums(summary) - _nums(src):
        return False
    src_words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9&'\-]*", src)}
    words = re.findall(r"[A-Za-z][A-Za-z0-9&'\-]*", summary)
    if any(w[0].isupper() and w not in _STOP and w.lower() not in src_words for w in words[1:]):
        return False
    mine = {w[:5] for w in re.findall(r"[a-z]{4,}", summary.lower())}
    theirs = {w[:5] for w in re.findall(r"[a-z]{4,}", src.lower())}
    return not mine or len(mine & theirs) / len(mine) >= 0.7


def _default_compress(items: list[EvidenceItem], theme: str) -> str:
    """LLM summary of low-priority items, VALIDATED against its inputs; deterministic fallback otherwise."""
    try:
        from app.llm import call_llm

        body = "\n".join(f"- {i.claim}" for i in items)
        out = call_llm(
            f"Theme: {theme}\nMerge these findings into ONE paragraph of at most 55 words. Keep every "
            f"number, name and date that appears. Add NOTHING new and do not connect facts that are not "
            f"connected in the text.\n{body}",
            model=settings.fast_model,
        ).strip()
        if out and len(out.split()) <= 70 and summary_is_grounded(out, items):
            return out
    except Exception:  # noqa: BLE001
        pass
    return _fallback_compress(items)


def _fallback_compress(items: list[EvidenceItem]) -> str:
    return "; ".join(" ".join(i.claim.split()[:14]) for i in items)


def _merge(items: list[EvidenceItem], summary: str) -> EvidenceItem:
    best = max(items, key=lambda i: (i.credibility, _date_ord(i.date)))
    domains = {d for i in items for d in ([i.domain] + i.supporting_domains)}
    conf = best.confidence if CONF_RANK[best.confidence] <= 2 else "MEDIUM"  # compressed => at most MEDIUM
    return EvidenceItem(
        id=items[0].id, claim=summary, source_id=best.source_id, url=best.url, domain=best.domain,
        date=max((i.date for i in items if i.date), default=None, key=_date_ord),
        credibility=best.credibility, confidence=conf, conflict=any(i.conflict for i in items),
        claim_ids=sorted({c for i in items for c in i.claim_ids}),
        source_ids=sorted({s for i in items for s in (i.source_ids or [i.source_id])}),
        supporting_domains=sorted(domains - {best.domain}), compressed=True,
        relevance=max(i.relevance for i in items),
    )


def enforce_budget(bundle: EvidenceBundle, budget: int, compress_fn=None, run_id: str | None = None) -> EvidenceBundle:
    """Shrink the bundle to <= budget tokens by compressing the LOWEST-ranked items and logging each cut."""
    compress_fn = compress_fn or _default_compress
    guard = 0
    while bundle_tokens(bundle) > budget and guard < 300:
        guard += 1
        before = bundle_tokens(bundle)
        for s in bundle.sections:
            s.evidence.sort(key=rank_key, reverse=True)
        # the lowest-ranked item among sections that still have >= 2 items
        cands = [(rank_key(s.evidence[-1]), s) for s in bundle.sections if len(s.evidence) >= 2]
        reason = (f"evidence bundle {before} tokens > budget {budget}; merged lowest-ranked items "
                  "(rank = recency, then credibility, then relevance)")
        if cands:
            _, sec = min(cands, key=lambda t: t[0])
            m = min(4, max(2, len(sec.evidence) - 1))
            tail = sec.evidence[-m:]
            summary = compress_fn(tail, sec.theme)
            merged = _merge(tail, summary)
            sec.evidence = sec.evidence[:-m] + [merged]
            if bundle_tokens(bundle) >= before:  # LLM was verbose -> force a shorter deterministic merge
                merged.claim = " ".join(_fallback_compress(tail).split()[:40])
            action, affected, sids = "compressed", len(tail), merged.source_ids
        else:
            # every section has <=1 item: shorten the longest claim, else drop the weakest item
            allitems = [(rank_key(i), s, i) for s in bundle.sections for i in s.evidence]
            if not allitems:
                break
            long = max(allitems, key=lambda t: len(t[2].claim))
            shorter = " ".join(long[2].claim.split()[:30])
            if len(long[2].claim) > 220 and len(shorter) < len(long[2].claim):
                long[2].claim = shorter
                action, affected, sids, sec = "compressed", 1, [long[2].source_id], long[1]
            else:
                _, sec, it = min(allitems, key=lambda t: t[0])
                sec.evidence.remove(it)
                action, affected, sids = "dropped", 1, [it.source_id]
                reason += "; cannot compress further"
        after = bundle_tokens(bundle)
        bundle.compression_log.append(CompressionLogEntry(
            section=sec.theme, action=action, items_affected=affected, source_ids=sids,
            reason=reason, tokens_before=before, tokens_after=after,
        ))
        if run_id:
            emit(run_id, AGENT, f"{action} {affected} item(s) in '{sec.theme}': {before} -> {after} tokens (budget {budget})")
    for s in bundle.sections:
        s.confidence = _section_confidence(s.evidence)
    bundle.token_count = bundle_tokens(bundle)
    return bundle


# ------------------------------------------------------------------ main entry
def synthesize(
    run_id: str, brief: str, task_graph: TaskGraph, failed: dict[str, str] | None = None,
    budget: int | None = None, store: ScratchStore | None = None, compress_fn=None,
) -> EvidenceBundle:
    store = store or ScratchStore(run_id)
    budget = budget or settings.evidence_token_budget
    failed = failed or {}

    # 1. resolve full content from the scratch store (allowed here, and only here)
    cands: list[_Cand] = []
    rels: list[Relationship] = []
    for st in task_graph.sub_tasks:
        ex = store.read_extraction(st.id)
        if not ex:
            continue
        rels.extend(ex.graph.relationships)
        for c in ex.claims:
            src = store.read(c.source_id)
            cands.append(_Cand(c.id, c.text, src.id, st.id, src.url, src.domain, norm_date(src.published_date),
                               src.credibility_score, c.entities, c.paragraph))
    emit(run_id, AGENT, f"resolved {len(cands)} claims from {len({c.source_id for c in cands})} sources")

    # 2. de-duplicate (same entity + same claim => keep highest-credibility source)
    brief_vec = embed([brief])[0]
    vecs = embed([c.text for c in cands]) if cands else np.zeros((0, 384), dtype=np.float32)
    order = sorted(range(len(cands)), key=lambda i: (-cands[i].cred, -_date_ord(cands[i].date)))
    kept: list[int] = []
    support: dict[int, dict] = {}
    for i in order:
        ents_i = {norm(e) for e in cands[i].entities}
        dup_of = None
        for k in kept:
            ents_k = {norm(e) for e in cands[k].entities}
            overlap = bool(ents_i & ents_k) or not ents_i or not ents_k
            if float(vecs[i] @ vecs[k]) >= (0.88 if overlap else 0.97):
                dup_of = k
                break
        if dup_of is None:
            kept.append(i)
            support[i] = {"domains": set(), "claims": [cands[i].claim_id], "sources": [cands[i].source_id]}
        else:
            if cands[i].domain != cands[dup_of].domain:
                support[dup_of]["domains"].add(cands[i].domain)
            support[dup_of]["claims"].append(cands[i].claim_id)
            support[dup_of]["sources"].append(cands[i].source_id)
    emit(run_id, AGENT, f"de-duplicated {len(cands)} -> {len(kept)} distinct claims")

    # 3. conflicts (entity extractor already flagged per sub-task; re-check globally across sub-tasks)
    rels = flag_conflicts(rels)
    conflicts = conflict_descriptions(rels)
    conflict_urls: dict[str, set[str]] = {}
    for r in rels:
        if r.conflict:
            conflict_urls.setdefault(norm(r.subject), set()).add(r.source_url)

    # 4. build evidence items per section
    sections: list[EvidenceSection] = []
    for st in sorted(task_graph.sub_tasks, key=lambda s: (-s.priority, s.id)):
        items: list[EvidenceItem] = []
        for i in kept:
            c = cands[i]
            if c.sub_task_id != st.id:
                continue
            conflict = any(c.url in urls and subj and subj in norm(c.text) for subj, urls in conflict_urls.items())
            doms = {c.domain} | support[i]["domains"]
            items.append(EvidenceItem(
                id="", claim=c.text, source_id=c.source_id, url=c.url, domain=c.domain, date=c.date,
                credibility=c.cred, confidence=_confidence_for(len(doms), c.cred, conflict), conflict=conflict,
                conflict_note="sources disagree on this entity" if conflict else "",
                claim_ids=support[i]["claims"], source_ids=sorted(set(support[i]["sources"])),
                supporting_domains=sorted(support[i]["domains"]),
                relevance=round(float(vecs[i] @ brief_vec), 3),
            ))
        items.sort(key=rank_key, reverse=True)
        sections.append(EvidenceSection(sub_task_id=st.id, theme=st.theme(), evidence=items))

    n = 0
    for s in sections:           # stable global ids E1, E2, ...
        for it in s.evidence:
            n += 1
            it.id = f"E{n}"
        s.confidence = _section_confidence(s.evidence)

    # 5. gaps the Writer must mention
    gaps = [f"Sub-task '{sid}' failed: {why}" for sid, why in failed.items()]
    gaps += [f"Section '{s.theme}' has no usable evidence." for s in sections if not s.evidence and s.sub_task_id not in failed]
    sk = store.skipped()
    if sk:
        gaps.append(f"{len(sk)} source URL(s) were skipped (paywall, login wall or fetch error).")

    bundle = EvidenceBundle(sections=sections, token_budget=budget, gaps=gaps, conflicts=conflicts)
    before = bundle_tokens(bundle)
    add_metric(run_id, "bundle_tokens_before_compression", before)
    enforce_budget(bundle, budget, compress_fn, run_id)
    add_metric(run_id, "bundle_tokens_final", bundle.token_count)
    add_metric(run_id, "compression_steps", len(bundle.compression_log))
    add_metric(run_id, "claims_resolved", len(cands))
    emit(run_id, AGENT, f"bundle ready: {bundle.token_count}/{budget} tokens, {len(bundle.compression_log)} compression step(s)")
    if bundle.compression_log:
        bundle.gaps.append(
            f"{sum(e.items_affected for e in bundle.compression_log)} lower-priority finding(s) were condensed to "
            f"fit the {budget}-token evidence budget."
        )
    return bundle
