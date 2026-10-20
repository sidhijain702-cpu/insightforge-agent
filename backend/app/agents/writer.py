"""Writer Agent: evidence bundle -> structured Markdown report with inline citations.

Design (map-reduce writing):
  * each section is written by its OWN LLM call that sees ONLY that section's evidence
  * the executive summary sees only the top findings per section
  * the Gaps section is deterministic (built from the bundle's gap notes, no LLM)
  * the orchestrator assembles headings + confidence labels itself, so those are never wrong

Grounding check ("the Writer must not invent facts"):
  the LLM cites with {E12} markers. For every claim unit we verify
    1. every cited id exists in that section's evidence              (no phantom sources)
    2. every number in the sentence appears in the cited evidence     (no invented figures)
    3. proper nouns in the sentence appear in the evidence            (no invented names)
    4. enough content words overlap with the cited evidence           (no off-topic text)
  Violations => retry the section with the violations spelled out as an explicit constraint.
  After the last retry, offending sentences are removed (and noted), never shipped.
"""
import re
from dataclasses import dataclass, field

from app.citations import MARKER, Unit, split_units
from app.config import settings
from app.events import add_metric, emit
from app.llm import call_llm
from app.schemas import EvidenceBundle, EvidenceItem, EvidenceSection
from app.scratch_store import ScratchStore
from app.tokens import count_tokens
from app.agents.synthesis import render_item

AGENT = "Writer Agent"

STOP_CAPS = {
    "The", "A", "An", "In", "On", "At", "It", "This", "These", "Those", "They", "However", "Meanwhile",
    "Overall", "Additionally", "According", "For", "As", "With", "By", "Its", "Their", "We", "There",
    "Several", "Some", "Most", "Many", "Both", "Each", "While", "Although", "Also", "Another", "One",
    "Two", "Three", "Evidence", "Source", "And", "But", "Or", "Of", "To", "From", "Most", "Both", "Conflict",
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
    "November", "December", "Q1", "Q2", "Q3", "Q4", "Notably", "Further", "Moreover", "Recent", "Recently",
}
STOPWORDS = set(
    "the and that with from this have has had were was been into over than their there which will would about "
    "also more most such other some only when what where while after before between among within across being "
    "they them these those then just very much many each both said says report reports according".split()
)

HEDGE = re.compile(
    r"\b(limited|thin|sparse|no (usable |sufficient |direct )?evidence|insufficient|not enough|unclear|disagree|"
    r"conflict|could not|unavailable|not found|no data)\b", re.I)

SECTION_SYSTEM = """You write one section of a market-intelligence report for busy strategy analysts.
HARD RULES:
1. Use ONLY the evidence items provided. Do not add facts, numbers, names, dates or context from anywhere else.
2. End EVERY sentence or bullet that states a fact with the id(s) of the item(s) it came from, written exactly
   like {E3} or {E3}{E7}. Never invent ids.
3. Never cite an item for something it does not say. Copy numbers exactly as given.
4. If an item is marked [CONFLICT], say sources disagree and give the differing figures, cited.
5. Every sentence must be cited, EXCEPT one short note saying that evidence is thin or missing. Never add your own
   conclusions, implications or comparisons that no item states. Do not pad.
6. No headings. 2-4 short paragraphs or bullets, at most 200 words. Plain, direct language."""


@dataclass
class WriterResult:
    markdown: str
    citations: dict[str, dict] = field(default_factory=dict)
    retries: int = 0
    removed: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- grounding check
def _numbers(s: str) -> set[str]:
    out = set()
    for n in re.findall(r"\d[\d,]*\.?\d*", s):
        n = n.replace(",", "").rstrip(".")
        if n:
            out.add(n.rstrip("0").rstrip(".") if "." in n else n)
    return out


def _proper_nouns(s: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9&'\-]*", s)
    out = set()
    for idx, w in enumerate(words):
        if idx == 0 or w in STOP_CAPS:
            continue
        if w[0].isupper():
            out.add(w.lower())
    return out


def _content_prefixes(s: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z]{4,}", s.lower()) if w not in STOPWORDS}


def check_unit(unit: Unit, evidence_by_id: dict[str, EvidenceItem], section_text: str) -> list[str]:
    """Return a list of human-readable violations for one claim unit (empty = grounded)."""
    v: list[str] = []
    missing = [c for c in unit.cites if c not in evidence_by_id]
    if missing:
        return [f"cites unknown evidence id(s) {missing}: \"{unit.text[:80]}\""]
    if not unit.cites:
        # An uncited sentence is allowed ONLY as a data-limitation note ("evidence is thin", "sources disagree").
        # Anything else is an unsupported inference and must be cited or removed.
        if HEDGE.search(unit.text) and len(unit.text.split()) <= 30 and not re.search(r"\d", unit.text):
            return []
        return [f"uncited statement that is not backed by evidence: \"{unit.text[:80]}\""]
    ev_text = " ".join(evidence_by_id[c].claim for c in unit.cites)
    bad_nums = _numbers(unit.text) - _numbers(ev_text)
    if bad_nums:
        v.append(f"numbers {sorted(bad_nums)} not in cited evidence: \"{unit.text[:80]}\"")
    ev_words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9&'\-]*", ev_text)}
    bad_names = _proper_nouns(unit.text) - ev_words
    if len(bad_names) >= 2:
        v.append(f"names {sorted(bad_names)} not in cited evidence: \"{unit.text[:80]}\"")
    mine, theirs = _content_prefixes(unit.text), _content_prefixes(ev_text)
    if mine and len(mine & theirs) / len(mine) < settings.writer_overlap_threshold:
        v.append(f"statement does not match the cited evidence: \"{unit.text[:80]}\"")
    return v


def _remove_spans(text: str, units: list[Unit]) -> str:
    for u in sorted(units, key=lambda u: u.start, reverse=True):
        text = text[: u.start] + text[u.end :]
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def write_grounded(
    prompt: str, evidence_by_id: dict[str, EvidenceItem], system: str, model: str,
    max_words: int | None = None, retries: int | None = None, llm=call_llm,
) -> tuple[str, int, list[str]]:
    """Draft -> check -> retry with explicit constraints -> strip leftovers. Returns (text, retries_used, removed)."""
    retries = settings.writer_max_retries if retries is None else retries
    section_text = " ".join(i.claim for i in evidence_by_id.values())
    constraint, used = "", 0
    text, violations, bad_units = "", [], []
    for attempt in range(retries + 1):
        text = llm(prompt + constraint, system=system, model=model, temperature=0.2).strip()
        units = split_units(text, MARKER)
        violations, bad_units = [], []
        for u in units:
            vs = check_unit(u, evidence_by_id, section_text)
            if vs:
                violations.extend(vs)
                bad_units.append(u)
        too_long = max_words is not None and len(MARKER.sub("", text).split()) > max_words
        if not violations and not too_long:
            return text, used, []
        used = attempt + 1
        constraint = "\n\nCORRECTION REQUIRED. Your previous draft was rejected:\n"
        if violations:
            constraint += "\n".join(f"- {x}" for x in violations[:8])
            constraint += "\nRewrite the text. Use ONLY facts and numbers from the evidence items, cite with real ids."
        if too_long:
            constraint += f"\n- It exceeded {max_words} words. Shorten it to at most {max_words} words."
    # retries exhausted: remove offending sentences instead of shipping them
    cleaned = _remove_spans(text, bad_units) if bad_units else text
    if max_words is not None and len(MARKER.sub("", cleaned).split()) > max_words:
        keep, n = [], 0
        for u in split_units(cleaned, MARKER):
            n += len(u.text.split())
            if n > max_words:
                break
            keep.append(u)
        cleaned = " ".join(f"{u.text} " + "".join("{%s}" % c for c in u.cites) for u in keep).strip()
    return cleaned, used, violations


# --------------------------------------------------------------------------- assembly helpers
def cite_str(it: EvidenceItem) -> str:
    return f"[Source: {it.domain}, {it.date or 'n.d.'}, credibility: {it.credibility:.2f}](cite:{it.id})"


_RUN = re.compile(r"(?:\{E\d+\}\s*)+")


def expand_markers(text: str, items: dict[str, EvidenceItem]) -> str:
    """Replace {E#} runs by citations; adjacent citations that would read identically are shown once."""
    def repl(m: re.Match) -> str:
        raw = m.group(0)
        trailing = raw[len(raw.rstrip()):]
        seen, parts = set(), []
        for eid in MARKER.findall(raw):
            it = items.get(eid)
            if not it:
                continue
            label = (it.domain, it.date, round(it.credibility, 2))
            if label in seen:
                continue
            seen.add(label)
            parts.append(cite_str(it))
        return " ".join(parts) + trailing

    out = _RUN.sub(repl, text)
    out = re.sub(r"(?<=\S)(\[Source:)", r" \1", out)     # never glue a citation onto the previous word
    return re.sub(r"[ \t]+\.", ".", out)


def _paragraph_for(item: EvidenceItem, claim_index: dict) -> str:
    paras = [claim_index[c].paragraph for c in item.claim_ids if c in claim_index and claim_index[c].paragraph]
    if not paras:
        return ""
    if item.compressed or len(paras) > 1:
        shown = " … ".join(p[:260] for p in paras[:3])
        return f"(condensed from {len(item.claim_ids)} findings) {shown}"
    return paras[0]


def _claim_index(store: ScratchStore, bundle: EvidenceBundle) -> dict:
    idx = {}
    for sec in bundle.sections:
        ex = store.read_extraction(sec.sub_task_id)
        if ex:
            idx.update({c.id: c for c in ex.claims})
    return idx


# --------------------------------------------------------------------------- main entry
def write_report(run_id: str, brief: str, bundle: EvidenceBundle, store: ScratchStore | None = None,
                 llm=call_llm) -> WriterResult:
    store = store or ScratchStore(run_id)
    all_items = {it.id: it for s in bundle.sections for it in s.evidence}
    retries_total, removed_all = 0, []
    parts: list[str] = [f"# Market Intelligence Report\n\n*Brief: {brief.strip()}*\n"]

    # ---- executive summary: only the strongest findings of each section
    top = []
    for s in bundle.sections:
        best = sorted(s.evidence, key=lambda i: (-{"HIGH": 3, "MEDIUM": 2, "LOW": 1}[i.confidence], -i.credibility))[:2]
        top.extend(best)
    emit(run_id, AGENT, "writing executive summary")
    if top:
        ev = {i.id: i for i in top}
        prompt = (
            "Write an executive summary of AT MOST 120 words in plain language for a non-expert executive, "
            "no jargon. Lead with the most important finding. Cite with {E#} ids.\n\nEvidence:\n"
            + "\n".join(render_item(i) for i in top)
        )
        text, r, rem = write_grounded(prompt, ev, SECTION_SYSTEM, settings.writer_model, max_words=150, llm=llm)
        retries_total += r
        removed_all += rem
    else:
        text = "No usable evidence could be retrieved for this brief. See Gaps & Limitations."
    parts.append("## Executive Summary\n\n" + expand_markers(text, all_items) + "\n")

    # ---- thematic sections (one isolated LLM call each)
    for sec in bundle.sections:
        emit(run_id, AGENT, f"writing section: {sec.theme} ({len(sec.evidence)} evidence items)")
        parts.append(f"## {sec.theme}\n\n**Confidence: {sec.confidence}**\n")
        if not sec.evidence:
            parts.append("No usable evidence was found for this section.\n")
            continue
        ev = {i.id: i for i in sec.evidence}
        prompt = f"Section theme: {sec.theme}\n\nEvidence items:\n" + "\n".join(render_item(i) for i in sec.evidence)
        add_metric(run_id, "agent_prompt_tokens", {"agent": "writer_section", "sub_task": sec.sub_task_id,
                   "tokens": count_tokens(prompt + SECTION_SYSTEM)}, "append")
        text, r, rem = write_grounded(prompt, ev, SECTION_SYSTEM, settings.writer_model, llm=llm)
        retries_total += r
        removed_all += rem
        if r:
            emit(run_id, AGENT, f"section '{sec.theme}': {r} grounding retr{'y' if r == 1 else 'ies'}", level="warn")
        parts.append(expand_markers(text, ev) + "\n")

    # ---- gaps & limitations (deterministic)
    gaps = list(bundle.gaps) + [f"Conflicting information: {c}" for c in bundle.conflicts]
    if removed_all:
        gaps.append(f"{len(removed_all)} generated statement(s) were removed because they could not be tied to the evidence.")
    if not gaps:
        gaps = ["No major gaps were detected, but findings reflect only the public sources retrieved at run time."]
    parts.append("## Gaps & Limitations\n\n" + "\n".join(f"- {g}" for g in gaps) + "\n")

    markdown = "\n".join(parts)

    # ---- citation map for the UI drawer (only ids actually cited)
    cidx = _claim_index(store, bundle)
    used = set(re.findall(r"\(cite:(E\d+)\)", markdown))
    citations = {}
    for eid in used:
        it = all_items[eid]
        src = store.read(it.source_id)
        citations[eid] = {
            "id": eid, "source_id": it.source_id, "claim": it.claim, "url": it.url, "domain": it.domain, "title": src.title,
            "date": it.date, "credibility": it.credibility, "confidence": it.confidence,
            "paragraph": _paragraph_for(it, cidx), "conflict": it.conflict, "compressed": it.compressed,
        }
    add_metric(run_id, "writer_retries", retries_total)
    add_metric(run_id, "writer_statements_removed", len(removed_all))
    return WriterResult(markdown, citations, retries_total, removed_all)
