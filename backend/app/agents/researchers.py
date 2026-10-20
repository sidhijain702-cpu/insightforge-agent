"""The Researcher pool: Web Search, Doc Reader, Memory, Entity Extractor.

LEARNING NOTE - why are these plain functions and not "LLM agents"?
The expensive, unreliable part of research is NOT the searching/fetching/scoring; it
is the LLM deciding what to do. So each researcher is a deterministic function (the
"tool"), and the DeepAgents *subagent* wrapped around it (see planner.py) only has to
make one tool call. This keeps subagents tiny and cheap, and it means the same
functions double as a fallback executor if the LLM tool-calling misbehaves.

Contract for EVERY researcher: write raw content to the scratch store, return only
[SourceRef(source_id, one-line summary)].
"""
import re
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from app import vector
from app.config import settings
from app.embeddings import embed
from app.entities import flag_conflicts, index_entities, norm
from app.events import add_metric, emit, set_subtask
from app.llm import call_json
from app.schemas import Claim, Entity, EntityGraph, ExtractionResult, Relationship, SourceRef, SubTask
from app.scratch_store import ScratchStore, domain_of
from app.tokens import count_tokens
from app.tools.credibility import score_source
from app.tools.fetch import fetch_article
from app.tools.search_providers import web_search

WEB = "Web Search Agent"
DOC = "Doc Reader Agent"
MEM = "Memory Agent"
ENT = "Entity Extractor Agent"


def _first_sentence(text: str, limit: int = 160) -> str:
    s = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    return s[:limit]


# =========================================================================== web
def run_web_research(run_id: str, st: SubTask) -> list[SourceRef]:
    store = ScratchStore(run_id)
    set_subtask(run_id, st.id, "Running", "searching the web")
    emit(run_id, WEB, f"[{st.id}] searching: {st.query!r} ({st.time_horizon})")
    want = settings.results_per_subtask
    try:
        results = web_search(st.query, want + 3, st.time_horizon)
    except Exception as e:  # noqa: BLE001
        emit(run_id, WEB, f"[{st.id}] search failed: {e}", level="error")
        set_subtask(run_id, st.id, "Failed", f"search error: {str(e)[:80]}")
        return []

    def work(r: dict):
        emit(run_id, WEB, f"[{st.id}] fetching {domain_of(r['url'])}…")
        return r, fetch_article(r["url"])

    with ThreadPoolExecutor(max_workers=min(len(results), want + 3) or 1) as pool:
        fetched = list(pool.map(work, results))  # keeps search-rank order

    refs: list[SourceRef] = []
    for r, f in fetched:
        if not f.ok:
            store.add_skipped(r["url"], f.skipped_reason)
            emit(run_id, WEB, f"[{st.id}] skipped {domain_of(r['url'])}: {f.skipped_reason}", level="warn")
            continue
        if len(refs) >= want:
            break
        date = f.date or r.get("published_date")
        domain = domain_of(r["url"])
        cred = score_source(domain, date, f.author)
        title = f.title or r["title"]
        ref = store.write_source(
            st.id, r["url"], summary=f"{title[:90]} — {_first_sentence(f.text)}",
            kind="web", title=title, published_date=date, author=f.author,
            body_text=f.text, credibility_score=cred,
        )
        toks = count_tokens(f.text)
        add_metric(run_id, "raw_tokens_fetched", toks, "incr")
        add_metric(run_id, "sources_fetched", 1, "incr")
        add_metric(run_id, "source_log", {"id": ref.source_id, "domain": domain, "tokens": toks}, "append")
        refs.append(ref)
    emit(run_id, WEB, f"[{st.id}] stored {len(refs)} sources in scratch store (returning ids only)")
    if not refs:
        set_subtask(run_id, st.id, "Failed", "no readable sources (all skipped)")
    else:
        set_subtask(run_id, st.id, "Running", f"{len(refs)} sources collected; extracting entities")
    return refs


# =========================================================================== docs
def run_doc_research(run_id: str, st: SubTask, k: int = 8) -> list[SourceRef]:
    store = ScratchStore(run_id)
    set_subtask(run_id, st.id, "Running", "searching uploaded documents")
    emit(run_id, DOC, f"[{st.id}] semantic search over uploaded PDFs (top-{k})")
    hits = vector.search("doc_chunks", embed([st.query])[0], limit=k, where={"run_id": run_id})
    refs = []
    for h in hits:
        p = h["payload"]
        ref = store.write_source(
            st.id, f"doc://{p['filename']}#page={p.get('page', 0)}",
            summary=f"{p['filename']} p.{p.get('page', '?')}: {p['text'][:120]}",
            kind="doc_chunk", title=f"{p['filename']} — {p.get('heading') or 'section'} (p.{p.get('page', '?')})",
            chunk_id=h["id"], published_date=p.get("uploaded_date"), credibility_score=0.8,
            meta={"relevance_score": round(h["score"], 3), "page": p.get("page"), "heading": p.get("heading")},
        )
        refs.append(ref)
    emit(run_id, DOC, f"[{st.id}] returning {len(refs)} chunk ids (text stays in Qdrant)")
    if not refs:
        set_subtask(run_id, st.id, "Failed", "no uploaded documents matched")
    else:
        set_subtask(run_id, st.id, "Running", f"{len(refs)} chunks retrieved; extracting entities")
    return refs


# =========================================================================== memory
def run_memory_research(run_id: str, st: SubTask, k: int = 8) -> list[SourceRef]:
    store = ScratchStore(run_id)
    set_subtask(run_id, st.id, "Running", "searching prior reports")
    emit(run_id, MEM, f"[{st.id}] recalling claims from prior reports")
    hits = [h for h in vector.search("report_claims", embed([st.query])[0], limit=k) if h["score"] > 0.35]
    refs = []
    for h in hits:
        p = h["payload"]
        text = p.get("paragraph") or p.get("claim", "")
        ref = store.write_source(
            st.id, p.get("url", "memory://prior-report"), summary=f"(prior report) {p.get('claim', '')[:140]}",
            kind="memory", title=p.get("title", "Prior report finding"), published_date=p.get("date"),
            body_text=text, credibility_score=float(p.get("credibility", 0.5)),
            meta={"from_report": p.get("report_id"), "relevance_score": round(h["score"], 3)},
        )
        refs.append(ref)
    if not refs:
        set_subtask(run_id, st.id, "Failed", "no relevant prior reports")
    else:
        set_subtask(run_id, st.id, "Running", f"{len(refs)} prior findings recalled; extracting entities")
    return refs


# =========================================================================== entity extraction
class _LLMClaim(BaseModel):
    source: str | int = ""
    text: str
    quote: str = ""
    entities: list[str] = Field(default_factory=list)
    relevant: bool = True


class _LLMRel(BaseModel):
    subject: str
    predicate: str = "other"
    object: str
    source: str | int = ""
    date: str | None = None


class _LLMExtraction(BaseModel):
    claims: list[_LLMClaim] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[_LLMRel] = Field(default_factory=list)


EXTRACT_SYSTEM = """You extract structured facts from source excerpts for a market-intelligence team.
Rules: only facts stated in the excerpts; never add outside knowledge; every claim must be a
self-contained sentence that names its entities explicitly (no pronouns).
Entity types: company, person, product, funding_round, money, date, region.
Relationship predicates (use exactly one): raised, valued_at, founded_in, headquartered_in, acquired,
launched, priced_at, partnered_with, employs, operates_in, competes_with, revenue, other."""


def split_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if len(p.strip()) >= 60]
    if len(paras) < 3:  # one giant block: cut into ~3-sentence groups
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        paras = [" ".join(sents[i : i + 3]) for i in range(0, len(sents), 3) if sents[i : i + 3]]
    return paras


def select_relevant_paragraphs(text: str, query: str, max_chars: int) -> list[str]:
    """Context engineering in miniature: give the LLM the paragraphs most relevant to the
    sub-task query (up to max_chars), in original order, instead of the whole page."""
    paras = split_paragraphs(text)
    if sum(len(p) for p in paras) <= max_chars:
        return paras
    vecs = embed([query] + paras)
    sims = vecs[1:] @ vecs[0]
    chosen, used = [], 0
    for i in sorted(range(len(paras)), key=lambda i: -sims[i]):
        if used + len(paras[i]) > max_chars:
            continue
        chosen.append(i)
        used += len(paras[i])
    return [paras[i] for i in sorted(chosen)]


def _locate_paragraph(quote: str, claim: str, paras: list[str]) -> str:
    """Find the excerpt paragraph a claim came from (verbatim match first, embeddings second)."""
    if not paras:
        return quote
    nq = norm(quote)[:70]
    if nq:
        for p in paras:
            if nq in norm(p):
                return p
    vecs = embed([claim] + paras)
    return paras[int((vecs[1:] @ vecs[0]).argmax())]


def run_entity_extraction(run_id: str, sub_task_id: str, query: str) -> dict:
    """Extract claims + entity graph for ONE sub-task from ITS OWN sources only."""
    store = ScratchStore(run_id)
    refs = store.refs_for(sub_task_id)
    if not refs:
        set_subtask(run_id, sub_task_id, "Failed", "no sources to extract from")
        return {"sub_task_id": sub_task_id, "claims": 0, "entities": 0, "conflicts": 0}

    emit(run_id, ENT, f"[{sub_task_id}] extracting entities/claims from {len(refs)} sources")
    blocks, para_map, src_map = [], {}, {}
    for i, ref in enumerate(refs, 1):
        src = store.read(ref.source_id)
        paras = select_relevant_paragraphs(src.body_text, query, settings.entity_chars_per_source)
        label = f"S{i}"
        para_map[label], src_map[label] = paras, src
        blocks.append(f"[{label}] {src.title} ({src.domain}, {src.published_date or 'n.d.'})\n" + "\n".join(paras))
    prompt = (
        f"Sub-task: {query}\nExtract up to {settings.claims_per_source} important, specific claims PER source "
        "(prefer numbers, dates, named companies, funding, pricing, product facts).\n"
        "RELEVANCE RULE: a claim must be about this sub-task's industry AND region. Set relevant=false for anything "
        "about other regions, other industries, or generic totals (e.g. all-startup funding, a different continent's market). "
        "If a source is mostly off-topic, extract little or nothing from it.\n"
        'Return JSON: {"claims":[{"source":"S1","text":"...","quote":"verbatim supporting sentence",'
        '"relevant":true,"entities":["..."]}],"entities":[{"name":"...","type":"company"}],'
        '"relationships":[{"subject":"...","predicate":"raised","object":"$20M","source":"S1","date":"2025-03"}]}\n\n'
        + "\n\n".join(blocks)
    )
    add_metric(run_id, "agent_prompt_tokens", {"agent": "entity_extractor", "sub_task": sub_task_id,
               "tokens": count_tokens(prompt + EXTRACT_SYSTEM)}, "append")
    try:
        out = call_json(prompt, _LLMExtraction, system=EXTRACT_SYSTEM, model=settings.fast_model)
    except Exception as e:  # noqa: BLE001
        emit(run_id, ENT, f"[{sub_task_id}] extraction failed: {e}", level="error")
        set_subtask(run_id, sub_task_id, "Failed", "entity extraction failed")
        return {"sub_task_id": sub_task_id, "claims": 0, "entities": 0, "conflicts": 0}

    claims: list[Claim] = []
    for n, c in enumerate(out.claims):
        label = str(c.source).strip().lstrip("S").lstrip("s")
        label = f"S{label}"
        if label not in src_map or len(c.text) < 20 or not c.relevant:
            continue
        claims.append(Claim(
            id=f"clm_{sub_task_id}_{n}", source_id=src_map[label].id, sub_task_id=sub_task_id,
            text=c.text.strip(), paragraph=_locate_paragraph(c.quote, c.text, para_map[label]),
            entities=[e for e in c.entities if e],
        ))
    rels = []
    for r in out.relationships:
        label = f"S{str(r.source).strip().lstrip('Ss')}"
        url = src_map[label].url if label in src_map else ""
        rels.append(Relationship(subject=r.subject, predicate=r.predicate, object=r.object, source_url=url, date=r.date))
    rels = flag_conflicts(rels)
    graph = EntityGraph(entities=out.entities, relationships=rels)
    store.write_extraction(ExtractionResult(sub_task_id=sub_task_id, claims=claims, graph=graph))
    try:
        index_entities(run_id, out.entities)
    except Exception as e:  # noqa: BLE001
        emit(run_id, ENT, f"entity indexing skipped: {e}", level="warn")

    n_conf = sum(1 for r in rels if r.conflict)
    emit(run_id, ENT, f"[{sub_task_id}] {len(claims)} claims, {len(out.entities)} entities, {n_conf} conflicting relationships")
    if claims:
        set_subtask(run_id, sub_task_id, "Done", f"{len(refs)} sources, {len(claims)} claims")
    else:
        set_subtask(run_id, sub_task_id, "Failed", "no claims could be extracted")
    return {"sub_task_id": sub_task_id, "claims": len(claims), "entities": len(out.entities), "conflicts": n_conf}


# =========================================================================== dispatcher
def run_researcher(run_id: str, st: SubTask) -> list[SourceRef]:
    if st.source_type == "uploaded_docs":
        return run_doc_research(run_id, st)
    if st.source_type == "prior_reports":
        return run_memory_research(run_id, st)
    return run_web_research(run_id, st)


def run_subtask_direct(run_id: str, st: SubTask) -> None:
    """Deterministic executor for one sub-task (used as fallback if the LLM subagent path fails)."""
    refs = run_researcher(run_id, st)
    if refs:
        run_entity_extraction(run_id, st.id, st.query)
