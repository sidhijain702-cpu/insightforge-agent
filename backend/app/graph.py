"""LangGraph state machine:

    PLANNING -> RESEARCHING -> SYNTHESIZING -> WRITING -> FACT_CHECKING -> COMPLETE
                 \\-> (vague brief) NEEDS_CLARIFICATION -> END

LEARNING NOTE
-------------
* LangGraph owns the *order* of stages and the typed `RunState` passed between them.
* Each node does ONE stage and returns only the fields it changed (a partial update).
* The RESEARCHING node blocks until the Planner's subagents (and the fallback executor)
  have all reported back; the graph cannot reach SYNTHESIZING before that. That is the
  "wait for all parallel branches" requirement.
"""
import functools
import uuid

from langgraph.graph import END, StateGraph

from app import db, memory
from app.agents import planner
from app.agents.factchecker import fact_check
from app.agents.synthesis import synthesize
from app.agents.writer import write_report
from app.config import settings
from app.events import add_metric, emit, get_metrics, set_status
from app.schemas import RunState
from app.scratch_store import ScratchStore


def guarded(fn):
    """Turn any exception in a node into a FAILED run with a readable message."""
    @functools.wraps(fn)
    def inner(state: RunState):
        try:
            return fn(state)
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {str(e)[:300]}"
            emit(state.run_id, "Orchestrator", f"stage failed: {msg}", level="error")
            set_status(state.run_id, "FAILED", msg)
            raise
    return inner


@guarded
def plan_node(state: RunState) -> dict:
    set_status(state.run_id, "PLANNING")
    ctx = planner.plan(state.run_id, state.brief, state.prefs, state.allow_clarification)
    if ctx.question:
        db.x("UPDATE runs SET clarifying_question=? WHERE id=?", (ctx.question, state.run_id))
        set_status(state.run_id, "NEEDS_CLARIFICATION")
        emit(state.run_id, "Planner", f"brief too vague, asking: {ctx.question}", level="warn")
        return {"status": "NEEDS_CLARIFICATION", "clarifying_question": ctx.question}
    planner.persist_graph(state.run_id, ctx.graph)
    return {"task_graph": ctx.graph}


@guarded
def research_node(state: RunState) -> dict:
    set_status(state.run_id, "RESEARCHING")
    planner.run_research(state.run_id, state.task_graph, state.prefs)   # blocks until ALL branches report
    store = ScratchStore(state.run_id)
    refs, failed = {}, {}
    for st in state.task_graph.sub_tasks:
        refs[st.id] = store.refs_for(st.id)
        row = db.q1("SELECT status, detail FROM subtasks WHERE run_id=? AND id=?", (state.run_id, st.id))
        if row and row["status"] == "Failed":
            failed[st.id] = row["detail"] or "failed"
    if len(failed) == len(state.task_graph.sub_tasks):
        raise RuntimeError("every sub-task failed; no evidence could be gathered (check API keys / network)")
    return {"refs": refs, "failed_subtasks": failed}


@guarded
def synthesize_node(state: RunState) -> dict:
    set_status(state.run_id, "SYNTHESIZING")
    bundle = synthesize(state.run_id, state.brief, state.task_graph, state.failed_subtasks)
    return {"bundle": bundle}


@guarded
def write_node(state: RunState) -> dict:
    set_status(state.run_id, "WRITING")
    res = write_report(state.run_id, state.brief, state.bundle)
    return {"report_markdown": res.markdown, "citations": res.citations}


@guarded
def fact_check_node(state: RunState) -> dict:
    set_status(state.run_id, "FACT_CHECKING")
    md, fc = fact_check(state.run_id, state.report_markdown, state.citations)
    return {"report_markdown": md, "factcheck": fc}


@guarded
def finalize_node(state: RunState) -> dict:
    report_id = f"rpt_{uuid.uuid4().hex[:10]}"
    title = state.brief.strip().replace("\n", " ")[:90]
    db.x(
        "INSERT INTO reports(id, run_id, watch_id, title, brief, markdown, bundle, citations, factcheck, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (report_id, state.run_id, state.watch_id, title, state.brief, state.report_markdown,
         state.bundle.model_dump_json(), db.jdump(state.citations), state.factcheck.model_dump_json(), db.now()),
    )
    try:
        n = memory.index_report_claims(report_id, state.citations)
        memory.save_session_summary(state.run_id, state.brief, state.report_markdown)
        emit(state.run_id, "Memory", f"indexed {n} claims; saved compressed session summary")
    except Exception as e:  # noqa: BLE001
        emit(state.run_id, "Memory", f"memory update skipped: {e}", level="warn")
    db.x("UPDATE runs SET report_id=? WHERE id=?", (report_id, state.run_id))
    m = get_metrics(state.run_id)
    emit(state.run_id, "Orchestrator",
         f"raw tokens fetched: {m.get('raw_tokens_fetched', 0)} -> evidence bundle: {state.bundle.token_count} tokens")
    set_status(state.run_id, "COMPLETE")
    return {"status": "COMPLETE", "report_id": report_id}


def _after_plan(state: RunState) -> str:
    return "end" if state.status == "NEEDS_CLARIFICATION" else "research"


@functools.lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(RunState)
    g.add_node("plan", plan_node)
    g.add_node("research", research_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("write", write_node)
    g.add_node("fact_check", fact_check_node)
    g.add_node("finalize", finalize_node)
    g.set_entry_point("plan")
    g.add_conditional_edges("plan", _after_plan, {"end": END, "research": "research"})
    g.add_edge("research", "synthesize")
    g.add_edge("synthesize", "write")
    g.add_edge("write", "fact_check")
    g.add_edge("fact_check", "finalize")
    g.add_edge("finalize", END)
    return g.compile()
