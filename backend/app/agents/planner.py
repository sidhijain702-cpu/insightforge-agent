"""Planner: a DeepAgents ROOT agent (`create_deep_agent`) that plans, then delegates.

HOW IT WORKS (read this once, it explains the whole file)
---------------------------------------------------------
PHASE 1 - PLAN.   The root agent gets the brief + memory hints. It calls the built-in
                  `write_todos` tool, then our typed `submit_task_graph` tool (or
                  `ask_clarifying_question` if the brief is vague). `submit_task_graph`
                  validates with Pydantic and returns validation errors to the agent so it
                  can self-correct: that is how free text becomes a typed TaskGraph.

PHASE 2 - DELEGATE. A *fresh* invocation receives ONLY the task list (never the brief).
                  It spawns isolated subagents with DeepAgents' built-in `task` tool:
                  web-researcher / doc-reader / memory-researcher, then entity-extractor.
                  Each subagent only has the one tool it needs and only ever sees its own
                  sub-task. Tools return source IDS + one-liners, never page text.

BARRIER.          When the root agent finishes, LangGraph (graph.py) is still waiting in the
                  RESEARCHING node. We then verify every sub-task finished, and run any that
                  did not through the deterministic executor. LangGraph advances to
                  SYNTHESIZING only after ALL branches are accounted for.

Typed results never come from parsing LLM prose: tools write to the scratch store / DB,
and we read the state back from there.
"""
import inspect
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from app import db, memory
from app.agents import researchers as R
from app.config import settings
from app.events import add_metric, emit, set_subtask
from app.llm import call_json, get_chat
from app.schemas import SourcePrefs, SubTask, TaskGraph
from app.scratch_store import ScratchStore

log = logging.getLogger("insightforge.planner")
AGENT = "Planner"
DEFAULT_HORIZON = "last 18 months"

PLANNER_SYSTEM = """You are the Planner of InsightForge, a market-intelligence research system.
You decompose research briefs into independent sub-tasks and delegate each to isolated subagents.
Follow the instructions in each user message exactly. Be concise. Never write the report yourself."""

WEB_PROMPT = """You are a web research subagent. You are given ONE sub-task (sub_task_id, query, time_horizon).
Call the `web_research` tool EXACTLY ONCE with those values, then reply with the tool's result and nothing else."""
DOC_PROMPT = """You are a document-reader subagent. You are given ONE sub-task id.
Call `doc_research` EXACTLY ONCE with it, then reply with the tool's result and nothing else."""
MEM_PROMPT = """You are a memory subagent. You are given ONE sub-task id.
Call `memory_research` EXACTLY ONCE with it, then reply with the tool's result and nothing else."""
ENT_PROMPT = """You are an entity-extraction subagent. You are given ONE sub-task id.
Call `extract_entities` EXACTLY ONCE with it, then reply with the tool's result and nothing else.
Only call `read_source` if the tool result asks you to; never fetch sources of other sub-tasks."""


# ======================================================================= helpers
@dataclass
class PlanCtx:
    run_id: str
    brief: str
    prefs: SourcePrefs
    enabled: list[str]
    allow_clarification: bool = True
    graph: TaskGraph | None = None
    question: str | None = None
    errors: list[str] = field(default_factory=list)


def enabled_source_types(run_id: str, prefs: SourcePrefs) -> list[str]:
    out = []
    if prefs.web:
        out.append("web")
    up = Path(settings.uploads_root) / run_id
    if prefs.docs and up.exists() and any(up.iterdir()):
        out.append("uploaded_docs")
    if prefs.prior and db.q1("SELECT 1 AS x FROM reports LIMIT 1"):
        out.append("prior_reports")
    return out


def clean_graph(raw: list[dict], enabled: list[str]) -> TaskGraph:
    """Validate + normalise what the LLM proposed. Raises ValueError with a message the agent can act on."""
    if not enabled:
        raise ValueError("no source types are enabled for this run")
    subs: list[SubTask] = []
    for i, r in enumerate(raw, 1):
        if not isinstance(r, dict) or not str(r.get("query", "")).strip():
            raise ValueError(f"sub-task #{i} has no 'query'")
        st_type = r.get("source_type", "web")
        if st_type not in enabled:
            st_type = "web" if "web" in enabled else enabled[0]
        try:
            pr = int(r.get("priority", 3))
        except (TypeError, ValueError):
            pr = 3
        subs.append(SubTask(
            id=f"t{i}", query=str(r["query"]).strip(), source_type=st_type,
            time_horizon=str(r.get("time_horizon") or DEFAULT_HORIZON), priority=min(5, max(1, pr)),
            title=str(r.get("title") or "").strip()[:80],
        ))
    if not subs:
        raise ValueError("provide at least 1 sub-task")
    if len(subs) > settings.max_parallel_subtasks:   # configurable cap on parallelism / cost
        subs = sorted(subs, key=lambda s: -s.priority)[: settings.max_parallel_subtasks]
        subs = [s.model_copy(update={"id": f"t{i}"}) for i, s in enumerate(subs, 1)]
    return TaskGraph(sub_tasks=subs)


def build_delegation_message(graph: TaskGraph) -> str:
    """PHASE 2 prompt. Deliberately contains NO brief text: only the sub-tasks themselves."""
    lines = [
        f"- sub_task_id={s.id} | type={s.source_type} | time_horizon={s.time_horizon} | query={s.query}"
        for s in graph.sub_tasks
    ]
    return (
        "PHASE 2 - DELEGATE. Do not plan again. Sub-tasks:\n" + "\n".join(lines) + "\n\n"
        "Step 1: call write_todos listing every sub_task_id as pending.\n"
        "Step 2: in ONE assistant message, issue one `task` call per sub-task so they run in parallel. "
        "Use subagent_type = web-researcher (type=web), doc-reader (type=uploaded_docs) or "
        "memory-researcher (type=prior_reports). The task description MUST contain only that one sub-task's "
        "sub_task_id, query and time_horizon - nothing about the other sub-tasks.\n"
        "Step 3: when results return, in ONE message issue one `task` call per sub-task with subagent_type="
        "entity-extractor and description 'sub_task_id: <id>' (skip sub-tasks that found no sources).\n"
        "Step 4: call write_todos marking completed sub-tasks, then reply 'DONE'."
    )


# ======================================================================= DeepAgents wiring
def _make_tools(ctx: PlanCtx):
    from langchain_core.tools import tool

    def lookup(sub_task_id: str) -> SubTask | None:
        return next((s for s in (ctx.graph.sub_tasks if ctx.graph else []) if s.id == sub_task_id.strip()), None)

    def audit(tool_name: str, sub_task_id: str) -> None:
        add_metric(ctx.run_id, "subagent_calls", {"tool": tool_name, "sub_task_id": sub_task_id}, "append")

    @tool
    def submit_task_graph(sub_tasks: list[dict]) -> str:
        """Submit the final decomposition (call exactly once). Each item needs: title (short heading),
        query (self-contained search query), source_type (web | uploaded_docs | prior_reports),
        time_horizon (e.g. 'last 18 months'), priority (1-5)."""
        try:
            ctx.graph = clean_graph(sub_tasks, ctx.enabled)
        except (ValueError, ValidationError) as e:
            return f"REJECTED: {e}. Fix and call submit_task_graph again."
        return f"ACCEPTED: {len(ctx.graph.sub_tasks)} sub-tasks. Phase 1 complete; stop now."

    @tool
    def ask_clarifying_question(question: str) -> str:
        """Ask ONE short clarifying question. Use only when the brief is too vague to plan any searches."""
        if not ctx.allow_clarification:
            return "Clarification is not allowed for this run: make reasonable assumptions and plan."
        ctx.question = question.strip()
        return "Question recorded. Stop now."

    @tool
    def web_research(sub_task_id: str, query: str = "", time_horizon: str = "") -> str:
        """Search the web for ONE sub-task, fetch full articles, score credibility, store them.
        Returns source ids + one-line summaries only (never article text)."""
        st = lookup(sub_task_id)
        if not st:
            return f"ERROR: unknown sub_task_id {sub_task_id}"
        audit("web_research", st.id)
        return json.dumps([r.model_dump() for r in R.run_web_research(ctx.run_id, st)])

    @tool
    def doc_research(sub_task_id: str, query: str = "") -> str:
        """Semantic search over the user's uploaded PDFs for ONE sub-task. Returns chunk ids + one-liners."""
        st = lookup(sub_task_id)
        if not st:
            return f"ERROR: unknown sub_task_id {sub_task_id}"
        audit("doc_research", st.id)
        return json.dumps([r.model_dump() for r in R.run_doc_research(ctx.run_id, st)])

    @tool
    def memory_research(sub_task_id: str, query: str = "") -> str:
        """Recall findings from PRIOR reports for ONE sub-task. Returns ids + one-liners."""
        st = lookup(sub_task_id)
        if not st:
            return f"ERROR: unknown sub_task_id {sub_task_id}"
        audit("memory_research", st.id)
        return json.dumps([r.model_dump() for r in R.run_memory_research(ctx.run_id, st)])

    @tool
    def extract_entities(sub_task_id: str) -> str:
        """Extract entities, claims and relationships from the sources of ONE sub-task. Returns counts."""
        st = lookup(sub_task_id)
        if not st:
            return f"ERROR: unknown sub_task_id {sub_task_id}"
        audit("extract_entities", st.id)
        return json.dumps(R.run_entity_extraction(ctx.run_id, st.id, st.query))

    @tool
    def read_source(source_id: str) -> str:
        """Read one stored source by id (first 4000 chars). Use sparingly."""
        try:
            return ScratchStore(ctx.run_id).read_source(source_id)
        except KeyError:
            return f"ERROR: unknown source id {source_id}"

    return submit_task_graph, ask_clarifying_question, web_research, doc_research, memory_research, extract_entities, read_source


def build_planner_agent(ctx: PlanCtx):
    """Create the DeepAgents root agent. Written defensively because the library is young."""
    from deepagents import create_deep_agent

    submit, ask, web, doc, mem, ext, read = _make_tools(ctx)

    def sub(name: str, desc: str, prompt: str, tools: list) -> dict:
        # both key spellings: deepagents renamed `prompt` -> `system_prompt` between releases
        return {"name": name, "description": desc, "system_prompt": prompt, "prompt": prompt, "tools": tools}

    subagents = [
        sub("web-researcher", "Searches the web for ONE sub-task. Give it only sub_task_id, query, time_horizon.", WEB_PROMPT, [web]),
        sub("doc-reader", "Searches uploaded PDFs for ONE sub-task. Give it only the sub_task_id.", DOC_PROMPT, [doc]),
        sub("memory-researcher", "Recalls prior-report findings for ONE sub-task. Give it only the sub_task_id.", MEM_PROMPT, [mem]),
        sub("entity-extractor", "Extracts entities/claims for ONE sub-task's stored sources. Give it only 'sub_task_id: <id>'.", ENT_PROMPT, [ext, read]),
    ]
    params = inspect.signature(create_deep_agent).parameters
    kwargs: dict = {"tools": [submit, ask], "subagents": subagents, "model": get_chat(settings.planner_model, 0.0, sdk_retries=8)}
    if "system_prompt" in params:
        kwargs["system_prompt"] = PLANNER_SYSTEM
    elif "instructions" in params:
        kwargs["instructions"] = PLANNER_SYSTEM
    if "summarization" in params:  # older releases expose this flag; newer ones summarise automatically
        kwargs["summarization"] = True
    return create_deep_agent(**kwargs)


def _invoke(agent, message: str, limit: int = 60):
    return agent.invoke({"messages": [{"role": "user", "content": message}]}, config={"recursion_limit": limit})


# ======================================================================= PHASE 1: plan
class _PlanOut(BaseModel):
    needs_clarification: bool = False
    question: str = ""
    sub_tasks: list[dict] = Field(default_factory=list)


PLAN_RULES = (
    "Decompose the brief into 3-{n} INDEPENDENT sub-tasks that can run in parallel. For each: title (<=6 words), "
    "query (a self-contained web search query that makes sense without the brief), source_type (one of {types}), "
    "time_horizon (e.g. 'last 18 months'), priority (1-5, 5 = most important). "
    "If the brief is too vague to choose searches (no identifiable industry, company or topic, or no clear goal), "
    "ask ONE clarifying question instead."
)


def _plan_direct(ctx: PlanCtx, memory_ctx: str) -> None:
    emit(ctx.run_id, AGENT, "planning with direct structured LLM call")
    no_ask = "" if ctx.allow_clarification else "\nNever ask questions: make reasonable assumptions and plan."
    prompt = (
        PLAN_RULES.format(n=settings.max_parallel_subtasks, types=ctx.enabled) + no_ask + memory_ctx
        + f"\n\nBRIEF: {ctx.brief}\n\nReturn JSON: "
        '{"needs_clarification": false, "question": "", "sub_tasks": [{"title":"","query":"","source_type":"web","time_horizon":"last 18 months","priority":3}]}'
    )
    out = call_json(prompt, _PlanOut, system=PLANNER_SYSTEM, model=settings.planner_model)
    if out.needs_clarification and out.question.strip() and ctx.allow_clarification:
        ctx.question = out.question.strip()
        return
    ctx.graph = clean_graph(out.sub_tasks, ctx.enabled)


def _plan_deepagent(ctx: PlanCtx, memory_ctx: str) -> None:
    emit(ctx.run_id, AGENT, "root DeepAgent planning (write_todos -> submit_task_graph)")
    agent = build_planner_agent(ctx)
    msg = (
        "PHASE 1 - PLAN ONLY. Do NOT delegate or research yet.\n"
        + PLAN_RULES.format(n=settings.max_parallel_subtasks, types=ctx.enabled) + memory_ctx
        + f"\n\nBRIEF: {ctx.brief}\n\n"
        + ("If vague: call ask_clarifying_question once and stop. " if ctx.allow_clarification else "")
        + "Otherwise: first call write_todos with the planned sub-tasks (all pending), then call "
        "submit_task_graph exactly once. If it replies REJECTED, fix the problem and call it again."
    )
    _invoke(agent, msg, limit=30)


def plan(run_id: str, brief: str, prefs: SourcePrefs, allow_clarification: bool = True) -> PlanCtx:
    ctx = PlanCtx(run_id, brief, prefs, enabled_source_types(run_id, prefs), allow_clarification)

    if allow_clarification and len(brief.split()) < 4:   # cheap pre-check before spending any tokens
        ctx.question = ("Which industry, market or companies should I research, and what do you most want to learn "
                        "(e.g. pricing, funding, product gaps, market size)?")
        return ctx

    # memory compaction: ONLY the last 3 compressed summaries, never transcripts
    sums = memory.recent_summaries(settings.session_summaries_to_load)
    ents = memory.related_entities(brief)
    memory_ctx = ""
    if sums:
        memory_ctx += "\n\nMemory (last sessions, compressed):\n" + "\n".join(f"- {s}" for s in sums)
    if ents:
        memory_ctx += "\nEntities seen in earlier research that may be relevant: " + ", ".join(ents)
    emit(run_id, AGENT, f"loaded {len(sums)} session summaries, {len(ents)} related entities from memory")

    if settings.planner_mode == "deepagent":
        try:
            _plan_deepagent(ctx, memory_ctx)
        except Exception as e:  # noqa: BLE001
            emit(run_id, AGENT, f"DeepAgent planning failed ({type(e).__name__}: {str(e)[:500]}); falling back", level="warn")
            add_metric(run_id, "planner_fallback", 1)
    if ctx.graph is None and ctx.question is None:
        _plan_direct(ctx, memory_ctx)
    return ctx


def persist_graph(run_id: str, graph: TaskGraph) -> None:
    """Log the task graph so the UI can render the live progress rows."""
    db.x("UPDATE runs SET task_graph=? WHERE id=?", (graph.model_dump_json(), run_id))
    for s in graph.sub_tasks:
        db.x(
            "INSERT OR REPLACE INTO subtasks(run_id,id,title,query,source_type,time_horizon,priority,status,detail) "
            "VALUES (?,?,?,?,?,?,?,'Queued','')",
            (run_id, s.id, s.theme(), s.query, s.source_type, s.time_horizon, s.priority),
        )
    emit(run_id, AGENT, f"task graph: {len(graph.sub_tasks)} sub-tasks " + ", ".join(f"{s.id}:{s.theme()}" for s in graph.sub_tasks))


# ======================================================================= PHASE 2: delegate + barrier
def _status_of(run_id: str, sub_id: str) -> str:
    row = db.q1("SELECT status FROM subtasks WHERE run_id=? AND id=?", (run_id, sub_id))
    return row["status"] if row else "Queued"


def run_research(run_id: str, graph: TaskGraph, prefs: SourcePrefs) -> None:
    """Delegate via the DeepAgents `task` tool, then guarantee every branch finished."""
    ctx = PlanCtx(run_id, "", prefs, enabled_source_types(run_id, prefs), graph=graph)   # brief intentionally empty
    if settings.planner_mode == "deepagent":
        try:
            emit(run_id, AGENT, f"delegating {len(graph.sub_tasks)} sub-tasks to isolated subagents via `task`")
            _invoke(build_planner_agent(ctx), build_delegation_message(graph), limit=80)
        except Exception as e:  # noqa: BLE001
            emit(run_id, AGENT, f"delegation error ({type(e).__name__}: {str(e)[:700]})", level="warn")

    # ---- barrier: every sub-task must be Done/Failed before LangGraph may advance
    store = ScratchStore(run_id)
    todo = []
    for st in graph.sub_tasks:
        if _status_of(run_id, st.id) in ("Done", "Failed"):
            continue
        todo.append(st)
    if todo:
        emit(run_id, AGENT, f"fallback executor finishing {len(todo)} sub-task(s) the subagents did not complete", level="warn")
        add_metric(run_id, "fallback_subtasks", len(todo))

        def finish(st: SubTask) -> None:
            if store.refs_for(st.id) and store.read_extraction(st.id) is None:
                R.run_entity_extraction(run_id, st.id, st.query)       # sources exist: only extraction missing
            else:
                R.run_subtask_direct(run_id, st)

        with ThreadPoolExecutor(max_workers=settings.max_parallel_subtasks) as pool:
            list(pool.map(finish, todo))

    for st in graph.sub_tasks:   # a sub-task left half-done is a failure, not silence
        if _status_of(run_id, st.id) not in ("Done", "Failed"):
            set_subtask(run_id, st.id, "Failed", "did not complete")
