# Learning Guide: how this project works, concept by concept

Read in this order. Each section says **what** the idea is, **where** it lives, and gives a small **exercise** so you learn by touching the code. You will be asked about these in a review or interview.

---

## 1. Context engineering: the one idea behind the whole project
**What:** An LLM only knows what is in its prompt, and prompts have limits (size, cost, rate limits, and attention). A research run reads 30+ web pages. Pasting them all into one prompt fails. So *no agent sees more than it needs.*

**The trick: pass pointers, not content.** A researcher saves the page in a store and returns `{source_id, one-line summary}`. Whoever needs the text later asks for that one id.

**Where:** `app/scratch_store.py` (read the file docstring), `app/schemas.py` (`Source` heavy vs `SourceRef` light).

**Exercise:** Run a brief, then `python -m scripts.context_report <run_id>`. Compare "raw tokens fetched" with "largest prompt any agent saw". That ratio *is* the value of this design.

## 2. Typed state between agents
**What:** Every handoff is a Pydantic model (`TaskGraph`, `SourceRef`, `EvidenceBundle`, `RunState`). If an agent emits garbage, validation fails *at the boundary* with a clear message instead of corrupting something three steps later.

**Where:** `app/schemas.py`; look at how `submit_task_graph` in `agents/planner.py` returns `REJECTED: ...` text so the LLM can fix its own output (a validation feedback loop).

**Exercise:** In `clean_graph`, change the cap and re-run `tests/test_planner.py`. Which test fails and why?

## 3. LangGraph: the state machine
**What:** A graph of nodes; each node does one stage and returns a *partial* update to a shared typed state. Edges define order; a conditional edge can end early (vague brief).

**Where:** `app/graph.py`. Notice `research_node` blocks until all sub-tasks report: that is the "wait for parallel branches" requirement.

**Exercise:** Add a new node `log_metrics` between `fact_check` and `finalize` that prints `state.factcheck.pass_rate`.

## 4. DeepAgents: a planner that spawns subagents
**What:** `create_deep_agent()` gives you an agent with built-in tools: `write_todos` (a task list it maintains) and `task` (spawn a *subagent* with its own isolated context and tools). The parent only sees the subagent's final answer.

**Our use (`agents/planner.py`):**
- PHASE 1: plan → tool `submit_task_graph` (validated, typed).
- PHASE 2: a fresh prompt with *only the task list* → the agent calls `task` once per sub-task → subagents each call exactly one tool.
- Why tools are deterministic Python (`agents/researchers.py`) and subagents just call them: LLMs are unreliable at orchestration, reliable at "call this one function". It also gives us a fallback executor.

**Exercise:** Set `PLANNER_MODE=direct` and run a brief. What changes in the logs? What do you lose?

## 5. Parallelism and the barrier
`task` calls issued in one assistant message run concurrently. After the agent finishes we check each sub-task's status in the DB; stragglers go to a thread-pool fallback (`run_research`). *Never trust an LLM to have done everything: verify state, then proceed.*

## 6. Credibility scoring
`app/tools/credibility.py`: `0.5·domain + 0.3·recency + 0.2·byline`. Simple and explainable beats clever and opaque. **Exercise:** change the recency half-life from 365 to 90 days; which test changes?

## 7. Embeddings and cosine similarity (used in 5 places)
An embedding maps text to a vector; similar meaning → small angle. We normalize vectors so cosine similarity is a dot product. Used for: choosing relevant paragraphs, de-duplicating claims, fact-checking, diffing reports, recalling memory. **Where:** `app/embeddings.py`. **Exercise:** in a Python shell, embed "Acme raised $20M" and "Acme secured twenty million dollars"; compare to an unrelated sentence.

## 8. Synthesis: ranking, confidence, and budget compression
`app/agents/synthesis.py` is the most important file. Follow `synthesize()` top to bottom:
1. resolve ids → 2. de-duplicate (same entity + near-identical claim) → 3. conflict flags → 4. confidence (HIGH = 2+ independent domains) → 5. rank (recency, credibility, relevance) → 6. `enforce_budget`: merge the *lowest-ranked* items, log every cut.

**Why compress instead of truncate?** Truncation silently drops whatever is last, which is arbitrary. Compression drops *detail*, keeps the fact, and leaves an audit trail.

**Exercise:** Run `tests/test_synthesis.py::test_token_budget...` with budget 450, then 1500. Read the compression log each time.

## 9. Grounded writing: how we stop hallucination
`app/agents/writer.py`. Three ideas:
1. **Narrow context:** one call per section, seeing only its own evidence.
2. **Cite by id:** the model writes `{E12}`; code expands it into the required `[Source: domain, date, credibility]`. The model can't mis-format a citation, and every citation is checkable.
3. **Verify, then retry with a constraint:** numbers and names in a sentence must appear in the cited evidence; otherwise the section is regenerated with the exact violations listed. After the last retry, offending sentences are *removed*, never shipped.

**Exercise:** In `tests/test_writer.py` read `test_retry_adds_explicit_constraint...`. Then change `writer_overlap_threshold` and see which statements start to fail.

## 10. Fact-checking with a vector DB
`app/agents/factchecker.py`: sample claims → re-fetch the source → embed its sentences into a *temporary* Qdrant collection → query with the claim → score ≥ 0.85 passes. The threshold is a dial: higher = fewer false passes, more false alarms. Look at `details[*].score` in a real run before touching it.

## 11. Streaming progress (SSE) without complexity
The pipeline runs in a thread and writes events to SQLite (`app/events.py`). The SSE endpoint (`main.py: research_status`) simply tails that table once a second. Benefits: survives reconnects, and the UI and CLI can share the same event log.

## 12. Scheduling and semantic diffing
`app/scheduler.py` + `agents/diff.py`. Reports are different text every time even when nothing happened. Diff works in two stages: cheap embedding match to find *candidates*, then an LLM judge that says new / changed / rephrase. Rephrase is dropped. **Exercise:** read `test_diff.py::test_new_finding_surfaces_and_rephrasing_does_not`.

---

## Debugging playbook (when things break on your machine)
| Symptom | Likely cause | Fix |
|---|---|---|
| `python -m scripts.smoke` fails at "tool-calling" | model can't call tools reliably | change `PLANNER_MODEL`; or `PLANNER_MODE=direct` |
| `413 / 429 / rate limit` in logs | Groq tokens-per-minute cap | lower `MAX_LLM_CONCURRENCY`, `RESULTS_PER_SUBTASK=3`; or upgrade tier |
| `tool_use_failed` from Groq | malformed tool call by the model | usually auto-handled by the fallback executor; check the log line "fallback executor finishing…" |
| `TypeError` in `create_deep_agent(...)` | library API changed | `pip show deepagents`; compare with its README; adjust `build_planner_agent` |
| Everything "Failed: no readable sources" | search key wrong, or sites blocking | run `python -m scripts.smoke -v` |
| Fact-check pass rate very low | threshold too strict for paraphrase | read scores in `details`; consider `FACTCHECK_THRESHOLD=0.80` and *document why* |
| PDF export looks plain | WeasyPrint system libs missing → fallback PDF | install pango (see WeasyPrint docs) |
