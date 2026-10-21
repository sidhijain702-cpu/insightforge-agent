# InsightForge Agent: Autonomous Market Intelligence Engine

Submit a plain-English research brief; get a structured, **cited** intelligence report in minutes. A watchlist re-runs topics on a schedule and alerts you only when something *materially* new appears.

Built with **DeepAgents** (planner + isolated subagents via the `task` tool), **LangGraph** (typed state machine), **Groq** (LLM), **Qdrant** (vectors), **FastAPI + SSE**, **APScheduler**, and **Next.js**.

> Status note for reviewers: items marked **TODO(you)** below need real numbers/screenshots from your own runs. They are deliberately left blank rather than invented.

---

## Architecture

```
 Next.js UI  ── brief · live SSE progress · report viewer + citation drawer · watchlist · export
      │
 FastAPI  ── /research/run · /research/status/{id} (SSE) · /watchlist · /alerts · /reports (+export, share)
      │
 LangGraph state machine (typed RunState between every node)
      │
 PLANNING ──► RESEARCHING ──► SYNTHESIZING ──► WRITING ──► FACT_CHECKING ──► COMPLETE
   │             │ (blocks until ALL branches report)
   │             ▼
   │      DeepAgents root Planner  ──`task`──►  web-researcher │ doc-reader │ memory-researcher
   │      (write_todos, typed task graph)       then entity-extractor      (isolated, one sub-task each)
   │             │ every researcher: raw text ──► SCRATCH STORE / Qdrant ; returns only {source_id, 1-line summary}
   ▼             ▼
 Memory store (last 3 compressed summaries + entity & claim embeddings in Qdrant)

 SYNTHESIS: resolve ids → de-duplicate → conflicts → confidence → rank → compress to token budget (logged)
 WRITER:    one isolated call per section, cites with {E#} markers, grounding check + retry
 FACT-CHECK: 20% sample (min 5) re-fetched, cosine ≥ 0.85 via Qdrant, failures marked [UNVERIFIED]

 Scheduler: APScheduler job per watchlist item → same pipeline → Diff Agent → alert (+ Slack webhook)
```

## Setup

**Prerequisites:** Python 3.11+, Node 18+. No Docker needed (Qdrant runs in local on-disk mode).

```bash
# 1. backend
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # then fill in GROQ_API_KEY and TAVILY_API_KEY
python -m scripts.smoke                                 # checks every dependency one by one; fix FAILs first
python -m pytest -q                                     # 40+ offline unit tests (no keys needed)

# 2. run a brief from the terminal (best way to debug)
python -m scripts.run_cli "Summarize the competitive landscape for B2B payroll software in Southeast Asia: pricing, funding, product gaps"

# 3. API + UI
uvicorn app.main:app --port 8000                        # do NOT use --reload (local Qdrant allows one process)
cd ../frontend && cp .env.local.example .env.local && npm install && npm run dev   # http://localhost:3000
```

### Required API keys

| Key | Used for | Where |
|---|---|---|
| `GROQ_API_KEY` | all LLM calls | console.groq.com |
| `TAVILY_API_KEY` | web search (or set `SEARCH_PROVIDER=brave` + `BRAVE_API_KEY`, or `serpapi` + `SERPAPI_API_KEY`) | tavily.com |

First run downloads the embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB) and, for PDFs, Docling's models.

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `PLANNER_MODEL` / `WRITER_MODEL` / `FAST_MODEL` | openai/gpt-oss-120b / same / openai/gpt-oss-20b.1-8b-instant | Groq models. Planner needs reliable tool calling. Verify current names at console.groq.com/docs/models |
| `SEARCH_PROVIDER` | tavily | tavily · brave · serpapi |
| `RESULTS_PER_SUBTASK` | 5 | full articles fetched per sub-task |
| `MAX_PARALLEL_SUBTASKS` | 6 | cap on parallel research branches |
| `EVIDENCE_TOKEN_BUDGET` | 6000 | max tokens in the evidence bundle given to the Writer |
| `ENTITY_CHARS_PER_SOURCE` | 1800 | query-relevant excerpt size per source for extraction |
| `CLAIMS_PER_SOURCE` | 4 | claims extracted per source |
| `MAX_LLM_CONCURRENCY` | 2 | simultaneous LLM calls (raise on paid Groq tiers) |
| `PLANNER_MODE` | deepagent | `direct` bypasses DeepAgents (debugging) |
| `FACTCHECK_THRESHOLD` / `_SAMPLE_RATE` / `_MIN_CLAIMS` | 0.85 / 0.2 / 5 | fact-checker settings |
| `EMBEDDINGS_BACKEND` | fastembed | `hash` is an offline test fallback only |
| `QDRANT_PATH` / `QDRANT_URL` | data/qdrant / empty | local on-disk, or a Qdrant server |
| `JWT_SECRET`, `SHARE_TTL_DAYS` | change-me, 7 | share-link signing and expiry |
| `FRONTEND_URL` | http://localhost:3000 | CORS + share-link base |
| `DEMO_INTERVAL_SECONDS` | 0 | >0 runs every watch item on this interval (use ~180 for the demo video) |

---

## Context engineering write-up

**Goal:** a run touches 30+ pages; no agent may ever hold more than its own slice.

### Scratch store design (`app/scratch_store.py`)
- Every fetched page / doc chunk / extraction result is written to a keyed store (`scratch/<run_id>/src_<random>.json`). Random ids, not a counter, because parallel subagents write simultaneously.
- Each source has a heavy file (full text) and a tiny `.ref.json` sidecar (`sub_task_id`, `source_id`, one-line summary). Researchers return only the ref. The `SourceRef` and `Source` types are separate on purpose (`schemas.py`).
- Doc chunks are **not** copied into the store: the store holds a `chunk_id`, and text is resolved from Qdrant only when Synthesis (or the on-demand `read_source` tool, capped at 4,000 chars) asks.
- Typed results never come from parsing LLM prose: tools write to the store/DB, and the orchestrator reads state back from there.

### Isolation
- The root Planner's PHASE 2 prompt contains **only the task list, never the brief** (`build_delegation_message`, covered by `tests/test_planner.py`).
- Each subagent has exactly one tool and sees one sub-task. The entity extractor only reads *its own* sub-task's sources, and only the paragraphs most similar to the sub-task query (≈1,800 chars per source), not whole pages.
- The Writer is map-reduce: one call per section seeing only that section's evidence.

### Token budget choices
- Default **6,000 tokens** for the evidence bundle (`EVIDENCE_TOKEN_BUDGET`): large enough for ~40 one-sentence claims with citations metadata, small enough to leave room for prompts + output inside Groq's per-minute token limits.
- Counting uses 1 token ≈ 4 characters (no tokenizer download); budget has ~10-15% headroom for that approximation.
- Over budget → **compress, never blind-truncate**: repeatedly merge the lowest-ranked items (rank = recency, then credibility, then relevance) of a section into one summarized item; if nothing is left to merge, shorten, then drop. Each step is logged (`compression_log`: section, items affected, source ids, reason, tokens before/after) and surfaced in the report's Gaps section and the UI.

### Memory compaction
- After every run a ≤80-word summary is stored; the Planner loads only the **last 3** (`SESSION_SUMMARIES_TO_LOAD`), never transcripts.
- DeepAgents' summarization is enabled if the installed version exposes the flag (otherwise the library's default applies).

### Concrete example from a real run: **TODO(you)**
After your first successful run:
```bash
python -m scripts.context_report <run_id>
```
It prints the measured numbers (raw tokens fetched, largest single prompt, bundle before → after compression) and a ready-to-edit paragraph. Paste it here with the run id and, ideally, a screenshot of the compression log from the report page.

---

## What the pipeline guarantees (and how it is tested)

| Requirement | Mechanism | Test |
|---|---|---|
| Typed boundaries | Pydantic models + LangGraph `RunState` | `schemas.py` |
| Parallel branches complete before synthesis | RESEARCHING node blocks; fallback executor finishes stragglers | `planner.run_research` |
| ≤ 6 parallel sub-tasks | `clean_graph` cap | `test_planner` |
| Token budget respected | `enforce_budget` | `test_synthesis` |
| Confidence HIGH/MEDIUM/LOW | 2+ independent domains / single credible source / otherwise or conflict | `test_synthesis` |
| Writer cannot invent facts | id, number, name, overlap checks + retry with explicit constraint | `test_writer` |
| Fact-check + `[UNVERIFIED]` | Qdrant cosine ≥ 0.85 on 20% sample | `test_factchecker` |
| Diff ignores rephrasing | embedding pre-filter + LLM judge | `test_diff` |
| 7-day signed share link | PyJWT | `test_export` |

## Walkthrough: one brief, input to PDF
1. Open http://localhost:3000, paste a brief (≤500 chars), choose sources, optionally drop PDFs. Click **Run research**.
2. Watch sub-task rows move *Queued → Running → Done*, with per-agent logs ("Web Search Agent: fetching reuters.com…").
3. When complete, click **Open report**. Click any `[Source: …]` to open the drawer with title, date, credibility and the exact source paragraph. Amber `[UNVERIFIED]` marks claims that failed the re-check.
4. **Download PDF** (WeasyPrint; falls back to a plain PDF if its system libraries are missing), **Download .md**, or **Copy share link**.
5. Add the topic on **Watchlist**; use **Run now** twice (or set `DEMO_INTERVAL_SECONDS`) to see a diff and alert.

**TODO(you):** attach one real generated report (e.g. `backend/out/<run_id>.md`) and its PDF under `samples/`.

## Evaluation
`python -m scripts.run_briefs && python -m scripts.evaluate` runs the 5 briefs in `backend/briefs/` and writes `EVALUATION.md` (fact-check pass rate, citation accuracy, structure checks, section coverage, LLM-judge scores, plus a manual-review table to fill in). **TODO(you):** run it and commit the generated file.

## Results and limitations
Results and limitations (from our own runs). Fact-check: 19 of 25 sampled claims verified (76%, below the 80% target; the sample is 5 claims per brief, so per-brief figures are noisy). No claim scored below 0.60 similarity; 3 of 6 failures were between 0.75 and 0.85, consistent with paraphrase, and one compound and one cross-source sentence cannot match a single page by design. Delegation: in 3 of 5 evaluation runs the DeepAgents subagents executed 7 task calls and a fallback finished 1 sub-task; in 2 of 5 (heavy Groq rate limiting) the deterministic fallback executed all sub-tasks. Evidence is thin: up to 125,838 raw tokens became about 600 tokens of claims, limited by the free-tier token cap and small extraction prompts. Compression was demonstrated by replaying saved claims with the budget lowered to 400 tokens; it did not trigger at the default 6,000.

## Results and limitations (from our own runs)

**Fact-check.** 19 of 25 sampled claims verified (76%), below the 80% target. The sample is 5 claims per brief, so one claim moves a brief by 20 points. No claim scored below 0.60 similarity; 3 of the 6 failures scored 0.75-0.85 (consistent with paraphrase); one was a compound sentence and one summarised two sources, which cannot match a single page.

**Context engineering example.** Run `3514005c7f44` (brief 05): the Web Search agents fetched 20 articles, about 125,838 tokens of raw text. In one prompt that is roughly 16x the 8,000 tokens-per-minute limit of the free Groq tier used here. Instead raw text stayed in the scratch store; the largest prompt any agent received was about 2,310 tokens (entity extractor), and Synthesis produced a 603-token evidence bundle. Compression did not trigger at the default 6,000-token budget, so it was demonstrated by replaying saved claims with the budget lowered to 400 (`python -m scripts.resynth 3514005c7f44 400`): 603 became 396 tokens in 4 logged steps.

**Delegation.** In briefs 01-03 the DeepAgents subagents made 7 tool calls each and a fallback executor finished 1 sub-task. In briefs 04-05, heavy Groq throttling meant the deterministic fallback ran all sub-tasks (logged as "fallback executor finishing...").

**Conflict handling on real data.** In one run, two sources disagreed on Employer-of-Record pricing ($199-$400 vs $300-$600 per employee per month). Both were flagged, the section confidence was LOW, and the Gaps section listed the disagreement.

**Known weaknesses.**
 - Evidence is thin: up to ~125,000 raw tokens become ~600 tokens of claims, limited by free-tier token limits and small extraction prompts.
 - Some sections are empty or off-topic (funding sections sometimes list adjacent fintech companies). A stricter category-level relevance rule was tested late; it caused 2 of 4 sub-tasks to extract nothing, so it was reverted. The evaluation was run with the looser rule.
 - Citation dates are page metadata and can differ from the date of the fact.
 - PDF export uses a plain fallback when WeasyPrint's system libraries are missing.
 - Free-tier Groq limits (8,000 tokens per minute, 200,000 per day on the model used) slow runs and trigger the fallback executor.
 - The embedding model loads with a time limit; if it cannot load, a weaker hash embedder is used and an error is logged.
 - PDF upload (Doc Reader) and Slack webhooks were implemented and unit-tested but not exercised end to end.

## Approach in brief
A plain-English brief is split by a DeepAgents planner into typed sub-tasks. Isolated subagents search and read sources in parallel; raw text is stored on disk and only ids travel between agents. A synthesis step ranks and compresses evidence to a token budget, a writer cites each sentence by evidence id and is checked against that evidence, and a fact-checker re-opens a sample of sources and flags claims it cannot verify. A scheduler re-runs watched topics and a diff agent alerts only on materially new findings.

## Key design decisions
- **Pass pointers, not content:** raw pages live in a scratch store; agents exchange ids and short claims, so no prompt holds the whole corpus.
- **Typed handoffs:** every boundary is a Pydantic model, so bad output fails loudly.
- **Deterministic tools under LLM subagents:** searching, fetching and scoring are plain functions; if subagent tool calling fails, a fallback executor runs the same functions (logged).
- **Cite by id, verify every sentence:** the writer cannot invent a citation, and numbers, names and ids are checked against the cited evidence before a sentence is kept.
- **Compress, never truncate:** over-budget evidence is merged lowest-rank first, with every step logged.
- **Time limits and fallbacks on every external call:** rate-limit retry for the LLM, and a timed embedding load with a logged fallback.

## Safety and guardrails
- Input validation: brief at most 500 characters, at most 10 PDFs of 20 MB each, PDF only, at least one source enabled (otherwise HTTP 400/413).
- Vague briefs (under 4 words) trigger a clarifying question before any model call.
- Paywalled or blocked pages are skipped and logged; skipped counts appear in the report's Gaps section.
- Share links are signed (JWT), expire after 7 days, and are read-only; tampered links return 403 and expired links 410 (unit-tested).
- No secrets in the repo: keys come from `.env` (git-ignored); only `.env.example` is committed.
- Unsupported claims are removed by the writer check or flagged `[UNVERIFIED]` by the fact-checker.
- Not implemented: user accounts or API authentication (the API is open on localhost), so a deployment would need auth, a Qdrant server and multiple workers.

## How to evaluate quickly
```bash
cd backend
python -m pytest -q                 # 46 offline unit tests, no API keys needed
python -m scripts.smoke             # checks Groq, search, embeddings, Qdrant, DeepAgents
python -m scripts.run_cli "Summarize the competitive landscape for B2B payroll software in Southeast Asia. Focus on pricing, recent funding, and product gaps."
python -m scripts.run_briefs && python -m scripts.evaluate    # the 5 sample briefs, writes EVALUATION.md
```
Edge cases to try in the UI: a brief over 500 characters, a non-PDF upload, a 2-word brief ("payroll software") to see the clarifying question, and a tampered share link. Included evidence: `EVALUATION.md` (metrics and manual review), `samples/` (a generated report), `docs/watchlist-alert.png` (a real diff alert).

## Known limitations
- Fact-check threshold 0.85 is strict for paraphrased sentences; inspect `factcheck.details[*].score` and tune `FACTCHECK_THRESHOLD` deliberately, don't just lower it.
- Compressed evidence items merge several sources under one citation (the strongest source), so they are capped at MEDIUM confidence.
- The Diff Agent compares reports of the same topic; day-to-day variation in search results can surface "new" facts that are merely newly *found*.
- Conflict detection covers single-valued predicates (funding, valuation, founding year, HQ, price, revenue, headcount).
- Local Qdrant allows one server process. Use `QDRANT_URL` for multi-worker deployments.
- DeepAgents is a young library; `planner.py` adapts to both `system_prompt`/`instructions` parameter names, and falls back to a deterministic executor if tool calling fails.

## Code walkthrough
docs/CODE_WALKTHROUGH.md explains every module with the real code, the design reasons, the failures we hit, and likely review questions.


## Layout
```
backend/app/        agents/ (planner, researchers, synthesis, writer, factchecker, diff)  tools/  graph.py  pipeline.py
                    scratch_store.py  vector.py  embeddings.py  memory.py  scheduler.py  export.py  main.py  schemas.py
backend/briefs/     5 sample briefs + expected sections
backend/scripts/    smoke · run_cli · run_briefs · evaluate · context_report
backend/tests/      offline unit tests
frontend/           Next.js 14 App Router UI
docs/               DEMO_SCRIPT.md
```

## Submission checklist
- [ ] Public GitHub repo named `insightforge-agent`; `.env` NOT committed
- [ ] README TODOs filled (context example, sample report, screenshot)
- [ ] `EVALUATION.md` generated from real runs
- [ ] 5–7 min demo (see `docs/DEMO_SCRIPT.md`)
- [ ] LinkedIn post tagging the cohort

## Demo
https://drive.google.com/file/d/1D_vBS6W1gtQRHBtVs3ioylIFMTygWhrb/view?usp=sharing
