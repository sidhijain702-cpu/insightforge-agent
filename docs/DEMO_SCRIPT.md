# Demo script (5-7 minutes)

**Before recording:** `DEMO_INTERVAL_SECONDS=240`, backend + frontend running, one baseline watchlist run already completed so the *second* run can produce an alert. Warm the embedding model with one throwaway run. Have one 10-20 page PDF ready.

| Time | Show | Say |
|---|---|---|
| 0:00 | Home page | Problem in one sentence: analysts burn 15-20 h/week gathering. This turns a brief into a cited report. |
| 0:30 | Paste the payroll brief, drop a PDF, toggle sources, point at the live char counter and runtime estimate | Inputs and controls. |
| 1:00 | Click Run. Progress view | Planner splits into sub-tasks; rows go Queued→Running→Done; agent log streams; progress bar and elapsed time. |
| 2:30 | Explain while it runs (architecture slide or README diagram) | Subagents are isolated: each sees one sub-task. Raw pages go to the scratch store; only ids come back. |
| 3:30 | Open the report | TOC, confidence badges, executive summary. |
| 4:00 | **Click a citation** | Drawer: source, date, credibility, the exact paragraph. This is the key moment. |
| 4:30 | Point at an `[UNVERIFIED]` flag + fact-check summary; show evidence bundle tokens / compression log | Fact-checker re-fetched sources; context budget enforced and logged. |
| 5:15 | Download PDF, copy share link | Export and 7-day JWT link. |
| 5:45 | Watchlist: show the alert that fired after the scheduled run; Slack message if configured | Diff Agent surfaced only material change. |
| 6:30 | Close | What you'd improve next. |
