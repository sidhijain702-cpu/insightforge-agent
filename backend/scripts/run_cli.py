"""Run one brief from the terminal (best way to debug the pipeline):

    python -m scripts.run_cli "Summarize the competitive landscape for B2B payroll software in Southeast Asia"
    python -m scripts.run_cli --no-docs --no-prior "..."        # toggle sources
    python -m scripts.run_cli --pdf a.pdf --pdf b.pdf "..."      # attach PDFs

Prints live agent logs, then writes the report to ./out/<run_id>.md
"""
import argparse
import threading
import time
from pathlib import Path

from app import db, pipeline
from app.schemas import SourcePrefs


def tail_events(run_id: str, stop: threading.Event) -> None:
    last = 0
    while not stop.is_set() or True:
        rows = db.q("SELECT id, agent, message, level FROM events WHERE run_id=? AND id>? ORDER BY id", (run_id, last))
        for r in rows:
            last = r["id"]
            if r["message"].startswith("state ->") or r["level"] != "info" or True:
                print(f"[{r['level'][:4]}] {r['agent']}: {r['message']}")
        if stop.is_set() and not rows:
            return
        time.sleep(0.7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("brief")
    ap.add_argument("--no-web", action="store_true")
    ap.add_argument("--no-docs", action="store_true")
    ap.add_argument("--no-prior", action="store_true")
    ap.add_argument("--pdf", action="append", default=[])
    a = ap.parse_args()

    db.init_db()
    prefs = SourcePrefs(web=not a.no_web, docs=not a.no_docs and bool(a.pdf), prior=not a.no_prior)
    run_id = pipeline.create_run(a.brief, prefs)
    if a.pdf:
        pipeline.save_uploads(run_id, [(p, Path(p).read_bytes()) for p in a.pdf])
    stop = threading.Event()
    t = threading.Thread(target=tail_events, args=(run_id, stop), daemon=True)
    t.start()
    state = pipeline.execute_run(run_id, a.brief, prefs)
    stop.set()
    t.join(timeout=5)
    row = db.q1("SELECT status, error, clarifying_question FROM runs WHERE id=?", (run_id,))
    print("\nSTATUS:", row["status"], row["error"] or "", row["clarifying_question"] or "")
    if state and state.report_markdown:
        Path("out").mkdir(exist_ok=True)
        out = Path("out") / f"{run_id}.md"
        out.write_text(state.report_markdown, encoding="utf-8")
        print(f"report -> {out}   (run id {run_id}; try: python -m scripts.context_report {run_id})")
        fc = state.factcheck
        print(f"fact-check: {fc.passed}/{fc.total_checked} verified ({fc.pass_rate:.0%})")


if __name__ == "__main__":
    main()
