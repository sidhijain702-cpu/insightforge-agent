"""Re-run ONLY the Synthesis step on a finished run with a smaller token budget.

    python -m scripts.resynth <run_id> <budget>        e.g.  python -m scripts.resynth 3514005c7f44 400

Why: your real runs produce small evidence bundles, so budget compression never triggers. This replays the
saved claims (nothing is re-fetched, no new research) under a tight budget so you can show the compression
log on REAL data. State clearly in the README that the budget was lowered for this demonstration.
Does not modify the original run or its report.
"""
import sys
from collections import Counter

from app import db
from app.agents.synthesis import render_bundle, synthesize
from app.schemas import TaskGraph
from app.scratch_store import ScratchStore


def main(run_id: str, budget: int) -> None:
    row = db.q1("SELECT brief, task_graph FROM runs WHERE id=?", (run_id,))
    if not row or not row["task_graph"]:
        sys.exit(f"run {run_id} not found or has no task graph")
    graph = TaskGraph.model_validate_json(row["task_graph"])
    store = ScratchStore(run_id)
    # distinct id so metrics/events of the ORIGINAL run are left untouched
    bundle = synthesize(f"{run_id}-resynth-{budget}", row["brief"], graph, store=store, budget=budget)

    log = bundle.compression_log
    print(f"\nRUN {run_id}  budget={budget} tokens")
    if not log:
        print("No compression needed: the bundle already fits. Try a smaller budget (e.g. 300).")
        return
    before, after = log[0].tokens_before, bundle.token_count
    print(f"evidence bundle: {before} -> {after} tokens in {len(log)} step(s)\n")
    for i, e in enumerate(log, 1):
        print(f"{i:>2}. {e.action:<10} {e.items_affected} item(s) in '{e.section}': {e.tokens_before} -> {e.tokens_after} tokens")
    print("\nreason (first step):", log[0].reason)
    kinds = Counter(e.action for e in log)
    print("\n--- README paragraph (edit wording, keep the numbers) ---\n")
    print(
        f"To demonstrate budget enforcement on real data, the saved claims of run `{run_id}` were re-synthesized with the "
        f"evidence budget lowered to {budget} tokens (default 6,000). The un-compressed bundle was {before} tokens. "
        f"The Synthesis Agent merged the lowest-ranked items (ranked by recency, then credibility, then relevance) in "
        f"{kinds.get('compressed', 0)} compression step(s)"
        + (f" and dropped items in {kinds['dropped']} step(s)" if kinds.get("dropped") else "")
        + f", ending at {after} tokens, with every step logged (section, items affected, source ids, reason, "
        f"tokens before/after).\n"
    )
    print("--- bundle the Writer would have received ---\n")
    print(render_bundle(bundle)[:2500])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m scripts.resynth <run_id> <budget>")
    main(sys.argv[1], int(sys.argv[2]))
