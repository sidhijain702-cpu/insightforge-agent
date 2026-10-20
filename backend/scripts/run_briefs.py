"""Run all 5 sample briefs (or a subset) and store the reports:

    python -m scripts.run_briefs            # all five
    python -m scripts.run_briefs 01 03      # only briefs whose id starts with 01 / 03
"""
import json
import sys
import time
from pathlib import Path

from app import db, pipeline
from app.schemas import SourcePrefs

BRIEFS = Path(__file__).resolve().parent.parent / "briefs"


def load(prefixes: list[str]) -> list[dict]:
    items = [json.loads(p.read_text()) for p in sorted(BRIEFS.glob("*.json"))]
    return [b for b in items if not prefixes or any(b["id"].startswith(x) for x in prefixes)]


def run_one(b: dict) -> dict:
    prefs = SourcePrefs(web=True, docs=False, prior=False)   # fair comparison: web only, no memory bleed
    run_id = pipeline.create_run(b["brief"], prefs)
    t0 = time.time()
    state = pipeline.execute_run(run_id, b["brief"], prefs)
    row = db.q1("SELECT status, error FROM runs WHERE id=?", (run_id,))
    return {"brief_id": b["id"], "run_id": run_id, "status": row["status"], "error": row["error"],
            "report_id": state.report_id if state else None, "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    db.init_db()
    out = []
    for b in load(sys.argv[1:]):
        print(f"\n=== {b['id']} ===")
        r = run_one(b)
        print(r)
        out.append(r)
    Path("evaluation").mkdir(exist_ok=True)
    Path("evaluation/runs.json").write_text(json.dumps(out, indent=2))
    print("\nsaved evaluation/runs.json  ->  now run: python -m scripts.evaluate")
