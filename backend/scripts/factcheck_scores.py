"""Why did claims fail the fact-check?  python -m scripts.factcheck_scores

Reads evaluation/runs.json, prints every FAILED claim with its similarity score, and a score histogram.
Decision rule:
  * failures mostly between 0.75 and 0.85  -> the claim is probably supported but PARAPHRASED; the 0.85 threshold is strict
  * failures far below 0.6                 -> the sentence is really not supported by the page (or the page changed / was fetched wrongly)
Do not change FACTCHECK_THRESHOLD before reading this output.
"""
import json
from pathlib import Path

from app import db

runs = json.loads(Path("evaluation/runs.json").read_text())
scores, failed = [], []
for r in runs:
    rep = db.q1("SELECT factcheck FROM reports WHERE id=?", (r["report_id"],))
    for d in (db.jload(rep["factcheck"], {}) or {}).get("details", []) if rep else []:
        scores.append(d["score"])
        if not d["passed"]:
            failed.append((r["brief_id"], d["score"], d["checked_against"], d["claim"]))

print(f"{len(scores)} claims checked, {len(failed)} failed\n")
for b, sc, how, claim in sorted(failed, key=lambda f: -f[1]):
    print(f"{b:<26} score={sc:<6} ({how})  {claim[:120]}")
buckets = [("< 0.60", lambda x: x < 0.60), ("0.60-0.75", lambda x: 0.60 <= x < 0.75),
           ("0.75-0.85", lambda x: 0.75 <= x < 0.85), (">= 0.85", lambda x: x >= 0.85)]
print("\nscore distribution of ALL checked claims:")
for name, f in buckets:
    print(f"  {name:<10} {sum(1 for s in scores if f(s))}")
near = sum(1 for _, s, _, _ in failed if 0.75 <= s < 0.85)
print(f"\nfailures in the 0.75-0.85 'probably paraphrase' band: {near} of {len(failed)}")
