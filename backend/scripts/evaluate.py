"""Score the reports produced by run_briefs and write ../EVALUATION.md.

    python -m scripts.evaluate

Metrics (all computed from your real runs, nothing is hard-coded):
  fact_check_pass_rate   from the Fact-Checker's own sample (>= 0.80 is the target)
  citation_accuracy      share of cited sentences whose text is >= 0.70 cosine-similar to the exact
                         source paragraph stored for that citation (checks the WHOLE report, not a sample)
  structure checks       exec summary <= 150 words, confidence label under every section, gaps section present
  section_coverage       share of the brief's expected sections matched by a report heading (cosine >= 0.45)
  judge scores (1-5)     LLM-as-judge for coverage / clarity / actionability. A SECOND opinion only:
                         the manual-review table in EVALUATION.md is the one the rubric asks for.
"""
import json
import re
import statistics
from pathlib import Path

from pydantic import BaseModel

from app import db
from app.citations import FINAL_CITE, MARKER, split_units, strip_citations
from app.config import settings
from app.embeddings import embed
from app.events import get_metrics
from app.llm import call_json

ROOT = Path(__file__).resolve().parent.parent
CIT_THRESHOLD, COVERAGE_THRESHOLD = 0.70, 0.45


class Judge(BaseModel):
    coverage: int = 3
    clarity: int = 3
    actionability: int = 3
    comment: str = ""


def citation_accuracy(markdown: str, citations: dict) -> tuple[float, int]:
    units = [u for u in split_units(markdown, FINAL_CITE) if u.cites]
    if not units:
        return 0.0, 0
    ok = 0
    for u in units:
        best = 0.0
        for c in u.cites:
            para = (citations.get(c) or {}).get("paragraph", "")
            if para:
                v = embed([strip_citations(u.text), para])
                best = max(best, float(v[0] @ v[1]))
        ok += best >= CIT_THRESHOLD
    return ok / len(units), len(units)


def structure(markdown: str) -> dict:
    heads = re.findall(r"^## (.+)$", markdown, flags=re.M)
    exec_txt = re.search(r"## Executive Summary\n(.*?)\n## ", markdown, flags=re.S)
    words = len(MARKER.sub("", strip_citations(exec_txt.group(1))).split()) if exec_txt else 999
    themed = heads[1:-1] if len(heads) > 2 else []
    labelled = sum(bool(re.search(rf"## {re.escape(h)}\n\n\*\*Confidence: (HIGH|MEDIUM|LOW)\*\*", markdown)) for h in themed)
    return {"exec_summary_words": words, "exec_ok": words <= 150, "sections": len(themed),
            "all_sections_labelled": bool(themed) and labelled == len(themed),
            "has_gaps": bool(heads) and heads[-1].startswith("Gaps"), "headings": themed}


def coverage(expected: list[str], headings: list[str]) -> float:
    if not expected or not headings:
        return 0.0
    e, h = embed(expected), embed(headings)
    return float(((e @ h.T).max(axis=1) >= COVERAGE_THRESHOLD).mean())


def judge(brief: str, expected: list[str], report: str) -> Judge:
    prompt = (
        f"BRIEF: {brief}\nEXPECTED TOPICS: {expected}\n\nREPORT:\n{report[:9000]}\n\n"
        "Score the report 1-5 (5 best): coverage (does it address the brief and expected topics with substance), "
        "clarity (plain, well structured), actionability (could a strategy team act on it). Be strict. "
        'Return JSON {"coverage":n,"clarity":n,"actionability":n,"comment":"one sentence"}'
    )
    return call_json(prompt, Judge, model=settings.writer_model)


def main() -> None:
    runs = json.loads((ROOT / "evaluation/runs.json").read_text())
    briefs = {json.loads(p.read_text())["id"]: json.loads(p.read_text()) for p in (ROOT / "briefs").glob("*.json")}
    rows = []
    for r in runs:
        if not r["report_id"]:
            rows.append({**r, "failed": True})
            continue
        rep = db.q1("SELECT markdown, citations, factcheck FROM reports WHERE id=?", (r["report_id"],))
        md, cits, fc = rep["markdown"], db.jload(rep["citations"], {}), db.jload(rep["factcheck"], {})
        b = briefs[r["brief_id"]]
        acc, n_units = citation_accuracy(md, cits)
        st = structure(md)
        try:
            j = judge(b["brief"], b["expected_sections"], md)
        except Exception as e:  # noqa: BLE001
            j = Judge(comment=f"judge unavailable: {e}")
        m = get_metrics(r["run_id"])
        rows.append({**r, "fact_check": fc.get("pass_rate"), "fact_checked": fc.get("total_checked"), "citation_accuracy": round(acc, 3),
                     "cited_sentences": n_units, "structure": st, "coverage": round(coverage(b["expected_sections"], st["headings"]), 2),
                     "judge": j.model_dump(), "writer_retries": m.get("writer_retries"), "removed": m.get("writer_statements_removed"),
                     "raw_tokens": m.get("raw_tokens_fetched"), "bundle_tokens": m.get("bundle_tokens_final"),
                     "compression_steps": m.get("compression_steps")})
    (ROOT / "evaluation/results.json").write_text(json.dumps(rows, indent=2))

    ok = [x for x in rows if not x.get("failed")]
    avg = lambda k: round(statistics.mean(x[k] for x in ok if x.get(k) is not None), 3) if ok else None
    lines = ["# Evaluation Report", "",
             f"Generated by `python -m scripts.evaluate` from {len(ok)}/{len(rows)} completed runs. Models: planner `{settings.planner_model}`, "
             f"writer `{settings.writer_model}`, fast `{settings.fast_model}`; embeddings `{settings.embedding_model}`; "
             f"fact-check threshold {settings.factcheck_threshold}.", "",
             "## Automated metrics", "",
             "| Brief | Status | Fact-check pass | Citation accuracy | Exec ≤150w | Sections labelled | Gaps section | Coverage | Judge (cov/clar/act) | Writer retries | Seconds |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        if x.get("failed"):
            lines.append(f"| {x['brief_id']} | FAILED: {x.get('error')} |||||||||{x['seconds']} |")
            continue
        s, j = x["structure"], x["judge"]
        lines.append(f"| {x['brief_id']} | {x['status']} | {x['fact_check']:.0%} ({x['fact_checked']} claims) | {x['citation_accuracy']:.0%} ({x['cited_sentences']}) | "
                     f"{'yes' if s['exec_ok'] else 'NO'} ({s['exec_summary_words']}) | {'yes' if s['all_sections_labelled'] else 'NO'} | {'yes' if s['has_gaps'] else 'NO'} | "
                     f"{x['coverage']:.0%} | {j['coverage']}/{j['clarity']}/{j['actionability']} | {x['writer_retries']} | {x['seconds']} |")
    if ok:
        lines += ["", f"**Mean fact-check pass rate:** {avg('fact_check'):.0%} (target ≥ 80%)  ",
                  f"**Mean citation accuracy:** {avg('citation_accuracy'):.0%}  ", f"**Mean section coverage:** {avg('coverage'):.0%}"]
    lines += ["", "## Manual review (fill in after reading each report)", "",
              "| Brief | Coherent (1-5) | Faithful to sources (1-5) | Synthesis quality (1-5) | Notes: contradictions handled? hallucinations spotted? |",
              "|---|---|---|---|---|"] + [f"| {x['brief_id']} |  |  |  |  |" for x in rows]
    lines += ["", "## Context engineering evidence", "", "| Brief | Raw tokens fetched | Evidence bundle tokens | Compression steps |", "|---|---|---|---|"]
    lines += [f"| {x['brief_id']} | {x.get('raw_tokens')} | {x.get('bundle_tokens')} | {x.get('compression_steps')} |" for x in ok]
    lines += ["", "## How to read these numbers", "",
              "- Fact-check pass rate is measured on the fact-checker's own random 20% sample, so it is a sample estimate; citation accuracy covers every cited sentence.",
              "- A low pass rate with high citation accuracy usually means the threshold (0.85) is strict for paraphrased sentences: inspect `factcheck.details[*].score` before changing the writer.",
              "- LLM-judge scores come from the same model family as the writer and are only a second opinion; trust the manual table above.", ""]
    (ROOT.parent / "EVALUATION.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote EVALUATION.md")


if __name__ == "__main__":
    main()
