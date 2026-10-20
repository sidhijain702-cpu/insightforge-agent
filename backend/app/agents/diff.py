"""Diff Agent: what MATERIALLY changed between two reports on the same topic?

Two stages so that rephrasing is not flagged as news:
  1. embeddings pre-filter  - match every new claim to its closest old claim
        sim >= 0.92 and same numbers  -> unchanged (ignore)
        sim >= 0.92 but numbers differ -> candidate "changed"
        0.75 <= sim < 0.92            -> "maybe": same fact reworded, or a real update? ask the judge
        sim < 0.75                    -> candidate "added"
     old claims with no counterpart in the new report -> candidate "removed"
  2. LLM judge - sees each (old, new) pair and returns new | changed | rephrase | removed | same.
     Only new / changed / removed verdicts reach the user.
"""
import re

from pydantic import BaseModel, Field

from app.config import settings
from app.embeddings import embed
from app.schemas import ChangeSummary

SAME, MATCH = 0.92, 0.75
MAX_CANDIDATES = 24


def _nums(s: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in re.findall(r"\d[\d,]*\.?\d*", s)}


class _Verdicts(BaseModel):
    verdicts: list[dict] = Field(default_factory=list)


def default_judge(cands: list[dict]) -> dict[int, str]:
    """LLM judge; falls back to deterministic verdicts if the model is unavailable."""
    from app.llm import call_json

    lines = []
    for c in cands:
        lines.append(f'{c["id"]}. KIND={c["kind"]}\n   OLD: {c.get("old") or "(none)"}\n   NEW: {c.get("new") or "(none)"}')
    prompt = (
        "You compare findings from an earlier and a later market-research report on the same topic. For each "
        "numbered item decide the verdict:\n"
        "- new: NEW states a materially new fact not present in OLD\n"
        "- changed: same fact but a figure, date, status or outcome differs\n"
        "- rephrase: same information in different words (NOT news)\n"
        "- removed: OLD fact has no counterpart now (KIND=removed)\n"
        "- same: identical\n"
        'Return JSON: {"verdicts":[{"id":0,"verdict":"new"}]}\n\n' + "\n".join(lines)
    )
    out = call_json(prompt, _Verdicts, model=settings.fast_model)
    return {int(v["id"]): str(v.get("verdict", "same")) for v in out.verdicts if "id" in v}


def fallback_judge(cands: list[dict]) -> dict[int, str]:
    m = {"added": "new", "changed": "changed", "maybe": "rephrase", "removed": "removed"}
    return {c["id"]: m[c["kind"]] for c in cands}


def diff_reports(old: list[dict], new: list[dict], judge=None) -> ChangeSummary:
    """`old`/`new` are lists of {'claim': str, ...}. Returns only materially new information."""
    if not old or not new:
        return ChangeSummary(added_findings=[n["claim"] for n in new][:10]) if new and not old else ChangeSummary()
    ov, nv = embed([o["claim"] for o in old]), embed([n["claim"] for n in new])
    sim = nv @ ov.T

    cands: list[dict] = []
    for i, n in enumerate(new):
        j = int(sim[i].argmax())
        s = float(sim[i, j])
        if s >= SAME:
            if _nums(n["claim"]) != _nums(old[j]["claim"]):
                cands.append({"kind": "changed", "old": old[j]["claim"], "new": n["claim"], "sim": s})
        elif s >= MATCH:
            cands.append({"kind": "maybe", "old": old[j]["claim"], "new": n["claim"], "sim": s})
        else:
            cands.append({"kind": "added", "old": "", "new": n["claim"], "sim": s})
    for j, o in enumerate(old):
        if float(sim[:, j].max()) < MATCH:
            cands.append({"kind": "removed", "old": o["claim"], "new": "", "sim": 0.0})

    cands = sorted(cands, key=lambda c: c["sim"])[:MAX_CANDIDATES]
    for k, c in enumerate(cands):
        c["id"] = k
    if not cands:
        return ChangeSummary()
    try:
        verdicts = (judge or default_judge)(cands)
    except Exception:  # noqa: BLE001
        verdicts = fallback_judge(cands)

    out = ChangeSummary()
    for c in cands:
        v = verdicts.get(c["id"], "same")
        if v == "new" and c["new"]:
            out.added_findings.append(c["new"])
        elif v == "changed":
            out.changed_findings.append(f'{c["old"]}  ->  {c["new"]}' if c["old"] and c["new"] else (c["new"] or c["old"]))
        elif v == "removed" and c["old"]:
            out.removed_findings.append(c["old"])
    return out
