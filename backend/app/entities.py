"""Entity graph helpers: conflict detection + persisting entity embeddings.

Conflict rule: for a *single-valued* predicate (raised, valued_at, founded_in ...) two
relationships about the same subject with materially different objects are a
conflict, e.g. "Acme raised $20M" (source A) vs "Acme raised $35M" (source B).
We mark both `conflict=True`; Synthesis then downgrades confidence to LOW.
"""
import re
from collections import defaultdict

from app.schemas import Entity, Relationship

SINGLE_VALUED = {"raised", "valued_at", "founded_in", "headquartered_in", "priced_at", "revenue", "employs"}
_UNITS = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()


def parse_amount(text: str) -> float | None:
    """'$20 million' -> 2e7, 'US$1.2bn' -> 1.2e9, '35M' -> 3.5e7, '2019' -> 2019.0, 'abc' -> None."""
    m = re.search(r"(\d[\d,]*\.?\d*)\s*(thousand|million|billion|mn|bn|k|m|b)?\b", text.lower())
    if not m:
        return None
    try:
        val = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    return val * _UNITS.get(m.group(2) or "", 1)


def _canon(value: str):
    amt = parse_amount(value)
    return amt if amt is not None else norm(value)


def _differs(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) / max(abs(a), abs(b), 1e-9) > 0.05  # >5% apart = real disagreement
    return a != b


def flag_conflicts(rels: list[Relationship]) -> list[Relationship]:
    """Return the same relationships with `conflict` set where sources disagree."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, r in enumerate(rels):
        if r.predicate in SINGLE_VALUED:
            groups[(norm(r.subject), r.predicate)].append(i)
    for idxs in groups.values():
        for i in idxs:
            for j in idxs:
                if i < j and rels[i].source_url != rels[j].source_url:
                    if _differs(_canon(rels[i].object), _canon(rels[j].object)):
                        rels[i].conflict = rels[j].conflict = True
    return rels


def conflict_descriptions(rels: list[Relationship]) -> list[str]:
    """Human-readable lines for the Gaps section, e.g. 'Acme raised: $20M (a.com) vs $35M (b.com)'."""
    groups: dict[tuple[str, str], list[Relationship]] = defaultdict(list)
    for r in rels:
        if r.conflict:
            groups[(r.subject, r.predicate)].append(r)
    out = []
    for (subj, pred), rs in groups.items():
        vals = "; ".join(f"{r.object} ({r.source_url.split('/')[2] if r.source_url.startswith('http') else r.source_url})" for r in rs)
        out.append(f"{subj} {pred.replace('_', ' ')}: sources disagree -> {vals}")
    return out


def index_entities(run_id: str, entities: list[Entity]) -> int:
    """Persist entity embeddings in Qdrant so future runs can recall related entities."""
    from app import vector
    from app.embeddings import embed

    seen, names, payloads, ids = set(), [], [], []
    for e in entities:
        key = (norm(e.name), e.type)
        if not key[0] or key in seen:
            continue
        seen.add(key)
        names.append(e.name)
        payloads.append({"name": e.name, "type": e.type, "last_run": run_id})
        ids.append(vector.stable_id(e.type, key[0]))
    if not names:
        return 0
    vector.upsert("entities", embed(names), payloads, ids=ids)
    return len(names)
