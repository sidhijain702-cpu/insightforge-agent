"""Split report text into *claim units* (a sentence/bullet + the citations that end it).

Used twice:
  * Writer      - parses draft text containing  {E12}  markers   (to check the draft)
  * FactChecker - parses final text containing    [Source: ...](cite:E12)  (to sample claims)

LEARNING NOTE: the Writer never types the long citation string itself. It writes a
short marker like {E12}; the orchestrator expands it to
`[Source: reuters.com, 2025-03-02, credibility: 0.82](cite:E12)`. That makes citation
formatting errors impossible and makes every citation checkable against the bundle.
"""
import re
from dataclasses import dataclass, field

MARKER = re.compile(r"\{(E\d+)\}")
FINAL_CITE = re.compile(r"\[Source: [^\]\n]*\]\(cite:(E\d+)\)")
_SKIP_LINE = re.compile(r"^\s*(#|\*\*Confidence)")


@dataclass
class Unit:
    text: str                      # claim text with citations removed
    cites: list[str] = field(default_factory=list)
    start: int = 0                 # offset in the source string where the unit's text begins
    end: int = 0                   # offset just after its last citation (or end of line if uncited)


def _clean(s: str) -> str:
    s = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", s)           # bullet / numbering
    return s.strip().lstrip(" .,;:-–—)").strip()


def split_units(text: str, cite_re: re.Pattern = MARKER) -> list[Unit]:
    units: list[Unit] = []
    base = 0
    for raw in text.splitlines(keepends=True):
        line = raw.rstrip("\r\n")
        if line.strip() and not _SKIP_LINE.match(line):
            ms = list(cite_re.finditer(line))
            pos, i = 0, 0
            while i < len(ms):
                m = ms[i]
                cites, end, j = [m.group(1)], m.end(), i + 1
                while j < len(ms) and re.fullmatch(r"[\s,;]*", line[end : ms[j].start()]):
                    cites.append(ms[j].group(1))
                    end = ms[j].end()
                    j += 1
                seg = _clean(line[pos : m.start()])
                if re.search(r"\w", seg):
                    units.append(Unit(seg, cites, base + pos, base + end))
                pos, i = end, j
            tail = _clean(line[pos:])
            if re.search(r"\w{3,}", tail):
                units.append(Unit(tail, [], base + pos, base + len(line)))
        base += len(raw)
    return units


def strip_citations(text: str) -> str:
    return re.sub(r"\s+", " ", MARKER.sub("", FINAL_CITE.sub("", text))).strip()
