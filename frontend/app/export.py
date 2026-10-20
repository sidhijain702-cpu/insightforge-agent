"""Report export (PDF / Markdown) and JWT-signed, expiring share links."""
import re
from datetime import datetime, timedelta, timezone

import jwt

from app.config import settings

CSS = """
@page { size: A4; margin: 22mm 18mm; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 11pt; line-height: 1.5; color: #14213d; }
h1 { font-size: 22pt; margin: 0 0 6pt; } h2 { font-size: 15pt; margin: 20pt 0 4pt; border-bottom: 1px solid #c9d1da; padding-bottom: 3pt; }
.cite { color: #4b5d75; font-size: 8.5pt; } mark { background: #fde8b0; padding: 0 2pt; }
.badge { font-family: Helvetica, Arial, sans-serif; font-size: 8.5pt; padding: 1pt 6pt; border-radius: 8pt; color: #fff; }
.badge.high { background: #1f7a4d; } .badge.medium { background: #b7791f; } .badge.low { background: #b42318; }
"""


def export_markdown(md: str) -> str:
    """Portable .md: citation links keep their visible text; internal cite: targets are dropped."""
    return re.sub(r"\[(Source: [^\]]*)\]\(cite:E\d+\)", r"[\1]", md)


def to_html(md: str, title: str = "Market Intelligence Report") -> str:
    import markdown as mdlib

    t = re.sub(r"\[(Source: [^\]]*)\]\(cite:E\d+\)", r'<span class="cite">[\1]</span>', md)
    t = t.replace("[UNVERIFIED]", '<mark>[UNVERIFIED]</mark>')
    t = re.sub(r"\*\*Confidence: (HIGH|MEDIUM|LOW)\*\*",
               lambda m: f'<span class="badge {m.group(1).lower()}">Confidence: {m.group(1)}</span>', t)
    body = mdlib.markdown(t, extensions=["extra", "sane_lists"])
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>"


_PUNCT = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2026": "...", "\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u200b": "",
    "\u2022": "-", "\u2192": "->", "\u2248": "~", "\u20ac": "EUR ", "\u20b9": "INR ",
}


def _to_latin1(s: str) -> str:
    """The built-in PDF fonts only support Latin-1: map common typographic characters to plain ASCII first,
    otherwise they would all turn into '?'."""
    for k, v in _PUNCT.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "replace").decode("latin-1")


def _fpdf_fallback(md: str) -> bytes:
    """Pure-python PDF (no system libraries). Plain styling; used if WeasyPrint can't load."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    for raw in export_markdown(md).splitlines():
        line = _to_latin1(raw).rstrip()
        if not line:
            pdf.ln(3)
            continue
        size, style = 11, ""
        if line.startswith("# "):
            size, style, line = 18, "B", line[2:]
        elif line.startswith("## "):
            size, style, line = 14, "B", line[3:]
        line = line.replace("**", "").replace("*", "")
        pdf.set_font("Helvetica", style, size)
        pdf.multi_cell(0, 6 if size == 11 else 8, line, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def to_pdf(md: str, title: str = "Market Intelligence Report") -> bytes:
    try:
        from weasyprint import HTML

        return HTML(string=to_html(md, title)).write_pdf()
    except (ImportError, OSError):   # OSError: pango/cairo shared libraries missing
        return _fpdf_fallback(md)


# ------------------------------------------------------------------ share links
def make_share_token(report_id: str) -> tuple[str, datetime]:
    exp = datetime.now(timezone.utc) + timedelta(days=settings.share_ttl_days)
    return jwt.encode({"rid": report_id, "exp": exp}, settings.jwt_secret, algorithm="HS256"), exp


def verify_share_token(token: str) -> str:
    """Return the report id, or raise jwt.PyJWTError (expired / tampered)."""
    return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])["rid"]
