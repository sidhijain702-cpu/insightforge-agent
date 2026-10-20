"""Fetch a URL and extract the FULL article body (not a snippet).

Paywall handling: HTTP 401/402/403 or "subscribe to continue"-style text on a very
short page => skipped gracefully with a reason (the caller logs it, and the report's
Gaps section mentions it).
"""
from dataclasses import dataclass

import httpx

from app.config import settings

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; InsightForgeBot/1.0; +research)",
    "Accept": "text/html,application/xhtml+xml",
}
PAYWALL_PATTERNS = (
    "subscribe to continue", "subscribe to read", "sign in to read", "log in to continue",
    "create a free account to continue", "this content is for subscribers", "become a member to read",
    "already a subscriber", "to continue reading", "unlock this article",
)
MIN_BODY_CHARS = 400


@dataclass
class Fetched:
    ok: bool
    url: str
    text: str = ""
    title: str = ""
    author: str | None = None
    date: str | None = None
    skipped_reason: str = ""


def fetch_article(url: str, timeout: float = 20.0) -> Fetched:
    try:
        r = httpx.get(url, headers=HEADERS, timeout=timeout, follow_redirects=True)
    except Exception as e:  # noqa: BLE001
        return Fetched(False, url, skipped_reason=f"fetch error: {type(e).__name__}")
    if r.status_code in (401, 402, 403):
        return Fetched(False, url, skipped_reason=f"paywall/blocked (HTTP {r.status_code})")
    if r.status_code >= 400:
        return Fetched(False, url, skipped_reason=f"HTTP {r.status_code}")
    ctype = r.headers.get("content-type", "")
    if "html" not in ctype and "text" not in ctype:
        return Fetched(False, url, skipped_reason=f"unsupported content-type {ctype.split(';')[0]}")

    html = r.text
    import trafilatura

    text = trafilatura.extract(html, include_comments=False, include_tables=False, favor_precision=True) or ""
    meta = trafilatura.extract_metadata(html)
    low = html.lower()
    if len(text) < 1500 and any(p in low for p in PAYWALL_PATTERNS):
        return Fetched(False, url, skipped_reason="paywall/login wall detected")
    if len(text) < MIN_BODY_CHARS:
        return Fetched(False, url, skipped_reason="body too short / not an article")
    return Fetched(
        True, url, text=text[: settings.max_body_chars],
        title=(getattr(meta, "title", None) or ""),
        author=getattr(meta, "author", None),
        date=getattr(meta, "date", None),
    )
