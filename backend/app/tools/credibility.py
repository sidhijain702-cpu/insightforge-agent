"""Simple, explainable credibility rubric (0..1).

    credibility = 0.5 * domain_type + 0.3 * recency + 0.2 * byline

* domain_type : .edu/.gov = 1.0 > major press/analysts = 0.85 > blogs = 0.45 > unknown = 0.30
* recency     : exponential decay, half-life 1 year (older = lower); unknown date = 0.30
* byline      : 1.0 if an author is named, else 0.0

LEARNING NOTE: a rubric you can read in 20 lines beats an opaque model here. An
analyst can disagree with a weight, but cannot disagree with "I don't know why".
"""
from datetime import datetime, timezone

MAJOR_PRESS = {
    "reuters.com", "bloomberg.com", "ft.com", "wsj.com", "nytimes.com", "bbc.com", "bbc.co.uk",
    "economist.com", "forbes.com", "cnbc.com", "techcrunch.com", "theguardian.com", "apnews.com",
    "washingtonpost.com", "businessinsider.com", "straitstimes.com", "channelnewsasia.com",
    "nikkei.com", "asia.nikkei.com", "scmp.com", "techinasia.com", "e27.co", "crunchbase.com",
    "pitchbook.com", "gartner.com", "mckinsey.com", "forrester.com", "statista.com", "idc.com",
    "sec.gov", "hbr.org", "dealstreetasia.com", "kr-asia.com", "vulcanpost.com", "businesstimes.com.sg",
    "theedgesingapore.com", "thejakartapost.com", "bangkokpost.com", "techcollectivesea.com", "finextra.com", "deloitte.com", "pwc.com", "bain.com", "bcg.com",
}
BLOG_HINTS = {
    "medium.com", "substack.com", "blogspot.com", "wordpress.com", "linkedin.com",
    "quora.com", "reddit.com", "tumblr.com", "wixsite.com",
}


def domain_score(domain: str) -> float:
    d = domain.lower()
    parts = d.split(".")
    if d.endswith((".edu", ".gov")) or ".gov." in d or ".edu." in d or ".ac." in d:
        return 1.0
    if any(d == m or d.endswith("." + m) for m in MAJOR_PRESS):
        return 0.85
    if any(d == b or d.endswith("." + b) for b in BLOG_HINTS) or "blog" in parts[0]:
        return 0.45
    return 0.30


_FORMATS = ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d", "%d %B %Y", "%B %d, %Y", "%b %d, %Y", "%Y-%m")


def parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s.strip()
    try:  # ISO 8601, with or without timezone / fractional seconds
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        pass
    for fmt in _FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:  # "2025-03-14 and some trailing junk"
        return datetime.strptime(s[:10], "%Y-%m-%d")
    except ValueError:
        return None


def recency_score(date: str | None, now: datetime | None = None) -> float:
    d = parse_date(date)
    if d is None:
        return 0.30
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    age_days = max(0.0, (now - d).total_seconds() / 86400)
    return max(0.05, 0.5 ** (age_days / 365.0))


def score_source(domain: str, published_date: str | None, author: str | None, now: datetime | None = None) -> float:
    s = 0.5 * domain_score(domain) + 0.3 * recency_score(published_date, now) + 0.2 * (1.0 if author else 0.0)
    return round(min(1.0, max(0.0, s)), 2)


def norm_date(s: str | None) -> str | None:
    """Canonical YYYY-MM-DD for display/citations; None if unknown."""
    d = parse_date(s)
    return d.strftime("%Y-%m-%d") if d else None
