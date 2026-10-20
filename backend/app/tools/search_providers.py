"""Web search adapters. Pick one with SEARCH_PROVIDER=tavily|brave|serpapi.

Every adapter returns the same shape so agents never care which one is used:
    [{"url", "title", "snippet", "published_date"}]
"""
import re

import httpx

from app.config import settings


def horizon_days(horizon: str) -> int | None:
    """'last 18 months' -> 540, 'past 2 years' -> 730, 'last 30 days' -> 30, unknown -> None."""
    m = re.search(r"(\d+)\s*(day|week|month|year)", horizon.lower())
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    return n * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]


def _tavily(query: str, n: int, days: int | None) -> list[dict]:
    from tavily import TavilyClient

    kwargs = {"query": query, "max_results": n, "search_depth": "basic"}
    if days is not None and days <= 365:
        kwargs["time_range"] = "month" if days <= 31 else "year"
    res = TavilyClient(api_key=settings.tavily_api_key).search(**kwargs)
    return [
        {"url": r["url"], "title": r.get("title", ""), "snippet": r.get("content", ""),
         "published_date": r.get("published_date")}
        for r in res.get("results", [])
    ]


def _brave(query: str, n: int, days: int | None) -> list[dict]:
    params = {"q": query, "count": min(n, 20)}
    if days is not None and days <= 365:
        params["freshness"] = "pm" if days <= 31 else "py"
    r = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"X-Subscription-Token": settings.brave_api_key, "Accept": "application/json"},
        params=params, timeout=20,
    )
    r.raise_for_status()
    return [
        {"url": x["url"], "title": x.get("title", ""), "snippet": x.get("description", ""),
         "published_date": x.get("page_age")}
        for x in r.json().get("web", {}).get("results", [])
    ]


def _serpapi(query: str, n: int, days: int | None) -> list[dict]:
    r = httpx.get(
        "https://serpapi.com/search.json",
        params={"q": query, "api_key": settings.serpapi_api_key, "engine": "google", "num": n},
        timeout=30,
    )
    r.raise_for_status()
    return [
        {"url": x["link"], "title": x.get("title", ""), "snippet": x.get("snippet", ""),
         "published_date": x.get("date")}
        for x in r.json().get("organic_results", [])
    ]


PROVIDERS = {"tavily": _tavily, "brave": _brave, "serpapi": _serpapi}


def web_search(query: str, n: int, time_horizon: str = "") -> list[dict]:
    provider = PROVIDERS.get(settings.search_provider)
    if provider is None:
        raise ValueError(f"unknown SEARCH_PROVIDER '{settings.search_provider}' (use tavily|brave|serpapi)")
    return provider(query, n, horizon_days(time_horizon))
