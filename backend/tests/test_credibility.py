from datetime import datetime

from app.tools.credibility import domain_score, norm_date, recency_score, score_source

NOW = datetime(2026, 10, 16)


def test_domain_ordering():
    assert domain_score("mit.edu") == 1.0 == domain_score("data.gov")
    assert domain_score("reuters.com") > domain_score("someone.medium.com") > domain_score("acme-unknown.io")


def test_recency_decays():
    fresh, old = recency_score("2026-10-01", NOW), recency_score("2023-01-01", NOW)
    assert fresh > 0.95 and old < 0.2 and recency_score(None, NOW) == 0.30


def test_byline_adds_score():
    assert score_source("reuters.com", "2026-10-01", "A. Writer", NOW) > score_source("reuters.com", "2026-10-01", None, NOW)


def test_norm_date():
    assert norm_date("March 3, 2026") == "2026-03-03" and norm_date("garbage") is None
