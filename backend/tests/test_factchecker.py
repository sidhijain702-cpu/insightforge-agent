import tempfile

from app.agents import factchecker
from app.config import settings
from app.scratch_store import ScratchStore
from app.tools.fetch import Fetched


def md(n_good, n_bad):
    lines = ["## Section", "", "**Confidence: HIGH**", ""]
    for i in range(n_good):
        lines.append(f"- Good fact number {i} is present in the page. [Source: a.com, 2026-01-01, credibility: 0.80](cite:E1)")
    for i in range(n_bad):
        lines.append(f"- Bad fact number {i} appears nowhere online. [Source: a.com, 2026-01-01, credibility: 0.80](cite:E1)")
    return "\n".join(lines) + "\n"


CIT = {"E1": {"url": "https://a.com/p", "source_id": "src_x", "claim": "c"}}


def run(markdown):
    orig = factchecker.fetch_article
    factchecker.fetch_article = lambda url: Fetched(True, url, text="Some page text about things. " * 20)
    try:
        store = ScratchStore("fc_run", root=tempfile.mkdtemp())
        sim = lambda claim, page: 0.95 if "Good" in claim else 0.30
        return factchecker.fact_check("fc_run", markdown, CIT, store=store, similarity=sim)
    finally:
        factchecker.fetch_article = orig


def test_failed_claims_are_marked_unverified_and_summary_is_consistent():
    out, res = run(md(12, 8))
    n_expected = max(settings.factcheck_min_claims, -(-20 * 20 // 100))   # ceil(20% of 20) = 4 -> min 5
    assert res.total_checked == n_expected == 5
    assert res.passed + res.failed == res.total_checked
    assert out.count("[UNVERIFIED]") == res.failed == len(res.unverified_claims)
    flagged = [ln for ln in out.splitlines() if "[UNVERIFIED]" in ln]
    assert all("Bad fact" in ln for ln in flagged)           # only claims that failed the check are flagged
    assert len(flagged) == res.failed
    assert res.pass_rate == round(res.passed / res.total_checked, 3)


def test_all_good_report_has_no_flags():
    out, res = run(md(10, 0))
    assert res.failed == 0 and res.pass_rate == 1.0 and "[UNVERIFIED]" not in out


def test_small_report_checks_everything():
    out, res = run(md(2, 1))
    assert res.total_checked == 3
