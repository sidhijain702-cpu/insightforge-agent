from app.agents.diff import diff_reports

OLD = [{"claim": "Acme raised $20 million in Series B funding led by Sequoia in March 2026."},
       {"claim": "Beta Corp operates payroll services across Indonesia and Vietnam."},
       {"claim": "Gamma Ltd was acquired by Zenith Holdings in 2025 for an undisclosed sum."}]


def judge_by_marker(cands):
    """Stand-in for the LLM judge: wording we flag as 'reworded' is a rephrase, anything else is news."""
    return {c["id"]: ("rephrase" if "reworded" in c["new"] else "removed" if c["kind"] == "removed" else "new") for c in cands}


def test_identical_report_has_no_changes():
    assert diff_reports(OLD, list(OLD), judge=judge_by_marker).is_empty()


def test_new_finding_surfaces_and_rephrasing_does_not():
    new = [OLD[0], OLD[1], OLD[2],
           {"claim": "Delta Inc launched an earned-wage-access product for Philippine employers in September 2026."},
           {"claim": "Beta Corp operates payroll services across Indonesia and Vietnam (reworded)."}]
    out = diff_reports(OLD, new, judge=judge_by_marker)
    assert any("Delta Inc" in x for x in out.added_findings)
    assert not any("reworded" in x for x in out.added_findings + out.changed_findings)


def test_figure_change_is_reported_as_changed():
    new = [{"claim": "Acme raised $35 million in Series B funding led by Sequoia in March 2026."}, OLD[1], OLD[2]]
    out = diff_reports(OLD, new, judge=lambda cands: {c["id"]: ("changed" if c["kind"] in ("changed", "maybe", "added") and "Acme" in (c["new"] or "") else "same") for c in cands})
    assert any("$20" in x and "$35" in x for x in out.changed_findings)


def test_removed_finding_is_reported():
    out = diff_reports(OLD, [OLD[0], OLD[1]], judge=judge_by_marker)
    assert any("Gamma" in x for x in out.removed_findings)


def test_judge_failure_falls_back_instead_of_crashing():
    def boom(c):
        raise RuntimeError("llm down")
    out = diff_reports(OLD, OLD + [{"claim": "Totally different fact about Omega Group opening a Manila hub."}], judge=boom)
    assert any("Omega" in x for x in out.added_findings)
