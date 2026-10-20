from app.agents.writer import check_unit, expand_markers, write_grounded
from app.citations import MARKER, split_units
from app.schemas import EvidenceItem


def item(i, claim, domain="a.com", date="2026-01-01", cred=0.8):
    return EvidenceItem(id=f"E{i}", claim=claim, source_id=f"s{i}", url=f"https://{domain}/p", domain=domain, date=date, credibility=cred)


EV = {"E1": item(1, "Acme raised $20 million in Series B funding led by Sequoia in March 2026."),
      "E2": item(2, "Acme prices its Pro plan at $12 per employee per month.", "b.com")}
SECTION = " ".join(i.claim for i in EV.values())


def units(text):
    return split_units(text, MARKER)


def test_grounded_statement_passes():
    u = units("Acme raised $20 million in a Series B round led by Sequoia. {E1}")[0]
    assert check_unit(u, EV, SECTION) == []


def test_invented_number_fails():
    u = units("Acme raised $50 million in a Series B round. {E1}")[0]
    assert any("numbers" in v for v in check_unit(u, EV, SECTION))


def test_phantom_id_fails():
    u = units("Acme raised money. {E9}")[0]
    assert any("unknown evidence" in v for v in check_unit(u, EV, SECTION))


def test_invented_names_fail():
    u = units("Acme was backed by Andreessen Horowitz and Accel. {E1}")[0]
    assert check_unit(u, EV, SECTION)


def test_retry_adds_explicit_constraint_then_succeeds():
    prompts = []

    def fake_llm(prompt, **kw):
        prompts.append(prompt)
        if len(prompts) == 1:
            return "Acme raised $50 million last year. {E1}"             # hallucinated figure
        return "Acme raised $20 million in Series B funding in March 2026. {E1}"

    text, retries, removed = write_grounded("WRITE", EV, "sys", "m", llm=fake_llm)
    assert retries == 1 and "$20 million" in text and removed == []
    assert "CORRECTION REQUIRED" in prompts[1] and "$50" in prompts[1] or "50" in prompts[1]


def test_unfixable_statement_is_removed_not_shipped():
    def bad_llm(prompt, **kw):
        return "Acme raised $99 million. {E1}\nAcme prices its Pro plan at $12 per employee per month. {E2}"

    text, retries, removed = write_grounded("WRITE", EV, "sys", "m", retries=1, llm=bad_llm)
    assert "$99" not in text and "{E2}" in text and removed


def test_exec_summary_word_limit_enforced():
    long = " ".join(["Acme raised $20 million in Series B funding led by Sequoia in March 2026. {E1}"] * 8)
    text, _, _ = write_grounded("W", EV, "s", "m", max_words=40, retries=0, llm=lambda p, **k: long)
    assert len(MARKER.sub("", text).split()) <= 40


def test_expand_markers_uses_required_citation_format():
    out = expand_markers("Acme raised $20 million. {E1}{E2}", EV)
    assert "[Source: a.com, 2026-01-01, credibility: 0.80](cite:E1)" in out and "(cite:E2)" in out


def test_uncited_inference_is_rejected_but_limitation_note_is_allowed():
    inference = units("Oracle's broader coverage potentially lowers total cost of ownership for buyers.")[0]
    assert any("uncited" in v for v in check_unit(inference, EV, SECTION))
    note = units("Evidence on this topic is limited to a single source.")[0]
    assert check_unit(note, EV, SECTION) == []


def test_identical_adjacent_citations_are_shown_once():
    same = {"E1": item(1, "Acme raised $20 million.", "a.com"), "E2": item(2, "Acme prices Pro at $12 monthly.", "a.com")}
    out = expand_markers("Acme raised $20 million and prices Pro at $12. {E1}{E2}", same)
    assert out.count("[Source: a.com") == 1


def test_citation_is_never_glued_to_the_previous_word():
    out = expand_markers("Acme raised $20 million{E1}.", EV)
    assert "million [Source: a.com" in out and "million[Source" not in out
