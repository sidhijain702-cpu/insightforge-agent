from app.citations import FINAL_CITE, MARKER, split_units


def test_markers_group_and_offsets():
    text = "## Heading\n\n**Confidence: HIGH**\n\nAcme raised $20M. {E1}{E2}\n- Beta launched a product. {E3}\nUncited transition."
    units = split_units(text, MARKER)
    assert [u.cites for u in units] == [["E1", "E2"], ["E3"], []]
    assert units[0].text == "Acme raised $20M."
    assert text[units[0].start : units[0].end].endswith("{E2}")


def test_final_citation_format():
    t = "Acme raised $20M [Source: reuters.com, 2026-01-02, credibility: 0.82](cite:E7) and more."
    (u, tail) = split_units(t, FINAL_CITE)
    assert u.cites == ["E7"] and "Source" not in u.text and t[u.end - 1] == ")"
