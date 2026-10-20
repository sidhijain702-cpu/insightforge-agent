from app.entities import flag_conflicts, parse_amount
from app.schemas import Relationship


def test_parse_amount():
    assert parse_amount("$20 million") == 20e6
    assert parse_amount("US$1.2bn") == 1.2e9
    assert parse_amount("35M") == 35e6
    assert parse_amount("Series B") is None


def R(obj, url, pred="raised", subj="Acme"):
    return Relationship(subject=subj, predicate=pred, object=obj, source_url=url)


def test_conflicting_funding_is_flagged():
    rels = flag_conflicts([R("$20 million", "https://a.com/x"), R("$35M", "https://b.com/y")])
    assert all(r.conflict for r in rels)


def test_same_value_different_format_is_not_a_conflict():
    rels = flag_conflicts([R("$20 million", "https://a.com/x"), R("$20M", "https://b.com/y")])
    assert not any(r.conflict for r in rels)


def test_same_source_never_conflicts_with_itself_and_multi_valued_ignored():
    assert not any(r.conflict for r in flag_conflicts([R("$1M", "https://a.com"), R("$9M", "https://a.com")]))
    assert not any(r.conflict for r in flag_conflicts([R("X", "https://a.com", "partnered_with"), R("Y", "https://b.com", "partnered_with")]))
