import tempfile

from app.agents.synthesis import bundle_tokens, synthesize
from app.schemas import Claim, EntityGraph, ExtractionResult, Relationship, SubTask, TaskGraph
from app.scratch_store import ScratchStore


def make_store():
    return ScratchStore("run_test_" + tempfile.mkdtemp()[-6:], root=tempfile.mkdtemp())


def add(store, sub, url, text, date="2026-05-01", cred=0.8, entities=("Acme",)):
    ref = store.write_source(sub, url, "summary", kind="web", title="T", published_date=date,
                             body_text="body " * 50, credibility_score=cred)
    return Claim(id=f"c_{ref.source_id}", source_id=ref.source_id, sub_task_id=sub, text=text,
                 paragraph="The exact paragraph. " + text, entities=list(entities))


def graph(*subs):
    return TaskGraph(sub_tasks=[SubTask(id=s, query=f"query {s}", title=f"Theme {s}") for s in subs])


def test_duplicates_merge_and_confidence_becomes_high():
    st = make_store()
    same = "Acme raised $20 million in Series B funding led by Sequoia in March 2026."
    c1 = add(st, "t1", "https://reuters.com/a", same, cred=0.9)
    c2 = add(st, "t1", "https://techcrunch.com/b", same, cred=0.7)
    c3 = add(st, "t1", "https://blog.example.io/c", "Beta Corp launched a payroll API for Indonesian SMEs in May 2026.", cred=0.3, entities=("Beta Corp",))
    st.write_extraction(ExtractionResult(sub_task_id="t1", claims=[c1, c2, c3]))
    b = synthesize(st.run_id, "payroll", graph("t1"), store=st)
    items = b.sections[0].evidence
    assert len(items) == 2
    top = next(i for i in items if "Series B" in i.claim)
    assert top.domain == "reuters.com" and top.confidence == "HIGH" and "techcrunch.com" in top.supporting_domains
    assert next(i for i in items if "Beta" in i.claim).confidence == "LOW"       # single low-credibility source
    assert [i.id for i in items] == ["E1", "E2"]


def test_conflicting_figures_are_flagged_low_confidence():
    st = make_store()
    c1 = add(st, "t1", "https://a.com/x", "Acme raised $20 million in its latest funding round.")
    c2 = add(st, "t1", "https://b.com/y", "Acme secured $35 million in new funding last quarter.", cred=0.7)
    rels = [Relationship(subject="Acme", predicate="raised", object="$20 million", source_url="https://a.com/x"),
            Relationship(subject="Acme", predicate="raised", object="$35 million", source_url="https://b.com/y")]
    st.write_extraction(ExtractionResult(sub_task_id="t1", claims=[c1, c2], graph=EntityGraph(relationships=rels)))
    b = synthesize(st.run_id, "funding", graph("t1"), store=st)
    assert b.conflicts and all(i.conflict and i.confidence == "LOW" for i in b.sections[0].evidence)
    assert b.sections[0].confidence == "LOW"


def test_ranking_is_recency_first():
    st = make_store()
    old = add(st, "t1", "https://reuters.com/old", "Gamma Ltd opened an office in Jakarta serving two hundred clients.", date="2024-01-01", cred=0.95, entities=("Gamma",))
    new = add(st, "t1", "https://x.io/new", "Delta Inc introduced annual billing discounts for startups across Vietnam.", date="2026-09-01", cred=0.40, entities=("Delta",))
    st.write_extraction(ExtractionResult(sub_task_id="t1", claims=[old, new]))
    items = synthesize(st.run_id, "x", graph("t1"), store=st).sections[0].evidence
    assert "Delta" in items[0].claim            # newer beats more credible-but-older


def test_token_budget_is_enforced_by_compression_with_a_log():
    st = make_store()
    claims = []
    for n in range(40):
        claims.append(add(st, "t1" if n % 2 else "t2", f"https://site{n}.com/p",
                          f"Company{n} announced product{n} with feature{n} pricing{n} for region{n} during quarter{n} "
                          f"after partnership{n} and expansion{n} into market{n}.",
                          date=f"2026-0{1 + n % 9}-1{n % 9}", entities=(f"Company{n}",)))
    for sub in ("t1", "t2"):
        st.write_extraction(ExtractionResult(sub_task_id=sub, claims=[c for c in claims if c.sub_task_id == sub]))
    budget = 450
    b = synthesize(st.run_id, "brief", graph("t1", "t2"), budget=budget, store=st,
                   compress_fn=lambda items, theme: "; ".join(" ".join(i.claim.split()[:6]) for i in items))
    assert bundle_tokens(b) <= budget and b.token_count <= budget
    assert b.compression_log, "compression must be logged, never silent"
    for e in b.compression_log:
        assert e.reason and e.tokens_after < e.tokens_before and e.items_affected >= 1
    assert any(i.compressed for s in b.sections for i in s.evidence)
    assert all(s.evidence for s in b.sections)    # nothing disappeared completely
    assert any("condensed" in g for g in b.gaps)


def test_no_evidence_section_is_kept_and_reported():
    st = make_store()
    b = synthesize(st.run_id, "x", graph("t1"), failed={"t1": "no readable sources"}, store=st)
    assert b.sections[0].evidence == [] and b.sections[0].confidence == "LOW"
    assert any("no readable sources" in g for g in b.gaps)


def test_compression_summary_that_invents_content_is_rejected():
    import app.llm as llm
    from app.agents.synthesis import _default_compress
    from app.schemas import EvidenceItem

    def item(i, claim):
        return EvidenceItem(id=f"E{i}", claim=claim, source_id="s", url="https://a.com", domain="a.com", credibility=0.5)

    items = [item(1, "Apptio Cloudability competes with Finout on multi-cloud cost views."),
             item(2, "Datadog charges a usage fee and adds charges when limits are exceeded.")]
    orig = llm.call_llm
    try:
        llm.call_llm = lambda *a, **k: "Apptio competes with Finout while Informatica pivoted to pricing for $3.2 billion."
        bad = _default_compress(items, "Pricing")
        llm.call_llm = lambda *a, **k: "Apptio Cloudability competes with Finout; Datadog charges usage fees and extra charges over limits."
        good = _default_compress(items, "Pricing")
    finally:
        llm.call_llm = orig
    assert "Informatica" not in bad and "$3.2" not in bad          # fell back to the deterministic merge
    assert "Datadog charges usage fees" in good                    # a faithful summary is accepted
