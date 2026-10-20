"""Chains Synthesis -> Writer -> Fact-Checker with a fake LLM to verify the final report's shape."""
import re
import tempfile

from app.agents import factchecker
from app.agents.synthesis import synthesize
from app.agents.writer import write_report
from app.schemas import Claim, ExtractionResult, SubTask, TaskGraph
from app.scratch_store import ScratchStore
from app.tools.fetch import Fetched


def fake_llm(prompt, **kw):
    items = re.findall(r"^\[(E\d+)\] (.*?) \(", prompt, flags=re.M)
    return "\n".join(f"- {claim} {{{eid}}}" for eid, claim in items[:3])


def build():
    st = ScratchStore("stage_run", root=tempfile.mkdtemp())
    texts = {"t1": ["Acme raised $20 million in Series B funding led by Sequoia in March 2026.",
                    "Acme prices its Pro plan at $12 per employee per month in Indonesia."],
             "t2": ["Beta Corp launched a payroll API for Vietnamese SMEs in May 2026."]}
    for sub, claims in texts.items():
        cl = []
        for k, t in enumerate(claims):
            ref = st.write_source(sub, f"https://site{sub}{k}.com/a", "s", kind="web", title=f"Title {sub}{k}",
                                  published_date="2026-06-01", body_text=t * 5, credibility_score=0.8)
            cl.append(Claim(id=f"c{sub}{k}", source_id=ref.source_id, sub_task_id=sub, text=t,
                            paragraph=f"PARAGRAPH: {t}", entities=[t.split()[0]]))
        st.write_extraction(ExtractionResult(sub_task_id=sub, claims=cl))
    g = TaskGraph(sub_tasks=[SubTask(id="t1", query="q1", title="Pricing and funding"), SubTask(id="t2", query="q2", title="Product gaps")])
    return st, g


def test_report_structure_citations_and_factcheck():
    st, g = build()
    bundle = synthesize(st.run_id, "payroll landscape", g, store=st)
    res = write_report(st.run_id, "payroll landscape", bundle, store=st, llm=fake_llm)
    md = res.markdown

    heads = re.findall(r"^## (.+)$", md, flags=re.M)
    assert heads[0] == "Executive Summary" and heads[-1] == "Gaps & Limitations"
    assert heads[1:-1] == ["Pricing and funding", "Product gaps"]                       # headings come from the task graph
    for h in heads[1:-1]:
        assert re.search(rf"## {re.escape(h)}\n\n\*\*Confidence: (HIGH|MEDIUM|LOW)\*\*", md)  # label right under the heading
    assert re.search(r"\[Source: site\w+\.com, 2026-06-01, credibility: 0\.\d\d\]\(cite:E\d+\)", md)
    assert res.citations and all(c["paragraph"].startswith("PARAGRAPH") for c in res.citations.values())
    assert res.retries == 0 and res.removed == []

    orig = factchecker.fetch_article
    factchecker.fetch_article = lambda url: Fetched(True, url, text="page " * 100)
    try:
        out, fc = factchecker.fact_check(st.run_id, md, res.citations, store=st, similarity=lambda c, p: 0.9)
    finally:
        factchecker.fetch_article = orig
    assert fc.total_checked >= 3 and fc.pass_rate == 1.0 and "[UNVERIFIED]" not in out


def test_every_claim_in_report_traces_to_the_bundle():
    st, g = build()
    bundle = synthesize(st.run_id, "x", g, store=st)
    res = write_report(st.run_id, "x", bundle, store=st, llm=fake_llm)
    ids = {i.id for s in bundle.sections for i in s.evidence}
    assert set(re.findall(r"\(cite:(E\d+)\)", res.markdown)) <= ids
