from app.agents.planner import build_delegation_message, clean_graph
from app.config import settings
from app.schemas import SubTask, TaskGraph


def test_delegation_prompt_never_contains_the_brief_or_other_subtask_text_leaks():
    brief = "SECRET-BRIEF-TEXT competitive landscape for B2B payroll in Southeast Asia"
    g = TaskGraph(sub_tasks=[SubTask(id="t1", query="payroll vendors pricing Indonesia"),
                             SubTask(id="t2", query="payroll funding rounds Vietnam")])
    msg = build_delegation_message(g)
    assert "SECRET-BRIEF-TEXT" not in msg and brief not in msg
    assert "sub_task_id=t1" in msg and "sub_task_id=t2" in msg and "task" in msg


def test_clean_graph_caps_parallelism_and_coerces_disabled_source_types():
    raw = [{"query": f"q{i}", "source_type": "uploaded_docs", "priority": i % 5 + 1} for i in range(10)]
    g = clean_graph(raw, enabled=["web"])
    assert len(g.sub_tasks) == settings.max_parallel_subtasks <= 6
    assert all(s.source_type == "web" for s in g.sub_tasks)
    assert [s.id for s in g.sub_tasks] == [f"t{i}" for i in range(1, len(g.sub_tasks) + 1)]


def test_clean_graph_rejects_garbage_with_actionable_message():
    for bad, msg in (([], "at least 1"), ([{"title": "x"}], "no 'query'")):
        try:
            clean_graph(bad, ["web"])
            raise AssertionError("should have raised")
        except ValueError as e:
            assert msg in str(e)
