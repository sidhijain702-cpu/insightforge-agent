"""Check your setup BEFORE a full run:   python -m scripts.smoke

Verifies: Groq key + models respond, tool-calling works, search provider works, embeddings load,
Qdrant opens, DeepAgents imports. Prints a clear PASS/FAIL per item so you debug one thing at a time.
"""
import sys
import traceback

from app.config import settings


def check(name, fn):
    try:
        out = fn()
        print(f"PASS  {name}" + (f"  -> {out}" if out else ""))
        return True
    except Exception as e:  # noqa: BLE001
        print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:200]}")
        if "-v" in sys.argv:
            traceback.print_exc()
        return False


def groq(model):
    from app.llm import call_llm
    return call_llm("Reply with exactly the word: ready", model=model).strip()[:20]


def tool_calling():
    from langchain_core.tools import tool
    from app.llm import get_chat

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    msg = get_chat(settings.planner_model, 0).bind_tools([add]).invoke("What is 17 + 25? Use the tool.")
    assert msg.tool_calls, "planner model did not produce a tool call (try another PLANNER_MODEL)"
    return msg.tool_calls[0]["args"]


def search():
    from app.tools.search_providers import web_search
    r = web_search("B2B payroll software Singapore", 3)
    assert r, "no results"
    return f"{len(r)} results, first: {r[0]['url'][:60]}"


def fetch():
    from app.tools.fetch import fetch_article
    f = fetch_article("https://www.python.org/about/")
    assert f.ok, f.skipped_reason
    return f"{len(f.text)} chars, title={f.title[:40]!r}"


def embeddings():
    from app.embeddings import backend_name, embed
    v = embed(["hello world"])
    return f"backend={backend_name()} dim={v.shape[1]}"


def qdrant():
    from app import vector
    from app.embeddings import embed
    vector.upsert("smoke_test", embed(["a"]), [{"x": 1}])
    return f"hits={len(vector.search('smoke_test', embed(['a'])[0], 1))}"


def deepagents():
    import deepagents
    from deepagents import create_deep_agent
    import inspect
    return f"params={list(inspect.signature(create_deep_agent).parameters)[:8]}"


if __name__ == "__main__":
    results = [
        check(f"Groq fast model ({settings.fast_model})", lambda: groq(settings.fast_model)),
        check(f"Groq planner model ({settings.planner_model})", lambda: groq(settings.planner_model)),
        check("Planner model tool-calling", tool_calling),
        check(f"Search provider ({settings.search_provider})", search),
        check("Article fetch + parse", fetch),
        check("Embeddings", embeddings),
        check("Qdrant", qdrant),
        check("deepagents import", deepagents),
    ]
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)
