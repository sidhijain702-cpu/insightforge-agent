"""Turn a REAL run's metrics into the 'context blowout prevented' paragraph for your README.

    python -m scripts.context_report <run_id>

Everything printed comes from numbers logged during that run (runs.metrics in SQLite).
"""
import sys

from app import db
from app.events import get_metrics


def main(run_id: str) -> None:
    m = get_metrics(run_id)
    if not m:
        sys.exit(f"no metrics for run {run_id}")
    raw = m.get("raw_tokens_fetched", 0)
    n_src = m.get("sources_fetched", 0)
    before = m.get("bundle_tokens_before_compression", 0)
    final = m.get("bundle_tokens_final", 0)
    prompts = m.get("agent_prompt_tokens", [])
    biggest = max(prompts, key=lambda p: p["tokens"]) if prompts else None
    largest_src = max(m.get("source_log", []), key=lambda s: s["tokens"], default=None)

    print(f"RUN {run_id}")
    print(f"  sources fetched ............ {n_src}")
    print(f"  raw page text fetched ...... ~{raw:,} tokens")
    if largest_src:
        print(f"  largest single page ........ ~{largest_src['tokens']:,} tokens ({largest_src['domain']})")
    if biggest:
        print(f"  largest prompt any agent saw  ~{biggest['tokens']:,} tokens ({biggest['agent']}, {biggest['sub_task']})")
    print(f"  evidence bundle ............ {before:,} -> {final:,} tokens after {m.get('compression_steps', 0)} compression step(s)")
    print(f"  subagent tool calls ........ {len(m.get('subagent_calls', []))}   fallback sub-tasks: {m.get('fallback_subtasks', 0)}")
    print(f"  writer retries / removed ... {m.get('writer_retries', 0)} / {m.get('writer_statements_removed', 0)}")
    print(f"  fact-check pass rate ....... {m.get('factcheck_pass_rate')}")

    if raw and biggest:
        cap = 8000   # tokens/minute limit observed on the free Groq tier for gpt-oss models; edit if yours differs
        print("\n--- README paragraph (edit wording, keep the numbers) ---\n")
        print(
            f"In run `{run_id}` the Web Search agents fetched {n_src} full articles totalling ~{raw:,} tokens of raw text. "
            f"Placed in a single prompt, that is about {raw / cap:.0f}x the {cap:,} tokens-per-minute limit of the free Groq "
            f"tier used here, so it would be rejected before any report was written. Instead, raw text stayed in the scratch "
            f"store; the largest prompt any agent actually received was ~{biggest['tokens']:,} tokens ({biggest['agent']}). "
            f"Synthesis produced an evidence bundle of {final:,} tokens ({m.get('compression_steps', 0)} compression step(s) at the "
            f"configured budget). Use `python -m scripts.resynth {run_id} <smaller budget>` to demonstrate compression on this data."
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m scripts.context_report <run_id>")
        rows = db.q("SELECT id, brief FROM runs ORDER BY created_at DESC LIMIT 5")
        for r in rows:
            print(f"  {r['id']}  {r['brief'][:70]}")
        sys.exit(1)
    main(sys.argv[1])
