"""Find exactly where the pipeline freezes:   python -m scripts.diagnose        (add --llm to also test Groq)

Every step runs in its own thread with a time limit, so this command ALWAYS finishes and prints
PASS / FAIL / TIMEOUT with the seconds taken. Send the whole output when asking for help.
"""
import os
import sys
import threading
import time


def step(name, fn, timeout=90):
    box = {}

    def run():
        try:
            box["out"] = fn()
        except Exception as e:  # noqa: BLE001
            box["err"] = f"{type(e).__name__}: {str(e)[:200]}"

    t0 = time.time()
    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout)
    dt = time.time() - t0
    if th.is_alive():
        print(f"TIMEOUT {name:<34} stuck after {timeout}s  <-- THIS is where it freezes")
        return False
    if "err" in box:
        print(f"FAIL    {name:<34} {box['err']}  ({dt:.1f}s)")
        return False
    print(f"PASS    {name:<34} {box.get('out') or ''}  ({dt:.1f}s)")
    return True


def main():
    ok = []
    from app.config import settings

    ok.append(step("settings + .env", lambda: f"embeddings={settings.embeddings_backend}, db={settings.db_path}"))
    from app import db

    def check_db():
        db.init_db()   # safe to repeat: creates tables only if missing
        stuck = db.q("select id from runs where status in ('PLANNING','RESEARCHING')")
        return f"{len(db.q('select id from runs'))} runs stored, {len(stuck)} still marked PLANNING/RESEARCHING (stuck or old; harmless)"

    ok.append(step("database", check_db))
    from app import embeddings, memory, vector

    ok.append(step("embedding model load", lambda: embeddings.backend_name(), timeout=200))
    ok.append(step("embed one sentence", lambda: str(embeddings.embed(["hello"]).shape)))
    ok.append(step("qdrant open + search", lambda: f"{len(vector.search('entities', embeddings.embed(['x'])[0], 1))} hit(s)"))
    ok.append(step("memory.related_entities", lambda: str(memory.related_entities("payroll software India"))[:80]))
    if "--llm" in sys.argv:
        from app.llm import call_llm

        ok.append(step("Groq tiny call (~50 tokens)", lambda: call_llm("Reply with one word: ok", model=settings.fast_model, max_retries=0)[:30]))
    print("\nALL STEPS PASSED" if all(ok) else "\nAT LEAST ONE STEP FAILED: send this output")
    sys.stdout.flush()
    os._exit(0 if all(ok) else 1)   # do not wait for abandoned threads


if __name__ == "__main__":
    main()
