"""Groq access: one place for throttling, retry/backoff, reasoning control and JSON parsing.

LEARNING NOTE
-------------
* Groq's free tier limits requests AND tokens per minute, so every call goes through `call_llm`,
  which limits concurrency (semaphore) and sleeps + retries on "rate limit, try again in 7.2s".
* `gpt-oss` models are REASONING models: they spend part of the output budget on hidden thinking
  before writing the visible answer. If the budget runs out mid-thought the visible text is EMPTY
  (finish_reason == "length"). We therefore (1) ask for low reasoning effort, (2) set an explicit
  output limit, and (3) automatically retry with a bigger limit when we see an empty/length reply.
* Groq counts requested tokens (prompt + max_tokens) against the per-minute limit, so we do not
  set the output limit higher than needed.
"""
import json
import logging
import os
import re
import threading
import time
from typing import TypeVar

from pydantic import BaseModel

from app.config import settings

log = logging.getLogger("insightforge.llm")
T = TypeVar("T", bound=BaseModel)

MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "3000"))
MAX_TOKENS_CEILING = int(os.environ.get("LLM_MAX_TOKENS_CEILING", "8000"))
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "low")   # low | medium | high ("" disables)

_sem = threading.Semaphore(max(1, settings.max_llm_concurrency))
_models: dict[tuple, object] = {}
_effort_supported = True


def _is_reasoning_model(model: str) -> bool:
    return "gpt-oss" in model


def get_chat(model: str | None = None, temperature: float = 0.1, json_mode: bool = False, max_tokens: int | None = None,
             sdk_retries: int = 0):
    """Cached ChatGroq instance. Imported lazily so tests can run without langchain."""
    from langchain_groq import ChatGroq

    model = model or settings.fast_model
    max_tokens = max_tokens or MAX_TOKENS
    use_effort = _effort_supported and REASONING_EFFORT and _is_reasoning_model(model)
    key = (model, temperature, json_mode, max_tokens, bool(use_effort), sdk_retries)
    if key not in _models:
        kwargs = dict(model=model, temperature=temperature, api_key=settings.groq_api_key, max_retries=sdk_retries, max_tokens=max_tokens)
        if use_effort:
            kwargs["reasoning_effort"] = REASONING_EFFORT
        llm = ChatGroq(**kwargs)
        if json_mode:
            llm = llm.bind(response_format={"type": "json_object"})
        _models[key] = llm
    return _models[key]


def _retry_after(msg: str) -> float:
    m = re.search(r"try again in (?:(\d+)m)?\s*([\d.]+)?s?", msg)
    if not m:
        return 0
    return float(m.group(1) or 0) * 60 + float(m.group(2) or 0)


def _is_rate_limit(err: Exception) -> bool:
    s = str(err).lower()
    return "429" in s or "rate limit" in s or "rate_limit" in s


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):   # some providers return content blocks
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def call_llm(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.1,
    json_mode: bool = False,
    max_retries: int = 20,
    max_tokens: int | None = None,
) -> str:
    """Single chat completion with throttling, rate-limit retry and empty-reply recovery. Returns text."""
    global _effort_supported
    msgs = []
    if system:
        msgs.append(("system", system))
    msgs.append(("human", prompt))
    delay = 2.0
    limit = max_tokens or MAX_TOKENS
    for attempt in range(max_retries + 1):
        try:
            with _sem:
                out = get_chat(model, temperature, json_mode, limit).invoke(msgs)
            text = _text(out.content)
            if text.strip():
                return text
            meta = getattr(out, "response_metadata", {}) or {}
            finish = meta.get("finish_reason")
            usage = meta.get("token_usage", {})
            if finish == "length" and limit < MAX_TOKENS_CEILING and attempt < max_retries:
                limit = min(limit * 2, MAX_TOKENS_CEILING)
                log.warning("empty reply (hit output limit, used %s tokens); retrying with max_tokens=%d",
                            usage.get("completion_tokens"), limit)
                continue
            raise ValueError(f"model returned an empty reply (finish_reason={finish}, usage={usage})")
        except Exception as e:  # noqa: BLE001
            if attempt >= max_retries:
                raise
            msg = str(e)
            if _is_rate_limit(e):
                wait = max(_retry_after(msg), delay) + 0.5
                log.warning("rate limited, sleeping %.1fs (attempt %d)", wait, attempt + 1)
                time.sleep(min(wait, 90))
                delay = min(delay * 2, 30)
            elif "reasoning_effort" in msg and _effort_supported:
                log.warning("model rejected reasoning_effort; disabling it")
                _effort_supported = False
            elif json_mode and "json" in msg.lower():
                json_mode = False  # model rejected JSON mode -> retry in plain mode
            else:
                raise
    raise RuntimeError("unreachable")


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (handles ```json fences + chatter)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in model output")
    return json.loads(t[start : end + 1])


def call_json(prompt: str, schema: type[T], system: str | None = None, model: str | None = None) -> T:
    """Ask for JSON and validate it against a Pydantic schema; one self-repair retry.
    On failure the error includes the start of the raw reply so you can SEE what the model said."""
    sys_msg = (system or "") + "\nRespond with a single valid JSON object only. No prose."
    last_err = ""
    for attempt in range(2):
        p = prompt if attempt == 0 else f"{prompt}\n\nYour previous JSON was invalid: {last_err[:200]}\nReturn corrected JSON only."
        raw = call_llm(p, system=sys_msg, model=model, json_mode=True)
        try:
            return schema.model_validate(extract_json(raw))
        except Exception as e:  # noqa: BLE001
            last_err = f"{str(e)[:200]} | raw reply starts: {raw[:200]!r}"
            log.warning("invalid %s JSON (attempt %d): %s", schema.__name__, attempt + 1, last_err)
    raise ValueError(f"LLM did not return valid {schema.__name__}: {last_err}")
