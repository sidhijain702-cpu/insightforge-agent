from langchain_groq import ChatGroq

from app.agents.researchers import EXTRACT_SYSTEM
from app.config import settings

PROMPT = (
 'Extract facts. Return JSON: {"claims":[{"source":"S1","text":"...","quote":"...","entities":["..."]}],'
 '"entities":[],"relationships":[]}\n\n'
 "[S1] Test (example.com, 2026-01-01)\nAcme Pay raised $20 million in Series B funding in March 2026, "
 "led by Sequoia. It serves 3,000 employers in Indonesia and Vietnam."
 )
SYSTEM = EXTRACT_SYSTEM + "\nRespond with a single valid JSON object only."

def show(label, llm):
 out = llm.invoke([("system", SYSTEM), ("human", PROMPT)])
 meta = out.response_metadata
 usage = meta.get("token_usage", {})
 print(label, "| content_len:", len(str(out.content)), "| finish:", meta.get("finish_reason"), "| completion_tokens:", usage.get("completion_tokens"))
 print(" ", repr(out.content)[:200])

BASE = dict(model=settings.fast_model, temperature=0.1, api_key=settings.groq_api_key, max_retries=0)
JSON_MODE = {"type": "json_object"}

show("A json_mode default", ChatGroq(**BASE).bind(response_format=JSON_MODE))
show("B json_mode 4096", ChatGroq(**BASE, max_tokens=4096).bind(response_format=JSON_MODE))
show("C plain 4096", ChatGroq(**BASE, max_tokens=4096))