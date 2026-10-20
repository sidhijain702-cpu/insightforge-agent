"""All tunables live here and come from environment variables / .env.

LEARNING NOTE: every knob the assignment calls "configurable" (max parallel
sub-tasks, token budget, fact-check threshold ...) is a field below. Nothing is
hard-coded inside agents, and no API key ever appears in source code.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- keys ---
    groq_api_key: str = ""
    tavily_api_key: str = ""
    brave_api_key: str = ""
    serpapi_api_key: str = ""

    # --- models ---
    planner_model: str = "openai/gpt-oss-120b"
    writer_model: str = "openai/gpt-oss-120b"
    fast_model: str = "openai/gpt-oss-20b"

    # --- search ---
    search_provider: str = "tavily"
    results_per_subtask: int = 5

    # --- context engineering ---
    max_parallel_subtasks: int = 6
    evidence_token_budget: int = 6000
    entity_chars_per_source: int = 1800
    claims_per_source: int = 4
    max_body_chars: int = 60000
    max_llm_concurrency: int = 2
    session_summaries_to_load: int = 3

    # --- planner ---
    planner_mode: str = "deepagent"  # deepagent | direct

    # --- writer ---
    writer_max_retries: int = 2
    writer_overlap_threshold: float = 0.30

    # --- fact check ---
    factcheck_threshold: float = 0.85
    factcheck_sample_rate: float = 0.2
    factcheck_min_claims: int = 5

    # --- embeddings / vectors ---
    embeddings_backend: str = "fastembed"  # fastembed | hash
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    qdrant_path: str = "data/qdrant"
    qdrant_url: str = ""

    # --- app ---
    data_dir: str = "data"
    jwt_secret: str = "change-me-in-real-use"
    share_ttl_days: int = 7
    frontend_url: str = "http://localhost:3000"
    demo_interval_seconds: int = 0

    @property
    def db_path(self) -> str:
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        return str(Path(self.data_dir) / "insightforge.sqlite3")

    @property
    def scratch_root(self) -> str:
        return str(Path(self.data_dir) / "scratch")

    @property
    def uploads_root(self) -> str:
        return str(Path(self.data_dir) / "uploads")


settings = Settings()
