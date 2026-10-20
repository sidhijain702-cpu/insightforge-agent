"""Typed contracts between every agent.

LEARNING NOTE
-------------
Two kinds of objects exist on purpose:

* HEAVY objects (Source, Claim) live in the scratch store / Qdrant. They hold raw
  text and are loaded only by the agent that really needs them.
* LIGHT objects (SourceRef, EvidenceItem, TaskGraph) are what flows through the
  LangGraph state and through prompts. They are small pointers/summaries.

If you ever see raw page text inside RunState, that is a context-engineering bug.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["web", "uploaded_docs", "prior_reports"]
SourceKind = Literal["web", "doc_chunk", "memory"]
Confidence = Literal["HIGH", "MEDIUM", "LOW"]
RunStatus = Literal[
    "PLANNING", "RESEARCHING", "SYNTHESIZING", "WRITING", "FACT_CHECKING",
    "COMPLETE", "NEEDS_CLARIFICATION", "FAILED",
]


# ---------------------------------------------------------------- planning
class SubTask(BaseModel):
    id: str
    query: str
    source_type: SourceType = "web"
    time_horizon: str = "last 18 months"
    priority: int = Field(default=3, ge=1, le=5)
    title: str = ""  # short heading used as the report section title

    def theme(self) -> str:
        if self.title.strip():
            return self.title.strip()
        words = self.query.split()
        return " ".join(words[:8]) + ("..." if len(words) > 8 else "")


class TaskGraph(BaseModel):
    sub_tasks: list[SubTask] = Field(min_length=1, max_length=6)


class SourcePrefs(BaseModel):
    web: bool = True
    docs: bool = True
    prior: bool = True


# ---------------------------------------------------------------- researchers
class Source(BaseModel):
    """FULL record, stored in the scratch store. Never put this in a prompt by default."""
    id: str
    run_id: str
    sub_task_id: str
    kind: SourceKind = "web"
    url: str
    domain: str = ""
    title: str = ""
    published_date: str | None = None
    author: str | None = None
    body_text: str = ""           # empty for doc_chunk: resolved on demand from Qdrant
    chunk_id: str | None = None   # Qdrant point id for doc chunks
    credibility_score: float = Field(default=0.3, ge=0, le=1)
    meta: dict = Field(default_factory=dict)


class SourceRef(BaseModel):
    """What a researcher hands BACK to the Planner: a pointer + one line. Nothing else."""
    source_id: str
    summary: str


class Claim(BaseModel):
    """One atomic factual statement extracted from a source."""
    id: str
    source_id: str
    sub_task_id: str
    text: str
    paragraph: str = ""           # exact supporting passage (shown in the citation drawer)
    entities: list[str] = Field(default_factory=list)


class Entity(BaseModel):
    name: str
    type: str = "other"


class Relationship(BaseModel):
    subject: str
    predicate: str
    object: str
    source_url: str = ""
    date: str | None = None
    conflict: bool = False


class EntityGraph(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    sub_task_id: str
    claims: list[Claim] = Field(default_factory=list)
    graph: EntityGraph = Field(default_factory=EntityGraph)


# ---------------------------------------------------------------- synthesis / writer
class EvidenceItem(BaseModel):
    id: str                       # E1, E2 ... what the Writer cites
    claim: str
    source_id: str
    url: str
    domain: str
    date: str | None = None
    credibility: float
    confidence: Confidence = "MEDIUM"
    conflict: bool = False
    conflict_note: str = ""
    claim_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    supporting_domains: list[str] = Field(default_factory=list)
    compressed: bool = False
    relevance: float = 0.0


class EvidenceSection(BaseModel):
    sub_task_id: str
    theme: str
    confidence: Confidence = "LOW"
    evidence: list[EvidenceItem] = Field(default_factory=list)


class CompressionLogEntry(BaseModel):
    section: str
    action: Literal["compressed", "dropped"]
    items_affected: int
    source_ids: list[str] = Field(default_factory=list)
    reason: str
    tokens_before: int
    tokens_after: int


class EvidenceBundle(BaseModel):
    sections: list[EvidenceSection] = Field(default_factory=list)
    token_count: int = 0
    token_budget: int = 6000
    compression_log: list[CompressionLogEntry] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)


class FactCheckResult(BaseModel):
    total_checked: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    unverified_claims: list[dict] = Field(default_factory=list)
    details: list[dict] = Field(default_factory=list)


class ChangeSummary(BaseModel):
    added_findings: list[str] = Field(default_factory=list)
    removed_findings: list[str] = Field(default_factory=list)
    changed_findings: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.added_findings or self.removed_findings or self.changed_findings)


# ---------------------------------------------------------------- LangGraph state
class RunState(BaseModel):
    """The typed state object LangGraph passes between nodes.

    Notice what is NOT here: page text. Only ids, summaries, small bundles.
    """
    run_id: str
    brief: str
    prefs: SourcePrefs = Field(default_factory=SourcePrefs)
    allow_clarification: bool = True
    watch_id: int | None = None

    status: RunStatus = "PLANNING"
    clarifying_question: str | None = None
    task_graph: TaskGraph | None = None
    refs: dict[str, list[SourceRef]] = Field(default_factory=dict)  # sub_task_id -> refs
    failed_subtasks: dict[str, str] = Field(default_factory=dict)   # sub_task_id -> reason
    bundle: EvidenceBundle | None = None
    report_markdown: str = ""
    citations: dict[str, dict] = Field(default_factory=dict)
    factcheck: FactCheckResult | None = None
    report_id: str | None = None
    error: str | None = None
