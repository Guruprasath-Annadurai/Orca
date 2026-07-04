"""
Explainability — "Explain this answer" backing store.

Every assistant message that goes through /api/stream gets an ExplainRecord
capturing exactly what produced it: the retrieval chain (query rewriting,
decomposition, HyDE, RRF fusion, reranking, sufficiency check), which tools
fired, the sub-agent's plan decision, and citation-level sourcing.

Records are kept in-memory per session (capped, oldest evicted first) — this
is working-session data, not a permanent audit record. For permanent
provenance, the audit log (orca/audit.py) already captures the high-level
event; this module captures the *reasoning*, which is too large/verbose to
put in every audit row.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

MAX_RECORDS_PER_SESSION = 50


@dataclass
class RetrievalStep:
    stage: str
    detail: str
    duration_ms: int = 0


@dataclass
class ExplainRecord:
    message_id: str
    created_at: float = field(default_factory=time.time)

    # Query intelligence
    original_query: str = ""
    rewritten_query: str = ""
    query_complexity: str = "simple"
    sub_queries: list[str] = field(default_factory=list)
    expanded_queries: list[str] = field(default_factory=list)
    used_hyde: bool = False

    # Retrieval chain
    retrieval_steps: list[RetrievalStep] = field(default_factory=list)
    stage_timings_ms: dict = field(default_factory=dict)
    retrieval_rounds: int = 1

    # Confidence & citations
    sufficiency_confidence: float = 0.0
    sources: list[str] = field(default_factory=list)
    citation_dna: dict = field(default_factory=dict)
    contradictions: list[str] = field(default_factory=list)

    # Agent reasoning
    plan_action: str = "direct"     # direct | tools
    tools_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "created_at": self.created_at,
            "query_intelligence": {
                "original_query": self.original_query,
                "rewritten_query": self.rewritten_query,
                "complexity": self.query_complexity,
                "sub_queries": self.sub_queries,
                "expanded_queries": self.expanded_queries,
                "used_hyde": self.used_hyde,
            },
            "retrieval_chain": {
                "steps": [{"stage": s.stage, "detail": s.detail, "duration_ms": s.duration_ms}
                          for s in self.retrieval_steps],
                "stage_timings_ms": self.stage_timings_ms,
                "rounds": self.retrieval_rounds,
            },
            "confidence": {
                "sufficiency_confidence": self.sufficiency_confidence,
                "sources": self.sources,
                "contradictions": self.contradictions,
            },
            "citations": self.citation_dna,
            "reasoning": {
                "plan_action": self.plan_action,
                "tools_used": self.tools_used,
            },
        }


class ExplainStore:
    """Per-session ring buffer of ExplainRecords, keyed by message_id."""

    def __init__(self):
        self._records: dict[str, ExplainRecord] = {}
        self._order: list[str] = []

    def add(self, record: ExplainRecord) -> None:
        self._records[record.message_id] = record
        self._order.append(record.message_id)
        while len(self._order) > MAX_RECORDS_PER_SESSION:
            oldest = self._order.pop(0)
            self._records.pop(oldest, None)

    def get(self, message_id: str) -> ExplainRecord | None:
        return self._records.get(message_id)


def build_from_rag_result(message_id: str, rag_result, plan_action: str, tools_used: list[str]) -> ExplainRecord:
    """Construct an ExplainRecord from a RAGResult (orca.docs.pipeline.RAGResult) plus agent trace data."""
    record = ExplainRecord(message_id=message_id, plan_action=plan_action, tools_used=tools_used)

    if rag_result is None:
        return record

    plan = rag_result.query_plan
    if plan:
        record.original_query = plan.original
        record.rewritten_query = plan.rewritten
        record.query_complexity = plan.complexity
        record.sub_queries = plan.sub_queries
        record.expanded_queries = plan.expanded_queries
        record.used_hyde = bool(plan.hyde_text)

    record.stage_timings_ms = rag_result.stage_timings_ms
    record.retrieval_rounds = rag_result.retrieval_rounds
    record.sufficiency_confidence = rag_result.sufficiency_confidence
    record.sources = rag_result.sources
    record.citation_dna = rag_result.citation_dna
    record.contradictions = rag_result.contradictions

    # Human-readable step-by-step retrieval chain, built from timings
    stage_labels = {
        "query_intelligence":   "Rewrote query, classified complexity, generated variants",
        "multi_signal_recall":  "Ran dense + keyword retrieval across query variants",
        "rrf_fusion":           "Merged retrieval results via Reciprocal Rank Fusion",
        "lexical_prefilter":    "Filtered candidates by keyword overlap before reranking",
        "cross_encoder_rerank": "Reranked candidates using the LLM as a cross-encoder",
        "sufficiency_check":    "Judged whether retrieved context answers the query",
    }
    for stage, ms in record.stage_timings_ms.items():
        record.retrieval_steps.append(RetrievalStep(
            stage=stage, detail=stage_labels.get(stage, stage), duration_ms=ms,
        ))

    return record
