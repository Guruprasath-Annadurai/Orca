"""
Deep RAG Pipeline — orchestrates the full 7-stage retrieval flow.

Stage 1: Query Intelligence   — rewrite, classify, decompose, expand, HyDE
Stage 2: Multi-Signal Recall  — dense (per query variant) + BM25 + HyDE vector
Stage 3: RRF Fusion           — merge all ranked lists into one
Stage 4: Lexical Prefilter    — cut candidates before expensive rerank
Stage 5: Cross-Encoder Rerank — LLM-as-reranker on (query, passage) pairs
Stage 6: Sufficiency Check    — self-reflective judge, one corrective retry
Stage 7: Citation DNA         — tag chunks D1/D2/... for inline citation

This is deliberately NOT a single black-box call — every stage is inspectable
and independently testable. Falls back gracefully at every stage if Ollama
is unavailable (degrades to plain BM25 retrieval, never crashes).
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from orca.docs.query_engine import build_query_plan, QueryPlan
from orca.docs.reranker import reciprocal_rank_fusion, cross_encoder_rerank, fast_lexical_prefilter, ScoredChunk
from orca.docs.sufficiency import check_sufficiency, detect_contradictions, make_citation_dna, format_context_with_citations


@dataclass
class RAGResult:
    context_block: str = ""            # formatted, citation-tagged context ready for system prompt
    citation_dna: dict = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    query_plan: Optional[QueryPlan] = None
    contradictions: list[str] = field(default_factory=list)
    sufficiency_confidence: float = 0.0
    retrieval_rounds: int = 1
    stage_timings_ms: dict = field(default_factory=dict)
    used_fallback: bool = False


def run_deep_rag(
    query: str,
    doc_store,                          # orca.docs.store.DocStore instance
    conversation_history: list[str] | None,
    ollama_host: str,
    llm_model: str,
    embed_model: str = "nomic-embed-text",
    *,
    top_k_final: int = 6,
    top_k_candidates: int = 20,
    enable_reranking: bool = True,
    enable_sufficiency_check: bool = True,
    enable_query_intelligence: bool = True,
    max_corrective_rounds: int = 1,
    fast_mode: bool = True,
) -> RAGResult:
    """
    Run the full deep RAG pipeline. Returns a RAGResult ready to inject into
    the chat system prompt via `context_block`.
    """
    result = RAGResult()
    timings: dict[str, int] = {}

    if doc_store.count() == 0:
        return result  # nothing indexed — skip RAG entirely

    # ── Stage 1: Query Intelligence ──────────────────────────────────────────
    t0 = time.monotonic()
    if enable_query_intelligence:
        plan = build_query_plan(
            query, conversation_history,
            ollama_host=ollama_host, llm_model=llm_model, embed_model=embed_model,
            fast_mode=fast_mode,
        )
    else:
        plan = QueryPlan(original=query, rewritten=query, expanded_queries=[query])
    result.query_plan = plan
    timings["query_intelligence"] = int((time.monotonic() - t0) * 1000)

    # ── Stage 2: Multi-Signal Recall ─────────────────────────────────────────
    t0 = time.monotonic()
    ranked_lists: list[list[dict]] = []

    # Sub-queries (multi-hop decomposition) each get their own retrieval
    queries_to_run = list(plan.expanded_queries) or [query]
    if plan.sub_queries:
        queries_to_run.extend(plan.sub_queries)

    for q in queries_to_run[:5]:  # cap fan-out
        hits = doc_store.retrieve(q, top_k=top_k_candidates)
        for h in hits:
            h["source"] = "dense_or_bm25"
        if hits:
            ranked_lists.append(hits)

    # HyDE: use the hypothetical-answer embedding as an extra query signal
    if plan.hyde_text:
        hyde_hits = doc_store.retrieve(plan.hyde_text, top_k=top_k_candidates)
        for h in hyde_hits:
            h["source"] = "hyde"
        if hyde_hits:
            ranked_lists.append(hyde_hits)

    timings["multi_signal_recall"] = int((time.monotonic() - t0) * 1000)

    if not ranked_lists:
        result.used_fallback = True
        result.stage_timings_ms = timings
        return result

    # ── Stage 3: RRF Fusion ───────────────────────────────────────────────────
    t0 = time.monotonic()
    fused = reciprocal_rank_fusion(ranked_lists)
    timings["rrf_fusion"] = int((time.monotonic() - t0) * 1000)

    # ── Stage 4: Lexical Prefilter ────────────────────────────────────────────
    t0 = time.monotonic()
    prefiltered = fast_lexical_prefilter(query, fused, keep=top_k_candidates)
    timings["lexical_prefilter"] = int((time.monotonic() - t0) * 1000)

    # ── Stage 5: Cross-Encoder Rerank ─────────────────────────────────────────
    t0 = time.monotonic()
    if enable_reranking and len(prefiltered) > 1:
        reranked = cross_encoder_rerank(
            query, prefiltered, ollama_host=ollama_host, llm_model=llm_model,
            top_k=top_k_final,
        )
        final_chunks = [
            {"text": r.text, "filename": r.filename, "chunk_idx": r.chunk_idx}
            for r in reranked
        ]
    else:
        final_chunks = prefiltered[:top_k_final]
    timings["cross_encoder_rerank"] = int((time.monotonic() - t0) * 1000)

    # ── Stage 6: Sufficiency Check (+ one corrective round) ──────────────────
    t0 = time.monotonic()
    rounds = 1
    if enable_sufficiency_check:
        # Sufficiency verdict and contradiction scan are independent given the
        # pre-correction chunk set — run them concurrently. If a corrective
        # round fires, contradictions are re-scanned on the merged set only
        # (cheap: contradiction detection is a single call regardless of round count).
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_verdict = pool.submit(check_sufficiency, query, final_chunks, ollama_host, llm_model)
            f_contra  = pool.submit(detect_contradictions, final_chunks, ollama_host, llm_model)
            verdict = f_verdict.result()
            contradictions = f_contra.result()

        result.sufficiency_confidence = verdict.confidence

        if not verdict.sufficient and verdict.reformed_query and max_corrective_rounds > 0:
            rounds += 1
            corrective_hits = doc_store.retrieve(verdict.reformed_query, top_k=top_k_candidates)
            if corrective_hits:
                combined = reciprocal_rank_fusion([final_chunks, corrective_hits])
                final_chunks = combined[:top_k_final]
                contradictions = detect_contradictions(final_chunks, ollama_host=ollama_host, llm_model=llm_model)

        result.contradictions = contradictions
    timings["sufficiency_check"] = int((time.monotonic() - t0) * 1000)
    result.retrieval_rounds = rounds

    # ── Stage 7: Citation DNA ─────────────────────────────────────────────────
    dna = make_citation_dna(final_chunks)
    result.citation_dna = dna
    result.context_block = format_context_with_citations(dna)
    result.sources = list({c.get("filename", "") for c in final_chunks if c.get("filename")})
    result.stage_timings_ms = timings

    return result
