"""
Query Intelligence Engine — transforms a raw user query into an optimized
multi-signal retrieval strategy.

Stages:
  1. Complexity classifier   — simple / compound / multi-hop
  2. Query rewriter          — resolve anaphors using conversation history
  3. Query decomposer        — break multi-hop queries into atomic sub-questions
  4. HyDE generator          — generate a hypothetical answer, embed it
  5. Multi-query expander    — generate 3-5 query variants for recall diversity
"""
from __future__ import annotations

import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QueryPlan:
    original: str
    rewritten: str = ""
    complexity: str = "simple"            # simple | compound | multi_hop
    sub_queries: list[str] = field(default_factory=list)
    expanded_queries: list[str] = field(default_factory=list)
    hyde_text: str = ""                   # hypothetical answer for HyDE embedding
    hyde_embedding: list[float] = field(default_factory=list)


def _ollama_generate(
    prompt: str,
    host: str,
    model: str,
    max_tokens: int = 256,
    temperature: float = 0.3,
) -> str:
    """Call Ollama /api/generate, return the response text."""
    try:
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }).encode()
        req = urllib.request.Request(
            f"{host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data.get("response", "").strip()
    except Exception:
        return ""


def _ollama_embed(text: str, host: str, model: str) -> list[float]:
    try:
        payload = json.dumps({"model": model, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{host}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        return data.get("embedding", [])
    except Exception:
        return []


_COMPLEXITY_PROMPT = """\
Classify this query into one of: simple, compound, multi_hop.

simple    = single fact or concept question
compound  = 2-3 related aspects in one question
multi_hop = requires connecting information from multiple sources or reasoning chains

Query: {query}

Reply with ONLY the single word: simple, compound, or multi_hop."""


_DECOMPOSE_PROMPT = """\
Break this complex query into 2-4 simpler atomic sub-questions.
Each sub-question should be independently answerable.

Query: {query}

Reply with ONLY a JSON array of strings. Example:
["sub-question 1", "sub-question 2", "sub-question 3"]"""


_EXPAND_PROMPT = """\
Generate 3 alternative phrasings of this query to improve document retrieval.
The variants should use different vocabulary and angles but ask the same thing.

Query: {query}

Reply with ONLY a JSON array of 3 strings."""


_REWRITE_PROMPT = """\
Rewrite this query to be self-contained, resolving any pronouns or references
using the conversation context.

Recent conversation:
{history}

Current query: {query}

Reply with ONLY the rewritten query (no explanation)."""


_HYDE_PROMPT = """\
Write a short, factual paragraph (2-4 sentences) that would perfectly answer this query.
Be specific. Write as if you are the authoritative source.

Query: {query}

Answer:"""


def build_query_plan(
    query: str,
    conversation_history: list[str] | None,
    ollama_host: str,
    llm_model: str,
    embed_model: str,
    *,
    enable_hyde: bool = True,
    enable_decompose: bool = True,
    enable_expand: bool = True,
    fast_mode: bool = True,
) -> QueryPlan:
    """
    Build a full QueryPlan for a user query.
    Falls back gracefully if Ollama is unavailable.

    fast_mode (default True): for `simple` queries, skip decompose/expand/HyDE
    entirely — those add 3 serial LLM calls (~20-40s on CPU-bound Ollama) for
    zero benefit on a one-fact lookup. Compound/multi-hop queries always get
    the full treatment regardless of fast_mode, since that's where they earn
    their cost.
    """
    plan = QueryPlan(original=query, rewritten=query)

    # ── 1. Rewrite with conversation context ─────────────────────────────────
    if conversation_history:
        history_str = "\n".join(conversation_history[-6:])  # last 3 turns
        rw = _ollama_generate(
            _REWRITE_PROMPT.format(history=history_str, query=query),
            host=ollama_host,
            model=llm_model,
            max_tokens=128,
        )
        plan.rewritten = rw if (rw and len(rw) < 500) else query
    else:
        plan.rewritten = query

    effective_query = plan.rewritten

    # ── 2. Classify complexity ────────────────────────────────────────────────
    complexity_raw = _ollama_generate(
        _COMPLEXITY_PROMPT.format(query=effective_query),
        host=ollama_host,
        model=llm_model,
        max_tokens=10,
        temperature=0.0,
    ).lower().strip()
    complexity_raw = complexity_raw.split()[0] if complexity_raw else "simple"

    if complexity_raw in ("simple", "compound", "multi_hop", "multi-hop"):
        plan.complexity = complexity_raw.replace("-", "_")
    else:
        plan.complexity = "simple"

    skip_deep_stages = fast_mode and plan.complexity == "simple"

    # ── 3/4/5. Decompose + Expand + HyDE — run concurrently, they're independent ──
    def _do_decompose() -> list[str]:
        if not (enable_decompose and plan.complexity in ("compound", "multi_hop")):
            return []
        raw = _ollama_generate(
            _DECOMPOSE_PROMPT.format(query=effective_query),
            host=ollama_host, model=llm_model, max_tokens=200, temperature=0.2,
        )
        try:
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                sub_qs = json.loads(match.group())
                return [q for q in sub_qs if isinstance(q, str) and q.strip()]
        except Exception:
            pass
        return []

    def _do_expand() -> list[str]:
        if not enable_expand or skip_deep_stages:
            return []
        raw = _ollama_generate(
            _EXPAND_PROMPT.format(query=effective_query),
            host=ollama_host, model=llm_model, max_tokens=200, temperature=0.5,
        )
        try:
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                variants = json.loads(match.group())
                return [q for q in variants if isinstance(q, str) and q.strip()]
        except Exception:
            pass
        return []

    def _do_hyde() -> tuple[str, list[float]]:
        if not enable_hyde or skip_deep_stages:
            return "", []
        text = _ollama_generate(
            _HYDE_PROMPT.format(query=effective_query),
            host=ollama_host, model=llm_model, max_tokens=200, temperature=0.4,
        )
        if not text:
            return "", []
        return text, _ollama_embed(text, host=ollama_host, model=embed_model)

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_decompose = pool.submit(_do_decompose)
        f_expand    = pool.submit(_do_expand)
        f_hyde      = pool.submit(_do_hyde)

        plan.sub_queries = f_decompose.result()
        expanded         = f_expand.result()
        plan.hyde_text, plan.hyde_embedding = f_hyde.result()

    # Fallback: always have at least the rewritten query
    plan.expanded_queries = [effective_query] + expanded if expanded else [effective_query]

    return plan
