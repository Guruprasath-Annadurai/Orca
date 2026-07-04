"""
Self-Reflective Sufficiency Checker — the RAG loop that most companies skip.

After retrieval, most RAG systems just stuff chunks into context and generate.
This module asks the LLM to judge whether the retrieved chunks actually answer
the query BEFORE generation. If insufficient, it triggers one corrective
retrieval round with a reformed query — closing the loop like Self-RAG / CRAG.

Also tags each chunk with a "citation DNA" id so the final answer can cite
exact sources, and detects contradictions between chunks (useful when a doc
was updated and old + new chunks both got embedded).
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field


@dataclass
class SufficiencyVerdict:
    sufficient: bool
    confidence: float          # 0-1
    missing_info: str = ""     # what's missing, used to reform the query
    reformed_query: str = ""
    contradictions: list[str] = field(default_factory=list)


_SUFFICIENCY_PROMPT = """\
Judge whether the CONTEXT below contains enough information to fully answer the QUERY.

QUERY: {query}

CONTEXT:
{context}

Reply with ONLY a JSON object in this exact shape:
{{"sufficient": true/false, "confidence": 0.0-1.0, "missing_info": "what's missing if insufficient, else empty string"}}"""


_REFORM_PROMPT = """\
The original query could not be fully answered because: {missing_info}

Original query: {query}

Write ONE better search query that would retrieve the missing information.
Reply with ONLY the new query text, nothing else."""


_CONTRADICTION_PROMPT = """\
Compare these passages ONLY for direct factual contradictions — where two passages
assert opposite things about the SAME specific claim (e.g. one says a value is 10,
another says it's 20; one says a feature was removed, another says it still exists).

Do NOT flag as contradictions:
- Different facts about different topics (not a contradiction)
- Complementary or additional information
- Two true statements that simply aren't about the same claim

Example of a real contradiction: "The API rate limit is 100 req/min" vs "The API rate limit is 500 req/min"
Example of NOT a contradiction: "Lists are mutable" vs "Tuples are immutable" (different types, both true)

{passages}

Reply with ONLY a JSON array of contradiction strings (each describing one specific
conflicting claim), or [] if there are no direct contradictions."""


def _ollama_generate(prompt: str, host: str, model: str, max_tokens: int = 200) -> str:
    try:
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.0},
        }).encode()
        req = urllib.request.Request(
            f"{host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read())
        return data.get("response", "").strip()
    except Exception:
        return ""


def check_sufficiency(
    query: str,
    chunks: list[dict],
    ollama_host: str,
    llm_model: str,
) -> SufficiencyVerdict:
    """Judge if retrieved chunks are sufficient to answer the query."""
    if not chunks:
        return SufficiencyVerdict(
            sufficient=False, confidence=0.0,
            missing_info="No documents retrieved at all.",
        )

    context = "\n\n".join(
        f"[{c.get('filename','?')} #{c.get('chunk_idx',0)}] {c['text'][:500]}"
        for c in chunks
    )

    raw = _ollama_generate(
        _SUFFICIENCY_PROMPT.format(query=query, context=context),
        host=ollama_host, model=llm_model, max_tokens=150,
    )

    verdict = SufficiencyVerdict(sufficient=True, confidence=0.7)  # optimistic default
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            verdict.sufficient   = bool(data.get("sufficient", True))
            verdict.confidence   = float(data.get("confidence", 0.7))
            verdict.missing_info = str(data.get("missing_info", ""))
    except Exception:
        pass  # keep optimistic default — don't block the pipeline on a parse miss

    if not verdict.sufficient and verdict.missing_info:
        reformed = _ollama_generate(
            _REFORM_PROMPT.format(missing_info=verdict.missing_info, query=query),
            host=ollama_host, model=llm_model, max_tokens=60,
        )
        if reformed:
            verdict.reformed_query = reformed

    return verdict


def detect_contradictions(
    chunks: list[dict],
    ollama_host: str,
    llm_model: str,
) -> list[str]:
    """Check retrieved chunks for factual contradictions (e.g. stale doc versions)."""
    if len(chunks) < 2:
        return []

    passages = "\n\n".join(
        f"[{i+1}] {c['text'][:400]}" for i, c in enumerate(chunks)
    )
    raw = _ollama_generate(
        _CONTRADICTION_PROMPT.format(passages=passages),
        host=ollama_host, model=llm_model, max_tokens=200,
    )
    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            items = json.loads(match.group())
            return [str(i) for i in items if isinstance(i, str)]
    except Exception:
        pass
    return []


def make_citation_dna(chunks: list[dict]) -> dict[str, dict]:
    """
    Assign each chunk a short citable ID (e.g. 'D1', 'D2') and return a lookup
    the answer-generation step can reference. Encourages the model to cite
    sources inline as [D1], [D2] instead of vague "the document says".
    """
    dna: dict[str, dict] = {}
    for i, c in enumerate(chunks):
        cid = f"D{i+1}"
        dna[cid] = {
            "filename":   c.get("filename", "?"),
            "chunk_idx":  c.get("chunk_idx", 0),
            "text":       c["text"],
        }
    return dna


def format_context_with_citations(dna: dict[str, dict]) -> str:
    """Render the citation-tagged context block for injection into the system prompt."""
    lines = []
    for cid, info in dna.items():
        lines.append(f"[{cid}] (source: {info['filename']} §{info['chunk_idx']+1})\n{info['text']}")
    return "\n\n".join(lines)
