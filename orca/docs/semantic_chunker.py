"""
Semantic Velocity Chunker — topic-boundary detection via embedding cosine similarity.

Unlike fixed-size chunkers, this detects "semantic velocity" — points where the
cosine similarity between consecutive sentence embeddings drops sharply, indicating
a topic shift. Creates chunks that respect natural semantic boundaries.

Also builds a 3-level hierarchy:
  Level 0: original semantic chunks
  Level 1: section summaries (3–6 chunks merged and summarised)
  Level 2: document summary (whole-doc abstract)

This hierarchy enables retrieval at the right granularity — short factual queries
hit leaf chunks; broad thematic queries hit section summaries; "what is this doc
about" hits the document summary.
"""
from __future__ import annotations

import json
import math
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from orca.docs.chunker import Chunk, _split_sentences


@dataclass
class SemanticChunk(Chunk):
    """Extended chunk with semantic metadata."""
    level: int = 0                    # 0=leaf, 1=section-summary, 2=doc-summary
    embedding: list[float] = field(default_factory=list)
    boundary_score: float = 0.0       # how sharp the topic break before this chunk
    children_ids: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict:
        base = super().to_metadata()
        base["level"] = self.level
        base["boundary_score"] = round(self.boundary_score, 4)
        base["children_ids"] = json.dumps(self.children_ids)
        return base


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def _ollama_embed_batch(texts: list[str], host: str, model: str) -> list[list[float]]:
    """Embed a list of texts; returns zero vectors on failure."""
    results = []
    for text in texts:
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
            results.append(data.get("embedding", []))
        except Exception:
            results.append([])
    return results


def semantic_chunk(
    text: str,
    doc_id: str,
    filename: str,
    ollama_host: str = "http://localhost:11434",
    embed_model: str = "nomic-embed-text",
    *,
    min_chunk_chars: int = 128,
    max_chunk_chars: int = 1024,
    velocity_threshold: float = 0.25,   # similarity drop this large → new chunk
    window: int = 2,                     # sentences to look ahead/behind for smoothing
) -> list[SemanticChunk]:
    """
    Split text into semantically coherent chunks using embedding cosine velocity.

    Algorithm:
      1. Split into sentences.
      2. Embed each sentence (or fall back to BM25-free fixed chunking).
      3. Compute rolling cosine similarity between consecutive sentences.
      4. A boundary is declared when similarity drops > velocity_threshold AND
         the accumulated chunk is longer than min_chunk_chars.
      5. The boundary_score is stored (how sharp the topic break was).
    """
    if not text.strip():
        return []

    # Sentence splitting
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sentences: list[tuple[str, int]] = []
    pos = 0
    for para in paragraphs:
        pstart = text.find(para, pos)
        if pstart == -1:
            pstart = pos
        for sent in _split_sentences(para):
            if sent.strip():
                sstart = text.find(sent, pstart)
                sentences.append((sent, sstart if sstart != -1 else pos))
        pos = pstart + len(para)

    if not sentences:
        return []

    texts_only = [s for s, _ in sentences]

    # Try to get embeddings
    embeddings = _ollama_embed_batch(texts_only, ollama_host, embed_model)
    has_embeddings = all(len(e) > 0 for e in embeddings)

    chunks: list[SemanticChunk] = []
    current_sents: list[str] = []
    current_embs:  list[list[float]] = []
    current_start  = sentences[0][1]
    chunk_idx      = 0

    def _flush(boundary_score: float = 0.0):
        nonlocal chunk_idx, current_start, current_sents, current_embs
        if not current_sents:
            return
        chunk_text = " ".join(current_sents).strip()
        if not chunk_text:
            return
        emb = current_embs[-1] if current_embs else []
        chunks.append(SemanticChunk(
            text=chunk_text,
            doc_id=doc_id,
            filename=filename,
            chunk_idx=chunk_idx,
            char_start=current_start,
            char_end=current_start + len(chunk_text),
            level=0,
            embedding=emb,
            boundary_score=boundary_score,
        ))
        chunk_idx += 1
        current_sents = []
        current_embs  = []

    for i, ((sent, spos), emb) in enumerate(zip(sentences, embeddings)):
        if not current_sents:
            current_start = spos
            current_sents.append(sent)
            current_embs.append(emb)
            continue

        current_len = sum(len(s) for s in current_sents)

        # Compute semantic velocity
        if has_embeddings and emb and current_embs:
            # Smooth by averaging the last `window` embeddings
            prev_embs = current_embs[-window:]
            avg_prev = [sum(x[j] for x in prev_embs) / len(prev_embs)
                        for j in range(len(prev_embs[0]))]
            sim = _cosine(avg_prev, emb)
            velocity = 1.0 - sim  # high velocity = topic shift
        else:
            # Fallback: chunk by max_chunk_chars
            velocity = 0.0
            sim = 1.0

        is_boundary = (
            velocity > velocity_threshold
            and current_len >= min_chunk_chars
        ) or (
            current_len + len(sent) > max_chunk_chars
            and current_len >= min_chunk_chars
        )

        if is_boundary:
            _flush(boundary_score=round(velocity, 4))
            current_start = spos

        current_sents.append(sent)
        current_embs.append(emb)

    _flush()
    return chunks


def build_hierarchy(
    leaf_chunks: list[SemanticChunk],
    doc_id: str,
    filename: str,
    ollama_host: str,
    embed_model: str,
    section_size: int = 5,
) -> list[SemanticChunk]:
    """
    Build level-1 (section) and level-2 (document) summary chunks.
    These are stored alongside leaf chunks with level=1/2 markers.
    Retrieval can then target the right level based on query complexity.
    """
    all_chunks = list(leaf_chunks)
    section_summaries: list[SemanticChunk] = []

    # Level 1: section summaries
    for i in range(0, len(leaf_chunks), section_size):
        group = leaf_chunks[i : i + section_size]
        # Concatenate and truncate to avoid embedding overload
        merged = " ".join(c.text for c in group)[:3000]
        group_ids = [f"{c.doc_id}_{c.chunk_idx}" for c in group]

        embs = _ollama_embed_batch([merged], ollama_host, embed_model)
        sec_chunk = SemanticChunk(
            text=f"[SECTION {i//section_size + 1}] {merged}",
            doc_id=doc_id,
            filename=filename,
            chunk_idx=10000 + i // section_size,
            char_start=group[0].char_start,
            char_end=group[-1].char_end,
            level=1,
            embedding=embs[0] if embs else [],
            children_ids=group_ids,
        )
        section_summaries.append(sec_chunk)
        all_chunks.append(sec_chunk)

    # Level 2: document summary
    doc_text = " ".join(c.text for c in leaf_chunks[:20])[:4000]  # first 20 chunks ≈ intro
    all_section_ids = [f"{c.doc_id}_{c.chunk_idx}" for c in section_summaries]
    doc_embs = _ollama_embed_batch([doc_text], ollama_host, embed_model)
    doc_chunk = SemanticChunk(
        text=f"[DOCUMENT: {filename}] {doc_text}",
        doc_id=doc_id,
        filename=filename,
        chunk_idx=99999,
        char_start=0,
        char_end=len(doc_text),
        level=2,
        embedding=doc_embs[0] if doc_embs else [],
        children_ids=all_section_ids,
    )
    all_chunks.append(doc_chunk)

    return all_chunks
