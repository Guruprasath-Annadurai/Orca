"""
Text chunking for RAG — splits documents into overlapping windows.

Strategy:
  1. Sentence-aware splitting (respects paragraph boundaries)
  2. 512-char target chunk size with 64-char overlap
  3. Each chunk carries metadata: doc_id, filename, chunk_idx, char_start
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    doc_id: str
    filename: str
    chunk_idx: int
    char_start: int
    char_end: int

    def to_metadata(self) -> dict:
        return {
            "doc_id":    self.doc_id,
            "filename":  self.filename,
            "chunk_idx": self.chunk_idx,
            "char_start": self.char_start,
            "char_end":   self.char_end,
        }


def chunk_text(
    text: str,
    doc_id: str,
    filename: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """
    Split text into overlapping chunks, respecting sentence/paragraph boundaries.
    Returns list of Chunk objects ready for embedding.
    """
    if not text.strip():
        return []

    # Split into paragraphs first, then sentences
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # Further split long paragraphs at sentence boundaries
    sentences: list[tuple[str, int]] = []  # (sentence_text, char_pos)
    pos = 0
    for para in paragraphs:
        para_start = text.find(para, pos)
        if para_start == -1:
            para_start = pos
        for sent in _split_sentences(para):
            if sent.strip():
                sent_start = text.find(sent, para_start)
                sentences.append((sent, sent_start if sent_start != -1 else pos))
        pos = para_start + len(para)

    if not sentences:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    current_start = sentences[0][1]
    chunk_idx = 0

    for sent, sent_pos in sentences:
        sent_len = len(sent)

        # If adding this sentence exceeds chunk_size, flush
        if current and current_len + sent_len > chunk_size:
            chunk_text_str = " ".join(current).strip()
            if chunk_text_str:
                chunks.append(Chunk(
                    text=chunk_text_str,
                    doc_id=doc_id,
                    filename=filename,
                    chunk_idx=chunk_idx,
                    char_start=current_start,
                    char_end=current_start + len(chunk_text_str),
                ))
                chunk_idx += 1

            # Overlap: keep last overlap chars worth of sentences
            overlap_sents: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_len += len(s)

            current = overlap_sents + [sent]
            current_len = sum(len(s) for s in current)
            current_start = sent_pos
        else:
            current.append(sent)
            current_len += sent_len

    # Flush remaining
    if current:
        chunk_text_str = " ".join(current).strip()
        if chunk_text_str:
            chunks.append(Chunk(
                text=chunk_text_str,
                doc_id=doc_id,
                filename=filename,
                chunk_idx=chunk_idx,
                char_start=current_start,
                char_end=current_start + len(chunk_text_str),
            ))

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex (no NLTK dependency)."""
    # Split on sentence-ending punctuation followed by space + capital
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text)
    return [p.strip() for p in parts if p.strip()]
