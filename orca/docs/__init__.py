"""Atheris Document Q&A — upload, chunk, embed, retrieve. Deep RAG pipeline included."""
from orca.docs.extractor import extract, SUPPORTED_EXTENSIONS, MAX_FILE_SIZE
from orca.docs.chunker import chunk_text, Chunk
from orca.docs.store import DocStore, register_doc, unregister_doc, list_docs
from orca.docs.pipeline import run_deep_rag, RAGResult

__all__ = [
    "extract", "SUPPORTED_EXTENSIONS", "MAX_FILE_SIZE",
    "chunk_text", "Chunk",
    "DocStore", "register_doc", "unregister_doc", "list_docs",
    "run_deep_rag", "RAGResult",
]
