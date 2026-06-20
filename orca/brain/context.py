"""
Orca Context Manager — token budgeting and compaction.
Runs locally, no external calls.
"""
from __future__ import annotations

COMPACT_THRESHOLD = 0.75


class ContextManager:
    """Manages context window and triggers compaction when needed."""

    CONTEXT_SIZE = 8192

    def __init__(self, brain):
        self.brain = brain

    def needs_compaction(self, messages: list[dict]) -> bool:
        tokens = self._estimate_tokens(messages)
        return tokens / self.CONTEXT_SIZE > COMPACT_THRESHOLD

    def compact(self, messages: list[dict]) -> list[dict]:
        history = "\n\n".join(
            f"[{m['role'].upper()}]: {m['content']}" for m in messages
        )
        summary = self.brain.complete(
            [{"role": "user", "content": f"Summarize this conversation preserving all key facts, decisions, and code:\n\n{history}"}],
            system="You are a context compactor. Output only the dense summary, nothing else.",
            max_tokens=1024,
        )
        return [
            {"role": "user", "content": f"[CONVERSATION SUMMARY]\n{summary}"},
            {"role": "assistant", "content": "Context loaded."},
        ]

    def _estimate_tokens(self, messages: list[dict]) -> int:
        return sum(len(m.get("content", "")) for m in messages) // 4
