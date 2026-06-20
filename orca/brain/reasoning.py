"""
Orca Reasoning Engine — local inference wrapper.
All reasoning runs on YOUR hardware through Ollama. No external API calls.
"""
from __future__ import annotations

from typing import Iterator

from orca.brain.providers import OrcaBrain, get_brain
from orca.config import CONFIG


class ReasoningEngine:
    """Wraps OrcaBrain with convenience methods used across variants."""

    def __init__(self, model: str | None = None):
        self.brain = get_brain(model)

    @property
    def model(self) -> str:
        return self.brain.name

    def think(
        self,
        messages: list[dict],
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return self.brain.complete(messages, system, temperature, max_tokens)

    def stream(
        self,
        messages: list[dict],
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        yield from self.brain.stream(messages, system, temperature, max_tokens)

    def count_tokens(self, messages: list[dict], system: str | None = None) -> int:
        """Estimate token count locally (no API call)."""
        text = system or ""
        for m in messages:
            text += m.get("content", "")
        # ~4 chars per token — good enough for context management
        return len(text) // 4
