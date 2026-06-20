"""
Orca Nano — lightweight, fast, terminal-native. 100% local.
Uses your fine-tuned orca-nano model, or best available small model via Ollama.
"""
from __future__ import annotations

from typing import Iterator

from orca.brain.providers import get_brain
from orca.config import CONFIG

NANO_SYSTEM = (
    "You are Orca Nano — a fast, precise terminal assistant. "
    "Give direct, concise answers. No filler. No sycophancy. "
    "Prefer short responses. When analyzing piped data, output structured results."
)


class OrcaNano:
    """Fast variant — runs locally on Ollama."""

    def __init__(self, model: str | None = None):
        self.brain = get_brain(model or CONFIG.ollama.model_nano)
        self._history: list[dict] = []

    def run(self, prompt: str, piped_input: str | None = None) -> str:
        content = f"{piped_input}\n\n{prompt}" if piped_input else prompt
        return self.brain.complete(
            [{"role": "user", "content": content}],
            system=NANO_SYSTEM,
        )

    def stream(self, prompt: str, piped_input: str | None = None) -> Iterator[str]:
        content = f"{piped_input}\n\n{prompt}" if piped_input else prompt
        yield from self.brain.stream(
            [{"role": "user", "content": content}],
            system=NANO_SYSTEM,
        )

    def chat(self, message: str) -> Iterator[str]:
        self._history.append({"role": "user", "content": message})
        full = ""
        for chunk in self.brain.stream(self._history, system=NANO_SYSTEM):
            full += chunk
            yield chunk
        self._history.append({"role": "assistant", "content": full})

    def reset(self) -> None:
        self._history.clear()
