"""
Orca Brain — 100% local, 100% yours.

No external APIs. No Anthropic. No OpenAI. No data leaving your machine.
All inference runs through Ollama on your hardware.

Model priority:
  1. Your fine-tuned 'orca' model (the goal)
  2. Best available open-weight model as interim brain
  3. Error — install Ollama

Set via environment:
  ORCA_CORE_MODEL=orca           ← your fine-tuned model name in Ollama
  ORCA_NANO_MODEL=orca-nano      ← lightweight variant
  ORCA_OLLAMA_HOST=localhost:11434
"""
from __future__ import annotations

import json
import os
from typing import Iterator

import httpx

from orca.config import CONFIG

PREFERRED_OPEN_MODELS = [
    "llama3.1:8b",
    "llama3.1:70b",
    "llama3:8b",
    "mistral:7b",
    "qwen2.5:7b",
    "gemma2:9b",
    "phi3:medium",
]


class OrcaBrain:
    """
    Orca's local brain — talks to Ollama, uses YOUR model when available.
    Zero network calls to any third party. Runs entirely on your machine.
    """

    def __init__(self, model: str | None = None, host: str | None = None):
        self.host = host or CONFIG.ollama.host
        self._requested_model = model
        self._resolved_model: str | None = None

    @property
    def model(self) -> str:
        if self._resolved_model is None:
            self._resolved_model = self._resolve_model()
        return self._resolved_model

    def _resolve_model(self) -> str:
        available = self._list_available()

        # Explicit model requested
        if self._requested_model:
            if self._requested_model in available:
                return self._requested_model
            raise RuntimeError(
                f"Model '{self._requested_model}' not found in Ollama.\n"
                f"Available: {', '.join(available)}\n"
                f"Pull it: ollama pull {self._requested_model}"
            )

        # Your fine-tuned Orca model takes priority
        orca_models = [m for m in available if m.startswith("orca")]
        if orca_models:
            return sorted(orca_models)[0]

        # Best available open-weight model
        for preferred in PREFERRED_OPEN_MODELS:
            if preferred in available:
                return preferred

        # Any model will do
        if available:
            return available[0]

        raise RuntimeError(
            "No models found in Ollama.\n"
            "Install a model first:\n"
            "  ollama pull llama3.1:8b\n"
            "Or fine-tune your own:\n"
            "  orca train run --preset prosumer"
        )

    def _list_available(self) -> list[str]:
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except httpx.ConnectError:
            raise RuntimeError(
                "Ollama is not running.\n"
                "Start it: ollama serve\n"
                "Install: curl -fsSL https://ollama.ai/install.sh | sh"
            )
        except Exception as e:
            raise RuntimeError(f"Cannot reach Ollama at {self.host}: {e}")

    def is_available(self) -> bool:
        try:
            httpx.get(f"{self.host}/api/tags", timeout=3).raise_for_status()
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            return self._list_available()
        except Exception:
            return []

    def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload = self._build_payload(messages, system, temperature, max_tokens, stream=False)
        try:
            r = httpx.post(f"{self.host}/api/chat", json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except httpx.ConnectError:
            raise RuntimeError("Ollama disconnected. Is 'ollama serve' still running?")

    def stream(
        self,
        messages: list[dict],
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        payload = self._build_payload(messages, system, temperature, max_tokens, stream=True)
        try:
            with httpx.stream(
                "POST", f"{self.host}/api/chat", json=payload, timeout=120
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if content := chunk.get("message", {}).get("content"):
                        yield content
                    if chunk.get("done"):
                        break
        except httpx.ConnectError:
            raise RuntimeError("Ollama disconnected mid-stream.")

    def _build_payload(
        self,
        messages: list[dict],
        system: str | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> dict:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        return {
            "model": self.model,
            "messages": all_messages,
            "stream": stream,
            "options": {
                "temperature": temperature or CONFIG.brain.temperature,
                "num_predict": max_tokens or CONFIG.brain.max_tokens,
                "top_p": CONFIG.brain.top_p,
                "num_ctx": CONFIG.brain.context_length,
            },
        }

    @property
    def name(self) -> str:
        try:
            return self.model
        except Exception:
            return "not connected"


# Single factory used everywhere
def get_brain(model: str | None = None) -> OrcaBrain:
    return OrcaBrain(model=model)
