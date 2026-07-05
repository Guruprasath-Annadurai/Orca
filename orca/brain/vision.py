"""
Vision / multimodal input — real gap this closes: Orca had zero image input
pipeline (Product Division audit flagged this as a hard zero).

HONEST SCOPE:
  - This is a v1, one-shot vision Q&A, NOT wired into the full AgentLoop
    (plan/tool-use/reflection). A vision message goes straight to
    OrcaBrain.complete() — no tool calls, no multi-step reasoning over an
    image. Wiring vision into the full agent loop is a real follow-up, not
    done here; this ships a working, honestly-scoped capability rather
    than a half-finished attempt at the larger one.
  - Requires a vision-CAPABLE model actually pulled in Ollama (llava,
    llama3.2-vision, qwen2-vl, moondream, bakllava, minicpm-v, etc.) — this
    project does not ship one. is_vision_capable() checks the model NAME
    against known patterns as a best-effort guard so a user gets a clear
    "pull a vision model" error instead of silently sending an image to a
    text-only model that will ignore it. This is a heuristic name check,
    not a query against Ollama's actual model capabilities (Ollama doesn't
    expose a clean "does this model support images" API as of this
    writing) — if a vision model doesn't match these patterns, this check
    will incorrectly reject it. Update the pattern list as new model
    families ship.
"""
from __future__ import annotations

import base64

VISION_MODEL_PATTERNS = [
    "llava", "vision", "bakllava", "moondream", "llama3.2-vision",
    "qwen2-vl", "qwen2.5-vl", "minicpm-v", "pixtral", "molmo",
]


def is_vision_capable(model_name: str) -> bool:
    lowered = model_name.lower()
    return any(pattern in lowered for pattern in VISION_MODEL_PATTERNS)


def encode_image(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def build_vision_message(text: str, image_b64: str) -> dict:
    """Ollama's /api/chat accepts a per-message 'images' field — a list of
    base64-encoded image strings. OrcaBrain._build_payload() (orca/brain/providers.py)
    passes messages through untouched, so this dict works with zero brain-layer changes."""
    return {"role": "user", "content": text, "images": [image_b64]}
