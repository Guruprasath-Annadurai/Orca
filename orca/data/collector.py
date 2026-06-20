"""
Orca Data Collector — logs every conversation turn in training-ready format.

Every time you use Orca, this silently builds your fine-tuning dataset.
Format: ShareGPT (conversations list) — compatible with Unsloth, Axolotl, LLaMA-Factory.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

from orca.config import ORCA_HOME

RAW_DATA_DIR = ORCA_HOME / "training" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

ORCA_SYSTEM_PROMPT = """\
You are Orca — a powerful, thoughtful AI assistant. You reason carefully before responding,
give direct and accurate answers, and adapt your style to the complexity of each question.
For simple questions, be concise. For complex ones, think step by step.
You have persistent memory, can execute code, search the web, and manage files.
"""


@dataclass
class Turn:
    role: Literal["system", "human", "gpt"]
    value: str


@dataclass
class Conversation:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = "orca"
    variant: str = "core"
    conversations: list[Turn] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def add_system(self, text: str = ORCA_SYSTEM_PROMPT) -> None:
        self.conversations.insert(0, Turn(role="system", value=text))

    def add_human(self, text: str) -> None:
        self.conversations.append(Turn(role="human", value=text))

    def add_gpt(self, text: str) -> None:
        self.conversations.append(Turn(role="gpt", value=text))

    def is_valid(self) -> bool:
        roles = [t.role for t in self.conversations]
        return (
            len(self.conversations) >= 2
            and any(r == "human" for r in roles)
            and any(r == "gpt" for r in roles)
            and all(len(t.value.strip()) > 3 for t in self.conversations if t.role != "system")
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "variant": self.variant,
            "conversations": [asdict(t) for t in self.conversations],
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


class DataCollector:
    """
    Wraps any Orca session and silently saves conversations to JSONL.
    Each line = one training example in ShareGPT format.
    """

    def __init__(self, variant: str = "core", auto_system: bool = True):
        self.variant = variant
        self.auto_system = auto_system
        self._current: Conversation | None = None
        self._output = RAW_DATA_DIR / f"{variant}_{time.strftime('%Y%m%d')}.jsonl"

    def start_conversation(self, metadata: dict | None = None) -> Conversation:
        self._current = Conversation(variant=self.variant, metadata=metadata or {})
        if self.auto_system:
            self._current.add_system()
        return self._current

    def log_turn(self, role: Literal["human", "gpt"], text: str) -> None:
        if self._current is None:
            self.start_conversation()
        if role == "human":
            self._current.add_human(text)  # type: ignore
        else:
            self._current.add_gpt(text)  # type: ignore

    def save(self) -> bool:
        if self._current is None or not self._current.is_valid():
            return False
        with open(self._output, "a") as f:
            f.write(json.dumps(self._current.to_dict()) + "\n")
        self._current = None
        return True

    def discard(self) -> None:
        self._current = None

    @staticmethod
    def count_examples() -> dict[str, int]:
        counts = {}
        for f in RAW_DATA_DIR.glob("*.jsonl"):
            counts[f.stem] = sum(1 for _ in open(f))
        return counts
