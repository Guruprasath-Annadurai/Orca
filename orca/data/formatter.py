"""
Orca Data Formatter — converts curated ShareGPT data into model-specific training formats.

Supports:
- Llama 3 (default for Orca model)
- ChatML (Mistral, Qwen, etc.)
- Alpaca (instruction-following)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from orca.config import ORCA_HOME

CURATED_DIR = ORCA_HOME / "training" / "curated"
FORMATTED_DIR = ORCA_HOME / "training" / "formatted"
FORMATTED_DIR.mkdir(parents=True, exist_ok=True)

Format = Literal["llama3", "chatml", "alpaca"]

# Llama 3 special tokens
LLAMA3 = {
    "bos": "<|begin_of_text|>",
    "sys_start": "<|start_header_id|>system<|end_header_id|>\n\n",
    "sys_end": "<|eot_id|>",
    "user_start": "<|start_header_id|>user<|end_header_id|>\n\n",
    "user_end": "<|eot_id|>",
    "asst_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "asst_end": "<|eot_id|>",
    "eos": "<|end_of_text|>",
}

# ChatML tokens (Mistral, Qwen, Phi)
CHATML = {
    "sys_start": "<|im_start|>system\n",
    "sys_end": "<|im_end|>\n",
    "user_start": "<|im_start|>user\n",
    "user_end": "<|im_end|>\n",
    "asst_start": "<|im_start|>assistant\n",
    "asst_end": "<|im_end|>",
}


def to_llama3(conv: dict) -> str:
    text = LLAMA3["bos"]
    for turn in conv["conversations"]:
        role = turn["role"]
        val = turn["value"]
        if role == "system":
            text += LLAMA3["sys_start"] + val + LLAMA3["sys_end"]
        elif role == "human":
            text += LLAMA3["user_start"] + val + LLAMA3["user_end"]
        elif role == "gpt":
            text += LLAMA3["asst_start"] + val + LLAMA3["asst_end"]
    text += LLAMA3["eos"]
    return text


def to_chatml(conv: dict) -> str:
    text = ""
    for turn in conv["conversations"]:
        role = turn["role"]
        val = turn["value"]
        if role == "system":
            text += CHATML["sys_start"] + val + CHATML["sys_end"]
        elif role == "human":
            text += CHATML["user_start"] + val + CHATML["user_end"]
        elif role == "gpt":
            text += CHATML["asst_start"] + val + CHATML["asst_end"]
    return text


def to_alpaca(conv: dict) -> dict | None:
    """Extract first human/gpt pair as instruction/output."""
    turns = conv["conversations"]
    human = next((t["value"] for t in turns if t["role"] == "human"), None)
    gpt = next((t["value"] for t in turns if t["role"] == "gpt"), None)
    system = next((t["value"] for t in turns if t["role"] == "system"), "")
    if not human or not gpt:
        return None
    return {"instruction": human, "input": "", "output": gpt, "system": system}


class DataFormatter:
    """Convert curated ShareGPT JSONL into final training format."""

    def __init__(self, fmt: Format = "llama3"):
        self.fmt = fmt

    def format(self, input_file: Path | None = None) -> dict:
        src = input_file or CURATED_DIR / "dataset.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"Curated dataset not found: {src}\nRun: orca data curate")

        out_path = FORMATTED_DIR / f"orca_{self.fmt}.jsonl"
        converted = skipped = 0

        with open(src) as f_in, open(out_path, "w") as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    conv = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                if self.fmt == "llama3":
                    record = {"text": to_llama3(conv), "id": conv.get("id", "")}
                elif self.fmt == "chatml":
                    record = {"text": to_chatml(conv), "id": conv.get("id", "")}
                elif self.fmt == "alpaca":
                    record = to_alpaca(conv)
                    if record is None:
                        skipped += 1
                        continue
                else:
                    skipped += 1
                    continue

                f_out.write(json.dumps(record) + "\n")
                converted += 1

        return {
            "format": self.fmt,
            "converted": converted,
            "skipped": skipped,
            "output": str(out_path),
        }

    def split(self, path: Path, train_ratio: float = 0.95) -> dict:
        """Split into train/eval sets."""
        lines = open(path).readlines()
        split_idx = int(len(lines) * train_ratio)
        train_path = path.parent / f"{path.stem}_train.jsonl"
        eval_path = path.parent / f"{path.stem}_eval.jsonl"
        open(train_path, "w").writelines(lines[:split_idx])
        open(eval_path, "w").writelines(lines[split_idx:])
        return {
            "train": str(train_path),
            "eval": str(eval_path),
            "train_examples": split_idx,
            "eval_examples": len(lines) - split_idx,
        }
