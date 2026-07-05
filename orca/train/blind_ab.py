"""
Blind A/B test harness — Orca vs a reference model (GPT-4o-mini, Claude
Haiku, or any other comparison target), same prompts, no branding shown to
the human rater.

HONEST SCOPE OF WHAT THIS MODULE DOES AND DOES NOT DO:
  - It generates paired responses (Orca + reference model) for a shared
    prompt set and writes them to a blind-labeled file (A/B, randomized per
    prompt, mapping kept separate so a rater can't infer which is which).
  - It does NOT run the human panel. A real blind A/B test requires actual
    human raters scoring actual output pairs — that's a human process, not
    something a script can substitute for. This module produces the INPUT
    to that process (the paired, randomized, blind-labeled responses) and a
    template for collecting the ratings back.
  - It does NOT call OpenAI/Anthropic APIs unless you provide real API keys
    via environment variables. Without them, reference_generate() raises
    clearly rather than silently returning fake data.

Usage:
    from orca.train.blind_ab import run_blind_ab, BlindABPrompt

    prompts = [BlindABPrompt(text="Explain quantum entanglement simply.")]
    run_blind_ab(prompts, orca_model="orca-ultra", reference_model="gpt-4o-mini")
    -> writes ~/.orca/training/blind_ab/blind_ab_<timestamp>.json
       (blind-labeled pairs for a human rater)
    -> writes ~/.orca/training/blind_ab/blind_ab_<timestamp>_KEY.json
       (the A/B -> real-model mapping, kept separate so raters can't peek)

After collecting ratings (a human process — see the printed instructions),
call score_blind_ab(ratings_file, key_file) to reveal which side won.
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from orca.config import ORCA_HOME

BLIND_AB_DIR = ORCA_HOME / "training" / "blind_ab"
BLIND_AB_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BlindABPrompt:
    text: str
    category: str = "general"


def orca_generate(prompt: str, model: str, ollama_host: str = "http://localhost:11434", max_tokens: int = 500) -> str:
    """Generate via local Ollama — no external API, no key needed."""
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.5},
    }).encode()
    req = urllib.request.Request(
        f"{ollama_host.rstrip('/')}/api/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read())
    return data.get("response", "")


def reference_generate(prompt: str, reference_model: str, max_tokens: int = 500) -> str:
    """
    Generate via an external reference model — GPT-4o-mini, Claude Haiku, etc.
    Requires a real API key. Raises clearly if not configured — this function
    will NOT fabricate a plausible-looking response to fill the gap; a fake
    comparison response would make the whole A/B test meaningless without
    telling you.
    """
    if reference_model.startswith("gpt-"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Blind A/B against a GPT reference model "
                "requires a real OpenAI API key — cannot proceed without one."
            )
        payload = json.dumps({
            "model": reference_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    if reference_model.startswith("claude-"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Blind A/B against a Claude reference model "
                "requires a real Anthropic API key — cannot proceed without one."
            )
        payload = json.dumps({
            "model": reference_model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data["content"][0]["text"]

    raise ValueError(f"Unrecognized reference model '{reference_model}' — expected a gpt-* or claude-* model id.")


def run_blind_ab(
    prompts: list[BlindABPrompt],
    orca_model: str,
    reference_model: str,
    ollama_host: str = "http://localhost:11434",
) -> dict:
    """
    Generates paired, blindly-labeled responses. Returns the file paths and
    a summary — does NOT score anything, since scoring requires a human rater.
    """
    pairs = []
    key = []

    for i, p in enumerate(prompts):
        orca_response = orca_generate(p.text, orca_model, ollama_host)
        reference_response = reference_generate(p.text, reference_model)

        # Randomize which side is "A" per prompt so a rater can't learn a
        # pattern (e.g. "A is always Orca") across the test set.
        if random.random() < 0.5:
            side_a, side_b = ("orca", orca_response), ("reference", reference_response)
        else:
            side_a, side_b = ("reference", reference_response), ("orca", orca_response)

        pairs.append({
            "id": i,
            "prompt": p.text,
            "category": p.category,
            "response_a": side_a[1],
            "response_b": side_b[1],
        })
        key.append({"id": i, "a_is": side_a[0], "b_is": side_b[0]})

    ts = int(time.time())
    ratings_template = BLIND_AB_DIR / f"blind_ab_{ts}.json"
    key_path = BLIND_AB_DIR / f"blind_ab_{ts}_KEY.json"

    with open(ratings_template, "w") as f:
        json.dump({
            "instructions": (
                "For each pair, read response_a and response_b WITHOUT knowing which model "
                "produced which. Add a 'winner' field to each pair: 'a', 'b', or 'tie'. "
                "Save this file, then call score_blind_ab() with both this file and the "
                "matching _KEY.json to reveal which model actually won."
            ),
            "orca_model": orca_model,
            "reference_model": reference_model,
            "pairs": pairs,
        }, f, indent=2)

    with open(key_path, "w") as f:
        json.dump({"orca_model": orca_model, "reference_model": reference_model, "key": key}, f, indent=2)

    return {
        "ratings_file": str(ratings_template),
        "key_file": str(key_path),
        "n_pairs": len(pairs),
        "next_step": f"Have a human rater fill in 'winner' for each pair in {ratings_template.name}, "
                     f"then call score_blind_ab() with both files.",
    }


def score_blind_ab(ratings_file: str, key_file: str) -> dict:
    """
    Reveals the actual win rate once a human has filled in 'winner' for each
    pair. Raises if any pair is missing a winner — a partially-rated test
    doesn't get a score, since that would misrepresent completion.
    """
    ratings = json.loads(Path(ratings_file).read_text())
    key_data = json.loads(Path(key_file).read_text())
    key_by_id = {k["id"]: k for k in key_data["key"]}

    orca_wins, reference_wins, ties, unrated = 0, 0, 0, 0

    for pair in ratings["pairs"]:
        winner = pair.get("winner")
        if winner not in ("a", "b", "tie"):
            unrated += 1
            continue
        if winner == "tie":
            ties += 1
            continue
        k = key_by_id[pair["id"]]
        winning_side = k["a_is"] if winner == "a" else k["b_is"]
        if winning_side == "orca":
            orca_wins += 1
        else:
            reference_wins += 1

    if unrated > 0:
        return {
            "status": "incomplete",
            "unrated_pairs": unrated,
            "total_pairs": len(ratings["pairs"]),
            "note": f"{unrated} pair(s) still need a 'winner' field before scoring.",
        }

    total = orca_wins + reference_wins + ties
    return {
        "status": "complete",
        "total_pairs": total,
        "orca_wins": orca_wins,
        "reference_wins": reference_wins,
        "ties": ties,
        "orca_win_rate": round(100 * orca_wins / total, 1) if total else 0.0,
        "orca_win_or_tie_rate": round(100 * (orca_wins + ties) / total, 1) if total else 0.0,
        "orca_model": ratings["orca_model"],
        "reference_model": ratings["reference_model"],
    }
