"""
Distillation pipeline — generate synthetic training data from a stronger
teacher model, in the exact ShareGPT format the existing fine-tuning
pipeline (orca/data/collector.py) already expects.

HONEST SCOPE:
  This module builds the DATA GENERATION pipeline. It does not, by itself,
  make the model smarter — that still requires:
    1. A real teacher model actually capable of better reasoning than what
       you're distilling into (a bigger local Ollama model, e.g. a 70B+
       model pulled and running — this needs real GPU/VRAM, the H100
       discussed earlier, not something this sandbox has access to).
    2. Running this generation loop for enough prompts (hundreds to
       thousands) to meaningfully shift training data composition — that's
       real wall-clock GPU time, not a quick script run.
    3. Actually re-running `orca train ultra` (or core/nano) on the
       resulting dataset — fine-tuning itself, separate from this step.
  This module is step 1 of that chain, built and tested against a small
  local model as a stand-in. Steps 2-3 need your hardware and your time.

Usage:
    from orca.train.distill import distill_from_seeds

    distill_from_seeds(
        teacher_model="llama3.1:70b",   # swap for your actual strong teacher
        n_examples=500,
        variant="ultra",
    )
    -> appends to ~/.orca/training/raw/{variant}_distilled_<date>.jsonl
       in the same ShareGPT format orca/data/collector.py already produces,
       so `orca data seed` / `orca train prepare` pick it up automatically.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

from orca.config import ORCA_HOME
from orca.data.collector import RAW_DATA_DIR, Conversation, ORCA_SYSTEM_PROMPT
from orca.data.seeds import ALL_DOMAINS, sample_domains, build_prompt

DISTILL_LOG_DIR = ORCA_HOME / "training" / "distill_logs"
DISTILL_LOG_DIR.mkdir(parents=True, exist_ok=True)

# A reasoning-trace instruction appended to the teacher prompt — asks the
# teacher to show its work, not just the final answer. Distilling reasoning
# traces (not just answers) is what actually improves the student's own
# reasoning quality, not just its factual recall.
REASONING_TRACE_SUFFIX = (
    "\n\nThink through this step by step before giving your final answer. "
    "Show your reasoning, then clearly state the conclusion."
)


def _teacher_generate(prompt: str, teacher_model: str, ollama_host: str, max_tokens: int = 700) -> str:
    payload = json.dumps({
        "model": teacher_model,
        "prompt": prompt + REASONING_TRACE_SUFFIX,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.3},  # low temp — favor consistency for a teacher
    }).encode()
    req = urllib.request.Request(
        f"{ollama_host.rstrip('/')}/api/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:  # generous timeout — teacher models are typically larger/slower
        data = json.loads(resp.read())
    return data.get("response", "")


def distill_from_seeds(
    teacher_model: str,
    n_examples: int,
    variant: str = "ultra",
    ollama_host: str = "http://localhost:11434",
    domains: list | None = None,
    on_log=None,
) -> dict:
    """
    Generates n_examples (prompt, teacher_response) pairs using orca's own
    seed domain prompts (orca/data/seeds.py — the same source `orca data
    seed` already draws from), but with the TEACHER model's response instead
    of the current generation pipeline's. Appends to the raw training data
    in ShareGPT format.

    Returns a summary dict — counts, output file, failures.
    """
    log = on_log or (lambda msg: None)
    domain_names = [d.name for d in domains] if domains else None

    output_path = RAW_DATA_DIR / f"{variant}_distilled_{time.strftime('%Y%m%d')}.jsonl"
    log_path = DISTILL_LOG_DIR / f"distill_{teacher_model.replace('/', '-')}_{int(time.time())}.jsonl"

    written = 0
    failed = 0

    log(f"[distill] teacher: {teacher_model}  target: {n_examples} examples  -> {output_path}")

    # sample_domains(n, names) returns [(Domain, count), ...] weighted across
    # domains — flatten into one (domain, index) job per requested example.
    allocation = sample_domains(n_examples, domain_names)
    jobs = [domain for domain, count in allocation for _ in range(count)]

    with open(output_path, "a") as out_f, open(log_path, "w") as log_f:
        for i, domain in enumerate(jobs):
            # build_prompt returns (system, user) — the user half is the
            # actual instruction; the domain's system half describes the
            # domain framing, distinct from ORCA_SYSTEM_PROMPT used below.
            _domain_system, prompt_text = build_prompt(domain)

            try:
                response = _teacher_generate(prompt_text, teacher_model, ollama_host)
            except Exception as e:
                failed += 1
                log(f"[distill] [{i+1}/{len(jobs)}] FAILED: {e}")
                continue

            if not response.strip() or len(response.strip()) < 20:
                failed += 1
                log(f"[distill] [{i+1}/{len(jobs)}] empty/too-short response, skipped")
                continue

            convo = Conversation(source=f"distill:{teacher_model}", variant=variant)
            convo.add_system(ORCA_SYSTEM_PROMPT)
            convo.add_human(prompt_text)
            convo.add_gpt(response)

            if convo.is_valid():
                out_f.write(json.dumps(convo.to_dict()) + "\n")
                written += 1
                log(f"[distill] [{i+1}/{len(jobs)}] written — domain={domain.name}")
            else:
                failed += 1
                log(f"[distill] [{i+1}/{len(jobs)}] failed validity check, skipped")

            log_f.write(json.dumps({
                "i": i, "domain": domain.name, "prompt": prompt_text[:200],
                "response_preview": response[:300],
            }) + "\n")

    result = {
        "teacher_model": teacher_model,
        "variant": variant,
        "requested": n_examples,
        "written": written,
        "failed": failed,
        "output_file": str(output_path),
        "log_file": str(log_path),
    }
    log(f"[distill] done — {written} written, {failed} failed")
    return result
