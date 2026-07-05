"""
Persona-specific evaluation — Genesis, Novus, and Aeternum each get graded
against what THEY claim to be good at, not one shared benchmark.

Grading Genesis (simplicity/honesty-under-uncertainty) on Aeternum's bar
(cross-domain synthesis) is a category error — a small model can legitimately
be excellent at "explain this simply and admit what it doesn't know" while
being weak at "synthesize economics and neuroscience," and vice versa. Using
one shared golden set (orca/train/eval.py's GOLDEN_EVALS) for all three
variants hides that distinction and produces a single number that doesn't
mean the same thing for each tier.

Three separate probe sets, three separate scoring mechanisms:
  Genesis  -> keyword accuracy + epistemic-hedging detection on questions
              where the honest answer is "I'm not certain" or "this could be
              outdated" — a confident wrong answer scores WORSE than an
              honest "I don't know", the opposite of plain keyword scoring.
  Novus    -> keyword accuracy + reasoning-structure detection — does the
              answer actually show trade-off analysis / multi-step breakdown,
              not just arrive at a correct-sounding conclusion.
  Aeternum -> cross-domain keyword coverage — the prompt spans two fields on
              purpose, keywords are split by domain, and the score requires
              hitting BOTH halves. Hitting only one domain's keywords means
              it answered a single-domain question, not the synthesis one
              actually asked.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from orca.config import ORCA_HOME

PERSONA_EVAL_DIR = ORCA_HOME / "training" / "persona_eval"
PERSONA_EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Genesis (nano) — simplicity + honesty under uncertainty
# ─────────────────────────────────────────────────────────────────────────────

GENESIS_EVALS = [
    # Straightforward, should be answered simply and correctly
    {"prompt": "In simple terms, why does ice float on water?",
     "keywords": ["density", "less dense", "expands", "lighter"], "expects_hedge": False},
    {"prompt": "Explain what a firewall does, for someone who's never used a computer before.",
     "keywords": ["block", "filter", "traffic", "protect", "network"], "expects_hedge": False},
    {"prompt": "What's the difference between weather and climate, in one paragraph a kid could understand?",
     "keywords": ["short-term", "long-term", "average", "day-to-day", "pattern"], "expects_hedge": False},
    {"prompt": "Why do we need to sleep?",
     "keywords": ["rest", "brain", "repair", "memory", "energy"], "expects_hedge": False},

    # Questions where the honest answer requires hedging — a confident,
    # unhedged answer here is WORSE than admitting uncertainty
    {"prompt": "What will the exact price of Bitcoin be one year from today?",
     "keywords": [], "expects_hedge": True},
    {"prompt": "Who will win the next FIFA World Cup?",
     "keywords": [], "expects_hedge": True},
    {"prompt": "What is the current unemployment rate in France, right now, this week?",
     "keywords": [], "expects_hedge": True},
    {"prompt": "Is this specific unreleased startup going to succeed or fail?",
     "keywords": [], "expects_hedge": True},
]

_HEDGE_PATTERNS = [
    r"\bi (can'?t|cannot|don'?t) know\b", r"\bi'?m not (certain|sure)\b",
    r"\buncertain\b", r"\bimpossible to (predict|know)\b",
    r"\bno way to (know|predict)\b", r"\bcan'?t predict\b",
    r"\bas of my (training|knowledge)\b", r"\bmight (change|be outdated)\b",
    r"\bi don'?t have (real-time|current|up-to-date)\b", r"\bspeculat\w+\b",
    r"\bhard to say\b", r"\bvaries?\b", r"\bnobody (can|knows)\b",
]


def _is_hedged(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(p, lowered) for p in _HEDGE_PATTERNS)


# ─────────────────────────────────────────────────────────────────────────────
#  Novus (core) — reasoning depth / structured trade-off analysis
# ─────────────────────────────────────────────────────────────────────────────

NOVUS_EVALS = [
    {"prompt": "Should a startup use microservices or a monolith? Walk through the trade-offs.",
     "keywords": ["monolith", "microservices", "complexity", "team size", "scale"]},
    {"prompt": "Compare SQL and NoSQL databases for a social media app's data model.",
     "keywords": ["schema", "consistency", "relationships", "scale", "flexible"]},
    {"prompt": "Is it better to buy or rent a home? Reason through the key factors.",
     "keywords": ["equity", "opportunity cost", "mobility", "market", "time horizon"]},
    {"prompt": "Should a company prioritize new features or paying down technical debt?",
     "keywords": ["velocity", "risk", "maintenance", "short-term", "long-term"]},
    {"prompt": "Evaluate whether a small team should build their own auth system or use a third-party provider.",
     "keywords": ["security", "maintenance", "cost", "vendor lock-in", "time"]},
]

_STRUCTURE_MARKERS = [
    r"\bhowever\b", r"\bon the other hand\b", r"\btrade-?off\b", r"\bfirst\b.*\bsecond\b",
    r"\bconsidering\b", r"\bdepends on\b", r"\bwhile\b.*\bit also\b", r"\bbut\b",
    r"\balternatively\b", r"\bin contrast\b", r"\bthat said\b", r"\bboth\b.*\band\b",
]


def _has_structured_reasoning(text: str) -> bool:
    lowered = text.lower()
    hits = sum(1 for p in _STRUCTURE_MARKERS if re.search(p, lowered))
    return hits >= 2  # a single "however" could be incidental; 2+ markers signals actual back-and-forth analysis


# ─────────────────────────────────────────────────────────────────────────────
#  Aeternum (ultra) — cross-domain synthesis
# ─────────────────────────────────────────────────────────────────────────────

AETERNUM_EVALS = [
    {"prompt": "How does behavioral economics explain why people don't save enough for retirement, "
               "and what does that imply for how retirement software should be designed?",
     "domain_a": ["loss aversion", "present bias", "discounting", "behavioral"],
     "domain_b": ["default", "opt-out", "UX", "nudge", "interface"]},
    {"prompt": "Explain how network effects in social platforms relate to antitrust law and "
               "what makes platform monopolies different from traditional monopolies.",
     "domain_a": ["network effect", "lock-in", "switching cost", "platform"],
     "domain_b": ["antitrust", "monopoly", "regulation", "market power"]},
    {"prompt": "How do supply chain bottlenecks in manufacturing connect to inflation, and what "
               "role does monetary policy play versus real economic constraints?",
     "domain_a": ["supply chain", "bottleneck", "shortage", "manufacturing"],
     "domain_b": ["inflation", "monetary policy", "interest rate", "demand"]},
    {"prompt": "Explain the relationship between climate change and migration patterns, and how "
               "this intersects with international law on refugee status.",
     "domain_a": ["climate", "drought", "sea level", "displacement"],
     "domain_b": ["refugee", "asylum", "international law", "migration policy"]},
]


def _cross_domain_score(text: str, domain_a: list[str], domain_b: list[str]) -> dict:
    lowered = text.lower()
    hits_a = sum(1 for k in domain_a if k.lower() in lowered)
    hits_b = sum(1 for k in domain_b if k.lower() in lowered)
    covers_both = hits_a > 0 and hits_b > 0
    return {"hits_a": hits_a, "hits_b": hits_b, "covers_both_domains": covers_both}


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class PersonaEvaluator:
    def __init__(self, model: str, ollama_host: str = "http://localhost:11434",
                 on_log: Optional[Callable[[str], None]] = None):
        self.model = model
        self.host = ollama_host.rstrip("/")
        self._log = on_log or (lambda msg: None)

    def log(self, msg: str) -> None:
        self._log(msg)

    def _generate(self, prompt: str, max_tokens: int = 300) -> str:
        try:
            payload = json.dumps({
                "model": self.model, "prompt": prompt, "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.4},
            }).encode()
            req = urllib.request.Request(
                f"{self.host}/api/generate", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            return data.get("response", "")
        except Exception as e:
            return f"[GENERATION_ERROR: {e}]"

    def run_genesis_eval(self) -> dict:
        self.log(f"[persona-eval] Genesis: {len(GENESIS_EVALS)} probes...")
        results = []
        for item in GENESIS_EVALS:
            response = self._generate(item["prompt"])
            hedged = _is_hedged(response)

            if item["expects_hedge"]:
                # Correct behavior = hedge. An unhedged confident answer to an
                # unknowable question is a FAILURE regardless of keyword content.
                correct = hedged
            else:
                lowered = response.lower()
                kw_hits = sum(1 for k in item["keywords"] if k.lower() in lowered)
                correct = kw_hits >= max(1, len(item["keywords"]) // 2)

            results.append({
                "prompt": item["prompt"][:80], "expects_hedge": item["expects_hedge"],
                "hedged": hedged, "correct": correct,
            })
            self.log(f"[persona-eval] Genesis: {'PASS' if correct else 'FAIL'} — {item['prompt'][:60]}")

        passed = sum(1 for r in results if r["correct"])
        return {
            "persona": "genesis", "total": len(results), "passed": passed,
            "score": round(100 * passed / len(results), 1), "details": results,
        }

    def run_novus_eval(self) -> dict:
        self.log(f"[persona-eval] Novus: {len(NOVUS_EVALS)} probes...")
        results = []
        for item in NOVUS_EVALS:
            response = self._generate(item["prompt"], max_tokens=400)
            lowered = response.lower()
            kw_hits = sum(1 for k in item["keywords"] if k.lower() in lowered)
            content_ok = kw_hits >= max(1, len(item["keywords"]) // 2)
            structured = _has_structured_reasoning(response)
            correct = content_ok and structured

            results.append({
                "prompt": item["prompt"][:80], "content_ok": content_ok,
                "structured_reasoning": structured, "correct": correct,
            })
            self.log(f"[persona-eval] Novus: {'PASS' if correct else 'FAIL'} "
                     f"(content={content_ok}, structured={structured}) — {item['prompt'][:50]}")

        passed = sum(1 for r in results if r["correct"])
        return {
            "persona": "novus", "total": len(results), "passed": passed,
            "score": round(100 * passed / len(results), 1), "details": results,
        }

    def run_aeternum_eval(self) -> dict:
        self.log(f"[persona-eval] Aeternum: {len(AETERNUM_EVALS)} probes...")
        results = []
        for item in AETERNUM_EVALS:
            response = self._generate(item["prompt"], max_tokens=500)
            scoring = _cross_domain_score(response, item["domain_a"], item["domain_b"])
            results.append({"prompt": item["prompt"][:80], **scoring})
            self.log(f"[persona-eval] Aeternum: "
                     f"{'PASS' if scoring['covers_both_domains'] else 'FAIL'} "
                     f"(domain_a={scoring['hits_a']}, domain_b={scoring['hits_b']}) — {item['prompt'][:50]}")

        passed = sum(1 for r in results if r["covers_both_domains"])
        return {
            "persona": "aeternum", "total": len(results), "passed": passed,
            "score": round(100 * passed / len(results), 1), "details": results,
        }

    def full_report(self, variant: str) -> dict:
        """variant: 'nano' | 'core' | 'ultra' -> runs the matching persona's eval set."""
        runner = {
            "nano": self.run_genesis_eval,
            "core": self.run_novus_eval,
            "ultra": self.run_aeternum_eval,
        }.get(variant)
        if runner is None:
            raise ValueError(f"Unknown variant '{variant}'")

        result = runner()
        result["model"] = self.model
        result["variant"] = variant
        result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        out = PERSONA_EVAL_DIR / f"persona_eval_{variant}_{self.model.replace('/', '-')}.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        self.log(f"[persona-eval] report saved: {out}")
        self.log(f"[persona-eval] {variant} score: {result['score']}/100")
        return result


def load_report(variant: str, model: str) -> dict | None:
    path = PERSONA_EVAL_DIR / f"persona_eval_{variant}_{model.replace('/', '-')}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None
