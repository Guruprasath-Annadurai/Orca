"""
Atheris Model Evaluator — two modes:

  OllamaEvaluator  — talks to a live Ollama model (no GPU deps, runs in CI)
  ModelEvaluator   — loads a merged HF/Unsloth checkpoint for offline eval

Both produce the same report schema:
  {model, speed, accuracy, style, overall_score}

CI usage:
  orca train eval --ollama orca-core --ci
  → writes ~/.orca/training/eval/eval_report.json
  → exit code 0 if overall_score >= threshold, else 1
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Callable, Optional

from orca.config import ORCA_HOME

EVAL_DIR = ORCA_HOME / "training" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Golden eval set — 50 prompts spanning all seed domains
# ─────────────────────────────────────────────────────────────────────────────

GOLDEN_EVALS = [
    # Python
    {"prompt": "What is the difference between a list and a tuple in Python?",
     "keywords": ["mutable", "immutable", "tuple", "list", "hashable"]},
    {"prompt": "Explain Python's GIL and when it matters.",
     "keywords": ["Global Interpreter Lock", "thread", "CPU-bound", "I/O", "multiprocessing"]},
    {"prompt": "Write a Python decorator that measures function execution time.",
     "keywords": ["def", "wrapper", "time", "functools", "wraps"]},
    {"prompt": "What are Python generators and when should you use them?",
     "keywords": ["yield", "lazy", "memory", "iterator", "generator"]},
    {"prompt": "How do dataclasses differ from regular classes in Python?",
     "keywords": ["dataclass", "__init__", "field", "decorator", "boilerplate"]},

    # Algorithms
    {"prompt": "What is the time complexity of binary search and why?",
     "keywords": ["O(log n)", "sorted", "mid", "divide", "half"]},
    {"prompt": "Explain dynamic programming with a concrete example.",
     "keywords": ["memoization", "subproblem", "overlapping", "optimal", "Fibonacci"]},
    {"prompt": "When would you use a hash map over a sorted array?",
     "keywords": ["O(1)", "lookup", "collision", "sorted", "binary search"]},
    {"prompt": "What is the sliding window technique?",
     "keywords": ["window", "pointer", "subarray", "O(n)", "contiguous"]},
    {"prompt": "Explain Dijkstra's algorithm in plain English.",
     "keywords": ["shortest path", "priority queue", "weighted", "greedy", "visited"]},

    # Debugging
    {"prompt": "What causes a Python 'maximum recursion depth exceeded' error?",
     "keywords": ["recursion", "base case", "stack", "sys.setrecursionlimit", "overflow"]},
    {"prompt": "How do you debug a race condition in async Python code?",
     "keywords": ["asyncio", "Lock", "race", "concurrent", "await"]},
    {"prompt": "Why is comparing floats with == a bug?",
     "keywords": ["floating point", "precision", "epsilon", "math.isclose", "IEEE"]},
    {"prompt": "What is the N+1 query problem in SQL and how do you fix it?",
     "keywords": ["query", "loop", "JOIN", "eager loading", "ORM"]},
    {"prompt": "How do you find a memory leak in a Python application?",
     "keywords": ["tracemalloc", "gc", "reference", "del", "memory"]},

    # Systems design
    {"prompt": "How would you design a rate limiter for an API?",
     "keywords": ["token bucket", "sliding window", "Redis", "limit", "request"]},
    {"prompt": "What is the CAP theorem and why does it matter?",
     "keywords": ["Consistency", "Availability", "Partition", "tradeoff", "distributed"]},
    {"prompt": "Explain the difference between vertical and horizontal scaling.",
     "keywords": ["vertical", "horizontal", "sharding", "load balancer", "replica"]},
    {"prompt": "When should you use a message queue instead of a direct API call?",
     "keywords": ["async", "decouple", "Kafka", "retry", "throughput"]},
    {"prompt": "What is a circuit breaker pattern?",
     "keywords": ["fail fast", "timeout", "fallback", "open", "half-open"]},

    # SQL / Databases
    {"prompt": "What is the difference between INNER JOIN and LEFT JOIN?",
     "keywords": ["matching", "NULL", "left", "all rows", "filter"]},
    {"prompt": "When would you use a database index and when not?",
     "keywords": ["B-tree", "write", "read", "selectivity", "overhead"]},
    {"prompt": "What are SQL window functions? Give an example.",
     "keywords": ["OVER", "PARTITION BY", "ROW_NUMBER", "rank", "window"]},
    {"prompt": "Explain database ACID properties.",
     "keywords": ["Atomicity", "Consistency", "Isolation", "Durability", "transaction"]},
    {"prompt": "What is cursor-based pagination and why is it better than OFFSET?",
     "keywords": ["cursor", "OFFSET", "performance", "stable", "large"]},

    # Bash / Linux
    {"prompt": "How do you find all files larger than 100MB in Linux?",
     "keywords": ["find", "-size", "+100M", "du", "ls"]},
    {"prompt": "What does 'set -euo pipefail' do in a bash script?",
     "keywords": ["error", "exit", "unset", "pipe", "strict"]},
    {"prompt": "How do you use awk to sum a column in a CSV?",
     "keywords": ["awk", "sum", "field", "NR", "print"]},
    {"prompt": "What is the difference between > and >> in bash?",
     "keywords": ["overwrite", "append", "redirect", "stdout", "file"]},
    {"prompt": "How do you monitor a log file in real time?",
     "keywords": ["tail", "-f", "follow", "grep", "stdout"]},

    # API design
    {"prompt": "What is the difference between REST and GraphQL?",
     "keywords": ["endpoint", "query", "overfetching", "schema", "flexible"]},
    {"prompt": "When should an API return 400 vs 422?",
     "keywords": ["Bad Request", "Unprocessable", "validation", "syntax", "semantic"]},
    {"prompt": "How do you handle API versioning?",
     "keywords": ["URL", "header", "v1", "backward", "deprecat"]},
    {"prompt": "What makes an API endpoint idempotent?",
     "keywords": ["same result", "PUT", "DELETE", "retry", "safe"]},
    {"prompt": "How do you design webhook delivery with guarantees?",
     "keywords": ["retry", "idempotent", "HMAC", "delivery", "queue"]},

    # Security
    {"prompt": "How do you prevent SQL injection?",
     "keywords": ["parameterized", "prepared statement", "ORM", "escape", "query"]},
    {"prompt": "What is XSS and how do you prevent it?",
     "keywords": ["Cross-Site Scripting", "escape", "CSP", "sanitize", "output"]},
    {"prompt": "Why is MD5 a bad choice for password hashing?",
     "keywords": ["fast", "brute force", "bcrypt", "salt", "GPU"]},
    {"prompt": "What is a timing attack and how do you prevent it?",
     "keywords": ["constant time", "hmac.compare_digest", "timing", "side channel", "compare"]},
    {"prompt": "What is CSRF and how does a CSRF token prevent it?",
     "keywords": ["Cross-Site Request Forgery", "token", "origin", "SameSite", "cookie"]},

    # Docker / K8s
    {"prompt": "What is the difference between CMD and ENTRYPOINT in Docker?",
     "keywords": ["override", "exec", "shell", "default", "append"]},
    {"prompt": "When would you use a Kubernetes StatefulSet vs Deployment?",
     "keywords": ["stateful", "persistent", "ordinal", "database", "identity"]},
    {"prompt": "What are Kubernetes liveness vs readiness probes?",
     "keywords": ["restart", "traffic", "healthy", "ready", "probe"]},
    {"prompt": "How does Kubernetes handle rolling updates?",
     "keywords": ["maxSurge", "maxUnavailable", "rolling", "zero downtime", "rollback"]},
    {"prompt": "What is a multi-stage Docker build and why use it?",
     "keywords": ["builder", "final", "size", "cache", "layer"]},

    # Orca voice / general reasoning
    {"prompt": "What is the difference between TCP and UDP?",
     "keywords": ["connection", "reliable", "stateless", "packet", "handshake"]},
    {"prompt": "Explain gradient descent in one paragraph.",
     "keywords": ["loss", "gradient", "minimize", "learning rate", "optimization"]},
    {"prompt": "What are the SOLID principles?",
     "keywords": ["Single", "Open", "Liskov", "Interface", "Dependency"]},
    {"prompt": "Write a Python function to flatten a nested list.",
     "keywords": ["def", "list", "append", "isinstance", "recursive"]},
    {"prompt": "What is eventual consistency?",
     "keywords": ["eventual", "replica", "converge", "distributed", "sync"]},
]

STYLE_JUDGE_PROMPT = """\
Score this AI response on "Orca style" — direct, precise, zero sycophancy.
Return ONLY JSON: {"score": int, "issues": [str]}

Scoring (1-10):
- Start at 10
- -2 if starts with "Great question", "Certainly!", "Of course"
- -2 if ends with "I hope this helps" or "Let me know if..."
- -1 for each hedge like "I think", "I believe", "As an AI"
- -1 for padding or unnecessary repetition
- -1 if the answer is vague where it should be specific
- +0 for dry humor if it works

Score 10 = perfect Orca: gets to the point, precise, no filler.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Ollama-native evaluator — no GPU / no ML deps required
# ─────────────────────────────────────────────────────────────────────────────

class OllamaEvaluator:
    """
    Evaluates a model served by Ollama — works in CI without GPU.

    Usage:
        ev = OllamaEvaluator("orca-core")
        report = ev.full_report()
    """

    def __init__(
        self,
        model: str,
        ollama_host: str = "http://localhost:11434",
        on_log: Callable[[str], None] | None = None,
        judge_model: str | None = None,
    ):
        self.model = model
        self.host = ollama_host.rstrip("/")
        self.log = on_log or print
        self.judge_model = judge_model or model  # use same model as judge by default

    def _generate(self, prompt: str, system: str = "", max_tokens: int = 512) -> str:
        """Single Ollama generation via the /api/generate endpoint."""
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.7},
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())["response"]
        except Exception as e:
            return f"[error: {e}]"

    def benchmark_speed(self, prompt: str = "Explain neural networks in one paragraph.") -> dict:
        """Measure tokens/second. Uses Ollama's built-in token stats."""
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 200, "temperature": 0.7},
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            return {"tokens_per_sec": 0, "error": str(e)}
        elapsed = time.time() - t0

        # Ollama reports eval_count and eval_duration (nanoseconds)
        eval_count = data.get("eval_count", 0)
        eval_ns = data.get("eval_duration", 0)
        tps = eval_count / (eval_ns / 1e9) if eval_ns > 0 else eval_count / max(elapsed, 0.001)
        self.log(f"[eval] speed: {tps:.1f} tok/s ({eval_count} tokens in {elapsed:.1f}s)")
        return {"tokens_per_sec": round(tps, 1), "total_tokens": eval_count, "elapsed_sec": round(elapsed, 2)}

    def accuracy_eval(self, n: int | None = None) -> dict:
        """Run golden eval set — keyword coverage score."""
        evals = GOLDEN_EVALS[:n] if n else GOLDEN_EVALS
        results = []
        for i, item in enumerate(evals):
            output = self._generate(item["prompt"], max_tokens=400)
            output_lower = output.lower()
            hits = sum(1 for kw in item["keywords"] if kw.lower() in output_lower)
            score = hits / len(item["keywords"])
            results.append({
                "prompt": item["prompt"][:70],
                "keyword_score": round(score, 2),
                "hits": hits,
                "total": len(item["keywords"]),
            })
            self.log(f"[eval] [{i+1:02d}/{len(evals)}] {item['prompt'][:55]:55s} → {score*100:.0f}%")

        avg = sum(r["keyword_score"] for r in results) / len(results)
        self.log(f"[eval] accuracy: {avg*100:.1f}% across {len(evals)} prompts")
        return {"accuracy": round(avg, 3), "results": results, "n_prompts": len(evals)}

    def style_eval(self, n: int = 10) -> dict:
        """Use the model to judge its own style (self-judge) or a separate judge model."""
        sample = GOLDEN_EVALS[:n]
        scores = []

        for item in sample:
            response = self._generate(item["prompt"], max_tokens=300)
            judge_input = f'Prompt: "{item["prompt"]}"\n\nResponse: "{response}"'

            # Use judge model to score
            payload = json.dumps({
                "model": self.judge_model,
                "prompt": judge_input,
                "system": STYLE_JUDGE_PROMPT,
                "stream": False,
                "options": {"num_predict": 100, "temperature": 0.1},
            }).encode()
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = json.loads(resp.read())["response"]
                s, e = raw.find("{"), raw.rfind("}") + 1
                data = json.loads(raw[s:e])
                scores.append(int(data.get("score", 5)))
            except Exception:
                scores.append(5)

        avg = sum(scores) / len(scores) if scores else 0
        self.log(f"[eval] style: {avg:.1f}/10 (n={len(scores)})")
        return {"style_score": round(avg, 2), "n_samples": len(scores), "scores": scores}

    def full_report(self, n_accuracy: int | None = None, n_style: int = 10) -> dict:
        self.log(f"[eval] evaluating model: {self.model}")
        self.log(f"[eval] host: {self.host}")
        self.log("")

        speed    = self.benchmark_speed()
        accuracy = self.accuracy_eval(n=n_accuracy)
        style    = self.style_eval(n=n_style)

        overall = round(
            accuracy["accuracy"] * 60 + (style["style_score"] / 10) * 40,
            1,
        )

        report = {
            "model":         self.model,
            "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "speed":         speed,
            "accuracy":      accuracy,
            "style":         style,
            "overall_score": overall,
        }

        out = EVAL_DIR / f"eval_{self.model.replace('/', '-')}.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2)

        # Regression testing needs history, not just "latest" — the file
        # above gets overwritten every run, so orca/train/regression.py
        # can't diff anything from it alone. Archive a timestamped copy
        # alongside it; existing readers of eval_{model}.json (model_cards.py)
        # are untouched since that file's path/format doesn't change.
        history_dir = EVAL_DIR / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        ts_safe = report["timestamp"].replace(":", "-")
        history_path = history_dir / f"{self.model.replace('/', '-')}_{ts_safe}.json"
        with open(history_path, "w") as f:
            json.dump(report, f, indent=2)

        self.log(f"\n[eval] report saved: {out}")
        self.log(f"[eval] history archived: {history_path}")
        self.log(f"[eval] overall score: {overall}/100")
        return report

    @staticmethod
    def compare(model_a: str, model_b: str, host: str = "http://localhost:11434", n: int = 20) -> dict:
        """Side-by-side comparison of two Ollama models on the same prompts."""
        results = []
        evals = GOLDEN_EVALS[:n]

        for item in evals:
            def _gen(model):
                payload = json.dumps({
                    "model": model,
                    "prompt": item["prompt"],
                    "stream": False,
                    "options": {"num_predict": 400, "temperature": 0.7},
                }).encode()
                req = urllib.request.Request(
                    f"{host.rstrip('/')}/api/generate",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        return json.loads(resp.read())["response"]
                except Exception:
                    return ""

            out_a = _gen(model_a)
            out_b = _gen(model_b)
            kws = item["keywords"]

            score_a = sum(1 for kw in kws if kw.lower() in out_a.lower()) / len(kws)
            score_b = sum(1 for kw in kws if kw.lower() in out_b.lower()) / len(kws)

            results.append({
                "prompt": item["prompt"][:70],
                model_a: round(score_a, 2),
                model_b: round(score_b, 2),
                "winner": model_a if score_a > score_b else (model_b if score_b > score_a else "tie"),
            })

        avg_a = sum(r[model_a] for r in results) / len(results)
        avg_b = sum(r[model_b] for r in results) / len(results)
        wins_a = sum(1 for r in results if r["winner"] == model_a)
        wins_b = sum(1 for r in results if r["winner"] == model_b)
        ties   = sum(1 for r in results if r["winner"] == "tie")

        return {
            "model_a": model_a,
            "model_b": model_b,
            "n_prompts": n,
            "avg_a": round(avg_a, 3),
            "avg_b": round(avg_b, 3),
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "winner": model_a if avg_a > avg_b else (model_b if avg_b > avg_a else "tie"),
            "results": results,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  HF/Unsloth checkpoint evaluator (GPU required)
# ─────────────────────────────────────────────────────────────────────────────

class ModelEvaluator:
    """Evaluates a merged HF/Unsloth checkpoint loaded from disk."""

    def __init__(self, model_path: str, on_log: Callable[[str], None] | None = None):
        self.model_path = model_path
        self.log = on_log or print

    def benchmark_speed(self, prompt: str = "Explain neural networks briefly.") -> dict:
        self._ensure_model()
        start = time.time()
        tokens = 0
        for _ in self._generate(prompt, max_new_tokens=200):
            tokens += 1
        elapsed = time.time() - start
        tps = tokens / elapsed
        self.log(f"[eval] speed: {tps:.1f} tok/s")
        return {"tokens_per_sec": round(tps, 1), "total_tokens": tokens, "elapsed_sec": round(elapsed, 2)}

    def accuracy_eval(self, n: int | None = None) -> dict:
        self._ensure_model()
        evals = GOLDEN_EVALS[:n] if n else GOLDEN_EVALS
        results = []
        for i, item in enumerate(evals):
            output = "".join(self._generate(item["prompt"], max_new_tokens=400))
            output_lower = output.lower()
            hits = sum(1 for kw in item["keywords"] if kw.lower() in output_lower)
            score = hits / len(item["keywords"])
            results.append({"prompt": item["prompt"][:70], "keyword_score": round(score, 2),
                            "hits": hits, "total": len(item["keywords"])})
            self.log(f"[eval] [{i+1:02d}/{len(evals)}] {item['prompt'][:55]:55s} → {score*100:.0f}%")
        avg = sum(r["keyword_score"] for r in results) / len(results)
        return {"accuracy": round(avg, 3), "results": results, "n_prompts": len(evals)}

    def style_eval(self, n: int = 10, brain=None) -> dict:
        self._ensure_model()
        if brain is None:
            try:
                from orca.brain import OrcaBrain
                brain = OrcaBrain()
                if not brain.is_available():
                    return {"style_score": 0, "n_samples": 0, "skipped": True}
            except Exception:
                return {"style_score": 0, "n_samples": 0, "skipped": True}

        scores = []
        for item in GOLDEN_EVALS[:n]:
            output = "".join(self._generate(item["prompt"], max_new_tokens=300))
            try:
                resp = brain.complete(
                    [{"role": "user", "content": f'Prompt: "{item["prompt"]}"\n\nResponse: "{output}"'}],
                    system=STYLE_JUDGE_PROMPT,
                    temperature=0.1,
                )
                s, e = resp.find("{"), resp.rfind("}") + 1
                data = json.loads(resp[s:e])
                scores.append(int(data.get("score", 5)))
            except Exception:
                scores.append(5)
        avg = sum(scores) / len(scores) if scores else 0
        self.log(f"[eval] style: {avg:.1f}/10")
        return {"style_score": round(avg, 2), "n_samples": len(scores)}

    def full_report(self) -> dict:
        self.log(f"[eval] evaluating checkpoint: {self.model_path}")
        speed    = self.benchmark_speed()
        accuracy = self.accuracy_eval()
        style    = self.style_eval(n=5)

        overall = round(accuracy["accuracy"] * 60 + (style["style_score"] / 10) * 40, 1)
        report = {
            "model":         self.model_path,
            "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "speed":         speed,
            "accuracy":      accuracy,
            "style":         style,
            "overall_score": overall,
        }
        out = EVAL_DIR / "eval_report.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        self.log(f"[eval] report saved: {out}")
        self.log(f"[eval] overall score: {overall}/100")
        return report

    def _ensure_model(self):
        if hasattr(self, "_model"):
            return
        self.log(f"[eval] loading: {self.model_path}")
        try:
            from unsloth import FastLanguageModel
            self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path, max_seq_length=2048, dtype=None, load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self._model)
        except ImportError:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16, device_map="auto",
            )

    def _generate(self, prompt: str, max_new_tokens: int = 256):
        import torch
        inputs = self._tokenizer(
            f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n",
            return_tensors="pt",
        ).to(self._model.device)
        with torch.no_grad():
            output = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=0.7, top_p=0.9, do_sample=True,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        yield self._tokenizer.decode(generated, skip_special_tokens=True)
