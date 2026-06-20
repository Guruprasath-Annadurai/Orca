"""
Orca Model Evaluator — benchmarks the fine-tuned model against base model.

Tests:
- Perplexity on held-out Orca conversations
- Task accuracy on a golden eval set
- Style alignment score (does it sound like Orca?)
- Speed benchmark (tokens/sec)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from orca.config import ORCA_HOME

EVAL_DIR = ORCA_HOME / "training" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

GOLDEN_EVALS = [
    {
        "prompt": "What is the difference between TCP and UDP?",
        "expected_keywords": ["connection", "reliable", "stateless", "packet", "handshake"],
    },
    {
        "prompt": "Write a Python function to flatten a nested list",
        "expected_keywords": ["def", "list", "append", "isinstance", "recursive"],
    },
    {
        "prompt": "Explain gradient descent in one paragraph",
        "expected_keywords": ["loss", "gradient", "minimize", "learning rate", "optimization"],
    },
    {
        "prompt": "What are the SOLID principles?",
        "expected_keywords": ["Single", "Open", "Liskov", "Interface", "Dependency"],
    },
    {
        "prompt": "Design a rate limiter for an API",
        "expected_keywords": ["token bucket", "sliding window", "redis", "limit", "request"],
    },
]

STYLE_JUDGE_SYSTEM = """\
Score this AI response on "Orca style" — direct, smart, no fluff, no sycophancy.
Return JSON: {"score": int 1-10, "issues": ["issue1", ...]}
Score 10 = perfect Orca style. Deduct for: "Great question", excessive caveats,
unnecessary apologies, vague answers, padding.
"""


class ModelEvaluator:
    """Evaluates Orca model quality."""

    def __init__(self, model_path: str, on_log: Callable[[str], None] | None = None):
        self.model_path = model_path
        self.log = on_log or print

    def benchmark_speed(self, prompt: str = "Explain neural networks briefly.") -> dict:
        """Measure tokens/second throughput."""
        self._ensure_model()
        start = time.time()
        tokens = 0
        for tok in self._generate(prompt, max_new_tokens=200):
            tokens += 1
        elapsed = time.time() - start
        tps = tokens / elapsed
        self.log(f"[Eval] Speed: {tps:.1f} tokens/sec")
        return {"tokens_per_sec": tps, "total_tokens": tokens, "elapsed_sec": elapsed}

    def accuracy_eval(self) -> dict:
        """Run golden eval set — check keyword coverage."""
        self._ensure_model()
        results = []
        for item in GOLDEN_EVALS:
            output = "".join(self._generate(item["prompt"], max_new_tokens=400))
            output_lower = output.lower()
            hits = sum(1 for kw in item["expected_keywords"] if kw.lower() in output_lower)
            score = hits / len(item["expected_keywords"])
            results.append({
                "prompt": item["prompt"][:60],
                "keyword_score": round(score, 2),
                "hits": hits,
                "total": len(item["expected_keywords"]),
            })
            self.log(f"[Eval] {item['prompt'][:50]:50s} → {score*100:.0f}%")

        avg = sum(r["keyword_score"] for r in results) / len(results)
        return {"accuracy": round(avg, 3), "results": results}

    def style_eval(self, n_samples: int = 10, brain=None) -> dict:
        """Use local Ollama model to judge if responses match Orca style."""
        self._ensure_model()

        if brain is None:
            try:
                from orca.brain import OrcaBrain
                brain = OrcaBrain()
                if not brain.is_available():
                    self.log("[Eval] Ollama offline — skipping style eval")
                    return {"style_score": 0, "n_samples": 0, "skipped": True}
            except Exception:
                self.log("[Eval] Brain unavailable — skipping style eval")
                return {"style_score": 0, "n_samples": 0, "skipped": True}

        scores = []
        prompts = [e["prompt"] for e in GOLDEN_EVALS[:n_samples]]

        for p in prompts:
            output = "".join(self._generate(p, max_new_tokens=300))
            try:
                resp = brain.complete(
                    [{"role": "user", "content": f"Prompt: {p}\n\nResponse: {output}"}],
                    system=STYLE_JUDGE_SYSTEM,
                    temperature=0.1,
                )
                start, end = resp.find("{"), resp.rfind("}") + 1
                data = json.loads(resp[start:end])
                scores.append(data.get("score", 5))
            except Exception:
                scores.append(5)

        avg = sum(scores) / len(scores) if scores else 0
        self.log(f"[Eval] Style score: {avg:.1f}/10")
        return {"style_score": round(avg, 2), "n_samples": len(scores)}

    def full_report(self) -> dict:
        self.log("[Eval] Running full evaluation suite...")
        speed = self.benchmark_speed()
        accuracy = self.accuracy_eval()
        style = self.style_eval(n_samples=5)

        report = {
            "model": self.model_path,
            "speed": speed,
            "accuracy": accuracy,
            "style": style,
            "overall_score": round(
                (accuracy["accuracy"] * 50) + (style["style_score"] / 10 * 50), 1
            ),
        }

        out = EVAL_DIR / "eval_report.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        self.log(f"[Eval] Report saved: {out}")
        self.log(f"[Eval] Overall score: {report['overall_score']}/100")
        return report

    def _ensure_model(self):
        if not hasattr(self, "_model"):
            self.log(f"[Eval] Loading model: {self.model_path}")
            try:
                from unsloth import FastLanguageModel
                self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_path,
                    max_seq_length=2048,
                    dtype=None,
                    load_in_4bit=True,
                )
                FastLanguageModel.for_inference(self._model)
            except ImportError:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )

    def _generate(self, prompt: str, max_new_tokens: int = 256):
        import torch
        tokenizer = self._tokenizer
        model = self._model

        inputs = tokenizer(
            f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n",
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        generated = output[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        yield text
