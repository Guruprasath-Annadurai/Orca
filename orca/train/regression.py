"""
Regression testing — did the new model version get WORSE on prompts it used
to pass? Real gap this closes: eval_{model}.json was overwritten every run
(orca/train/eval.py), so there was never anything to compare against — no
way to know if a fine-tune improved, regressed, or just shuffled which
prompts pass without an overall score change hiding a real per-prompt
regression underneath.

Compares at PER-PROMPT granularity, not just the aggregate score — two
models can have the identical overall accuracy while one silently got worse
on "explain the GIL" and better on "binary search," which the aggregate
number alone would never reveal.
"""
from __future__ import annotations

import json
from pathlib import Path

from orca.train.eval import EVAL_DIR

HISTORY_DIR = EVAL_DIR / "history"


def list_eval_history(model: str) -> list[dict]:
    """All historical eval reports for a model, sorted oldest to newest."""
    if not HISTORY_DIR.exists():
        return []
    prefix = model.replace("/", "-")
    reports = []
    for path in HISTORY_DIR.glob(f"{prefix}_*.json"):
        try:
            reports.append(json.loads(path.read_text()))
        except Exception:
            continue
    reports.sort(key=lambda r: r.get("timestamp", ""))
    return reports


def compare_reports(baseline: dict, current: dict) -> dict:
    """
    Per-prompt diff between two eval reports for the SAME model. Returns
    regressions (prompts that scored lower), improvements (scored higher),
    and unchanged — plus headline score deltas.
    """
    baseline_by_prompt = {r["prompt"]: r for r in baseline.get("accuracy", {}).get("results", [])}
    current_by_prompt = {r["prompt"]: r for r in current.get("accuracy", {}).get("results", [])}

    regressions, improvements, unchanged, new_prompts = [], [], [], []

    for prompt, cur_result in current_by_prompt.items():
        base_result = baseline_by_prompt.get(prompt)
        if base_result is None:
            new_prompts.append({"prompt": prompt, "score": cur_result["keyword_score"]})
            continue

        delta = cur_result["keyword_score"] - base_result["keyword_score"]
        entry = {
            "prompt": prompt,
            "baseline_score": base_result["keyword_score"],
            "current_score": cur_result["keyword_score"],
            "delta": round(delta, 3),
        }
        if delta < -0.001:
            regressions.append(entry)
        elif delta > 0.001:
            improvements.append(entry)
        else:
            unchanged.append(entry)

    return {
        "baseline_timestamp": baseline.get("timestamp"),
        "current_timestamp": current.get("timestamp"),
        "overall_score_delta": round(current.get("overall_score", 0) - baseline.get("overall_score", 0), 1),
        "accuracy_delta": round(current.get("accuracy", {}).get("accuracy", 0) - baseline.get("accuracy", {}).get("accuracy", 0), 3),
        "style_delta": round(current.get("style", {}).get("style_score", 0) - baseline.get("style", {}).get("style_score", 0), 2),
        "regressions": regressions,
        "improvements": improvements,
        "unchanged_count": len(unchanged),
        "new_prompts": new_prompts,
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
    }


def check_regression(model: str, max_allowed_regressions: int = 0) -> tuple[bool, dict]:
    """
    Compares the two most recent eval reports for a model. Returns
    (passed, report). passed=False if regression_count exceeds
    max_allowed_regressions (default 0 — any per-prompt regression fails).

    If there's no prior history to compare against (first eval run ever for
    this model), passes automatically — there's nothing to regress FROM.
    """
    history = list_eval_history(model)

    if len(history) < 2:
        return True, {
            "status": "no_baseline",
            "note": f"Only {len(history)} eval report(s) on record for '{model}' — "
                    f"need at least 2 to compare. This run establishes/extends the baseline.",
        }

    baseline, current = history[-2], history[-1]
    comparison = compare_reports(baseline, current)
    comparison["status"] = "compared"
    passed = comparison["regression_count"] <= max_allowed_regressions
    return passed, comparison
