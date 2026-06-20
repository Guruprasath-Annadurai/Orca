"""
Orca Data Curator — cleans, scores, deduplicates, and filters training data.

Quality gates (all local, no external APIs):
- Deduplication via SHA-256 hash
- Minimum length checks
- Sycophancy pattern stripping
- Optional local AI judge (uses OrcaBrain / Ollama — free, private)
- Optional parallel processing
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
from pathlib import Path
from typing import Iterator

from orca.config import ORCA_HOME

RAW_DATA_DIR = ORCA_HOME / "training" / "raw"
CURATED_DATA_DIR = ORCA_HOME / "training" / "curated"
CURATED_DATA_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_JUDGE_SYSTEM = """\
You are a training data quality judge. Score this AI conversation.
Output ONLY JSON:
{"accuracy": int, "helpfulness": int, "clarity": int, "style": int, "overall": int, "keep": bool}

Score 1-10. keep=true if overall >= 7.
Reject if: sycophantic opener, hallucination, excessive hedging, "I hope this helps", padding.
"""

# Patterns that make training data toxic
_BAD_PATTERNS = [
    r"^(Great|Excellent|Good|Wonderful|Sure|Absolutely|Of course|Certainly)[!,]?\s*",
    r"I hope (this|that) helps?[!.]?\s*",
    r"Feel free to ask (if|any).*",
    r"Is there anything else.*\?",
    r"Let me know if you (need|have|want).*",
    r"As an AI (language model|assistant).*",
    r"I('m| am) just an AI.*",
    r"I don't have (the ability|access) to.*",
]

# Hard reject patterns — these examples should never reach training
_REJECT_PATTERNS = [
    r"Great question!",
    r"Certainly!",
    r"Of course!",
    r"I'd be happy to",
    r"I'm happy to help",
]


def _hash_conversation(conv: dict) -> str:
    text = " ".join(t.get("value", t.get("content", "")) for t in conv["conversations"])
    return hashlib.sha256(text.encode()).hexdigest()


def _load_jsonl(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _strip_sycophancy(text: str) -> str:
    for p in _BAD_PATTERNS:
        text = re.sub(p, "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()


def _has_hard_reject(text: str) -> bool:
    for p in _REJECT_PATTERNS:
        if re.search(p, text, flags=re.IGNORECASE):
            return True
    return False


def _min_length_ok(conv: dict) -> bool:
    for turn in conv["conversations"]:
        role = turn.get("role", "")
        val = turn.get("value", turn.get("content", ""))
        if role == "gpt" and len(val.split()) < 10:
            return False
        if role == "human" and len(val.strip()) < 3:
            return False
    return True


def local_score(conv: dict, brain) -> dict | None:
    """Use local Ollama model as quality judge — free, private, no API key."""
    try:
        sample = json.dumps(conv["conversations"], indent=2)[:2000]
        resp = brain.complete(
            [{"role": "user", "content": f"Score this conversation:\n{sample}"}],
            system=LOCAL_JUDGE_SYSTEM,
            temperature=0.1,
        )
        s, e = resp.find("{"), resp.rfind("}") + 1
        return json.loads(resp[s:e])
    except Exception:
        return None


class DataCurator:
    """
    Reads raw JSONL files, applies quality gates, outputs curated dataset.
    All processing is local — no external APIs.
    """

    def __init__(
        self,
        use_local_judge: bool = False,
        min_score: int = 7,
        workers: int = 1,
        brain=None,
    ):
        self.use_local_judge = use_local_judge
        self.min_score = min_score
        self.workers = workers
        self.brain = brain
        self._seen_hashes: set[str] = set()

    def curate(self, input_files: list[Path] | None = None) -> dict:
        files = input_files or list(RAW_DATA_DIR.glob("*.jsonl"))
        if not files:
            return {"processed": 0, "kept": 0, "rejected": 0}

        output = CURATED_DATA_DIR / "dataset.jsonl"
        kept = rejected = 0

        # Load all raw examples
        all_convs = []
        for f in files:
            all_convs.extend(_load_jsonl(f))

        if self.workers > 1 and not self.use_local_judge:
            # Parallel processing (no shared state — safe for dedup-free pass)
            processed = self._parallel_process(all_convs)
        else:
            processed = [self._process(c) for c in all_convs]

        with open(output, "w") as out:
            for result in processed:
                if result:
                    out.write(json.dumps(result) + "\n")
                    kept += 1
                else:
                    rejected += 1

        return {
            "processed": kept + rejected,
            "kept": kept,
            "rejected": rejected,
            "reject_rate": f"{rejected/(kept+rejected)*100:.1f}%" if (kept+rejected) else "0%",
            "output": str(output),
        }

    def _parallel_process(self, convs: list[dict]) -> list[dict | None]:
        """Process examples in parallel using thread pool."""
        hashes: set[str] = set()

        def process_with_dedup(conv):
            h = _hash_conversation(conv)
            if h in hashes:
                return None
            hashes.add(h)
            return _process_single(conv)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(process_with_dedup, convs))

    def _process(self, conv: dict) -> dict | None:
        # Dedup
        h = _hash_conversation(conv)
        if h in self._seen_hashes:
            return None
        self._seen_hashes.add(h)

        result = _process_single(conv)
        if result is None:
            return None

        # Local AI judge (optional)
        if self.use_local_judge and self.brain:
            score = local_score(result, self.brain)
            if score:
                if not score.get("keep", True):
                    return None
                if score.get("overall", 10) < self.min_score:
                    return None
                result.setdefault("metadata", {})["quality_score"] = score.get("overall", 0)

        return result

    def stats(self) -> dict:
        curated = CURATED_DATA_DIR / "dataset.jsonl"
        if not curated.exists():
            return {"examples": 0}
        count = sum(1 for _ in open(curated))
        return {"examples": count, "path": str(curated)}

    def inspect(self, n: int = 5) -> list[dict]:
        """Sample n curated examples for manual review."""
        curated = CURATED_DATA_DIR / "dataset.jsonl"
        if not curated.exists():
            return []
        import random
        all_lines = open(curated).readlines()
        sampled = random.sample(all_lines, min(n, len(all_lines)))
        return [json.loads(l) for l in sampled]


def _process_single(conv: dict) -> dict | None:
    """Stateless processing — safe to run in parallel."""
    # Length gate
    if not _min_length_ok(conv):
        return None

    # Hard reject on sycophancy
    for turn in conv["conversations"]:
        val = turn.get("value", turn.get("content", ""))
        role = turn.get("role", "")
        if role == "gpt" and _has_hard_reject(val):
            return None

    # Strip softcoded bad patterns
    for turn in conv["conversations"]:
        role = turn.get("role", "")
        if role == "gpt":
            key = "value" if "value" in turn else "content"
            turn[key] = _strip_sycophancy(turn[key])
            if len(turn[key].split()) < 5:
                return None

    return conv
