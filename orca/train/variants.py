"""
Atheris model variant definitions.

nano  → Qwen2.5-3B-Instruct   — fast, local, minimal VRAM (fits 8GB)
core  → Llama-3.1-8B-Instruct — balanced default
ultra → Llama-3.1-70B         — maximum quality, cloud-only

Each variant has:
- base model for fine-tuning
- Ollama model name (what `ollama run` uses)
- Modelfile parameters tuned for the variant's role
- LoRA rank appropriate for the base model size
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from orca.config import ORCA_HOME
from orca.data.collector import ORCA_SYSTEM_PROMPT

MODELS_DIR = ORCA_HOME / "models"
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class VariantSpec:
    name: str                    # orca-nano | orca-core | orca-ultra
    ollama_name: str             # name used in `ollama run`
    base_model: str              # HuggingFace model ID for fine-tuning
    lora_rank: int
    batch_size: int
    gradient_accumulation: int
    max_seq_length: int
    temperature: float
    top_p: float
    num_ctx: int
    description: str
    vram_gb: int                 # minimum VRAM for fine-tuning
    preset: str                  # training preset alias


VARIANTS: dict[str, VariantSpec] = {
    "nano": VariantSpec(
        name="orca-nano",
        ollama_name="orca-nano",
        base_model="unsloth/Qwen2.5-3B-Instruct",
        lora_rank=32,
        batch_size=4,
        gradient_accumulation=4,
        max_seq_length=4096,
        temperature=0.7,
        top_p=0.9,
        num_ctx=4096,
        description="Fast 3B model — instant responses, minimal RAM",
        vram_gb=6,
        preset="laptop",
    ),
    "core": VariantSpec(
        name="orca-core",
        ollama_name="orca-core",
        base_model="unsloth/Meta-Llama-3.1-8B-Instruct",
        lora_rank=64,
        batch_size=2,
        gradient_accumulation=8,
        max_seq_length=8192,
        temperature=0.7,
        top_p=0.9,
        num_ctx=8192,
        description="Balanced 8B model — quality + speed sweet spot",
        vram_gb=16,
        preset="prosumer",
    ),
    "ultra": VariantSpec(
        name="orca-ultra",
        ollama_name="orca-ultra",
        base_model="unsloth/Meta-Llama-3.1-70B-Instruct",
        lora_rank=64,
        batch_size=4,
        gradient_accumulation=4,
        max_seq_length=8192,
        temperature=0.6,
        top_p=0.95,
        num_ctx=16384,
        description="Maximum quality 70B — cloud GPU only",
        vram_gb=48,
        preset="cloud_xl",
    ),
}


def get_variant(name: str) -> VariantSpec:
    """Resolve 'nano'/'orca-nano'/'core'/'orca-core' → VariantSpec."""
    key = name.removeprefix("orca-")
    if key not in VARIANTS:
        raise ValueError(f"Unknown variant '{name}'. Available: {list(VARIANTS)}")
    return VARIANTS[key]


def build_modelfile(variant: VariantSpec, gguf_path: str) -> str:
    """Generate an Ollama Modelfile for this variant."""
    system = ORCA_SYSTEM_PROMPT.strip()
    return (
        f"FROM {gguf_path}\n\n"
        f'SYSTEM """\n{system}\n"""\n\n'
        f"PARAMETER temperature {variant.temperature}\n"
        f"PARAMETER top_p {variant.top_p}\n"
        f"PARAMETER num_ctx {variant.num_ctx}\n"
        f"PARAMETER stop \"<|eot_id|>\"\n"
        f"PARAMETER stop \"<|im_end|>\"\n"
    )


def register_with_ollama(
    variant: VariantSpec,
    gguf_path: str,
    on_log=None,
) -> bool:
    """
    Create / update the Ollama model for this variant.
    Returns True on success.
    """
    log = on_log or print
    modelfile_path = MODELS_DIR / f"Modelfile.{variant.ollama_name}"
    modelfile_path.write_text(build_modelfile(variant, gguf_path))
    log(f"[register] Modelfile → {modelfile_path}")

    result = subprocess.run(
        ["ollama", "create", variant.ollama_name, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        log(f"[register] ✓ {variant.ollama_name} registered in Ollama")
        return True
    log(f"[register] ✗ ollama create failed: {result.stderr[:300]}")
    log(f"[register]   Manual: ollama create {variant.ollama_name} -f {modelfile_path}")
    return False


def list_ollama_models() -> set[str]:
    """Return set of model names currently in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=5,
        )
        models = set()
        for line in result.stdout.splitlines()[1:]:  # skip header
            name = line.split()[0] if line.split() else ""
            if name:
                models.add(name.split(":")[0])  # strip :latest etc.
        return models
    except Exception:
        return set()


def status() -> list[dict]:
    """Return build status for all three variants."""
    pulled = list_ollama_models()
    rows = []
    for key, v in VARIANTS.items():
        gguf_files = list(MODELS_DIR.glob(f"{v.ollama_name}*q4*.gguf"))
        merged_path = MODELS_DIR / f"{v.ollama_name}-merged"
        rows.append({
            "variant":     key,
            "name":        v.ollama_name,
            "description": v.description,
            "vram_gb":     v.vram_gb,
            "in_ollama":   v.ollama_name in pulled,
            "gguf_exists": bool(gguf_files),
            "gguf_path":   str(gguf_files[0]) if gguf_files else None,
            "merged_exists": merged_path.exists(),
            "merged_path": str(merged_path) if merged_path.exists() else None,
        })
    return rows
