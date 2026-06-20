"""
Orca Model Exporter — converts merged HuggingFace model to GGUF for Ollama.

Pipeline:
  merged HF model → GGUF (llama.cpp) → Ollama Modelfile → `ollama create orca`

After this, you run:  ollama run orca
And Orca CLI uses YOUR model instead of any API.
"""
from __future__ import annotations

import json
import os
import subprocess
import shutil
from pathlib import Path
from typing import Callable

from orca.config import ORCA_HOME

MODELS_DIR = ORCA_HOME / "models"
EXPORT_DIR = ORCA_HOME / "export"
EXPORT_DIR.mkdir(exist_ok=True)

QUANTIZATION_LEVELS = {
    "q4_k_m": "Best quality/size tradeoff — recommended",
    "q5_k_m": "Higher quality, larger file",
    "q8_0":   "Near-lossless, 2x size of q4",
    "f16":    "Full precision — huge, for reference only",
}

MODELFILE_TEMPLATE = """\
FROM {gguf_path}

SYSTEM \"\"\"\\
{system_prompt}
\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx {context_length}
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
"""

ORCA_SYSTEM = """\
You are Orca — a powerful, thoughtful AI assistant built for speed and depth.
You reason carefully, give direct answers, and never pad responses with filler.
You have persistent memory, can execute code, and work on complex multi-step tasks.\
"""


class ModelExporter:
    """Export fine-tuned Orca model to GGUF and register with Ollama."""

    def __init__(
        self,
        merged_model_path: str,
        quantization: str = "q4_k_m",
        on_log: Callable[[str], None] | None = None,
    ):
        self.model_path = Path(merged_model_path)
        self.quant = quantization
        self.log = on_log or print

        if quantization not in QUANTIZATION_LEVELS:
            raise ValueError(f"Unknown quantization: {quantization}. Options: {list(QUANTIZATION_LEVELS)}")

    def export(self, model_name: str = "orca") -> dict:
        """Full export pipeline: HF → GGUF → Ollama."""
        self.log(f"[Export] Source: {self.model_path}")
        self.log(f"[Export] Quantization: {self.quant} — {QUANTIZATION_LEVELS[self.quant]}")

        # Step 1: Convert to GGUF using llama.cpp
        gguf_path = self._to_gguf()

        # Step 2: Create Ollama Modelfile
        modelfile_path = self._write_modelfile(gguf_path, model_name)

        # Step 3: Register with Ollama
        self._ollama_create(model_name, modelfile_path)

        result = {
            "model_name": model_name,
            "gguf_path": str(gguf_path),
            "modelfile": str(modelfile_path),
            "quantization": self.quant,
            "ollama_run": f"ollama run {model_name}",
            "orca_use": f"ORCA_MODEL_BACKEND=ollama ORCA_OLLAMA_MODEL={model_name} orca core chat",
        }

        out = EXPORT_DIR / "export_meta.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2)

        self.log(f"\n[Export] Done!")
        self.log(f"[Export] Test your model: ollama run {model_name}")
        self.log(f"[Export] Use in Orca: export ORCA_MODEL_BACKEND=ollama ORCA_OLLAMA_MODEL={model_name}")
        return result

    def _to_gguf(self) -> Path:
        """Convert HuggingFace model to GGUF using llama.cpp's convert script."""
        gguf_out = EXPORT_DIR / f"orca_{self.quant}.gguf"

        if gguf_out.exists():
            self.log(f"[Export] GGUF already exists: {gguf_out}")
            return gguf_out

        # Try unsloth's built-in GGUF export first (fastest)
        try:
            self.log("[Export] Converting to GGUF via Unsloth...")
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(self.model_path),
                max_seq_length=4096,
                dtype=None,
                load_in_4bit=False,
            )
            model.save_pretrained_gguf(
                str(EXPORT_DIR / "orca"),
                tokenizer,
                quantization_method=self.quant,
            )
            # Unsloth names it differently — find it
            for f in EXPORT_DIR.glob("*.gguf"):
                f.rename(gguf_out)
                break
            self.log(f"[Export] GGUF saved: {gguf_out}")
            return gguf_out
        except ImportError:
            pass

        # Fallback: llama.cpp convert.py
        self.log("[Export] Unsloth not available, using llama.cpp...")
        llamacpp = self._find_llamacpp()
        if not llamacpp:
            raise RuntimeError(
                "llama.cpp not found. Install it:\n"
                "  git clone https://github.com/ggerganov/llama.cpp\n"
                "  cd llama.cpp && make -j$(nproc)\n"
                "Or install unsloth for automatic export."
            )

        convert_script = llamacpp / "convert_hf_to_gguf.py"
        f16_path = EXPORT_DIR / "orca_f16.gguf"

        subprocess.run([
            "python", str(convert_script),
            str(self.model_path),
            "--outfile", str(f16_path),
            "--outtype", "f16",
        ], check=True)

        # Quantize
        quantize_bin = llamacpp / "llama-quantize"
        subprocess.run([
            str(quantize_bin), str(f16_path), str(gguf_out), self.quant.upper()
        ], check=True)

        if f16_path.exists() and f16_path != gguf_out:
            f16_path.unlink()

        return gguf_out

    def _write_modelfile(self, gguf_path: Path, model_name: str) -> Path:
        modelfile_path = EXPORT_DIR / f"Modelfile.{model_name}"
        content = MODELFILE_TEMPLATE.format(
            gguf_path=str(gguf_path),
            system_prompt=ORCA_SYSTEM,
            context_length=4096,
        )
        modelfile_path.write_text(content)
        self.log(f"[Export] Modelfile: {modelfile_path}")
        return modelfile_path

    def _ollama_create(self, model_name: str, modelfile_path: Path) -> None:
        if not shutil.which("ollama"):
            self.log("[Export] Ollama not found — skipping registration.")
            self.log(f"[Export] Install ollama then run manually:")
            self.log(f"  ollama create {model_name} -f {modelfile_path}")
            return

        self.log(f"[Export] Registering with Ollama as '{model_name}'...")
        subprocess.run([
            "ollama", "create", model_name, "-f", str(modelfile_path)
        ], check=True)
        self.log(f"[Export] Ollama model '{model_name}' ready.")

    def _find_llamacpp(self) -> Path | None:
        candidates = [
            Path.home() / "llama.cpp",
            Path("/usr/local/llama.cpp"),
            Path("./llama.cpp"),
        ]
        for c in candidates:
            if c.exists() and (c / "convert_hf_to_gguf.py").exists():
                return c
        return None
