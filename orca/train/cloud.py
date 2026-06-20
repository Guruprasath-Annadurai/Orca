"""
Orca Cloud Trainer — SSH-based orchestrator for rented GPU instances.

Workflow:
  1. Connect to remote GPU via SSH
  2. Install Unsloth + training deps
  3. Upload training data + training script via rsync
  4. Launch QLoRA training, stream logs live
  5. Download merged model back via rsync
  6. Trigger local GGUF export + ollama create orca

Usage:
    trainer = CloudTrainer(ssh="ssh root@1.2.3.4 -p 22 -i ~/.ssh/id_rsa", preset="cloud")
    trainer.run()
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.text import Text
from rich import box

from orca.config import ORCA_HOME

console = Console()

FORMATTED_DIR = ORCA_HOME / "training" / "formatted"
MODELS_DIR    = ORCA_HOME / "models"
MODELS_DIR.mkdir(exist_ok=True)

REMOTE_WORKDIR = "/root/orca_train"

# ─────────────────────────────────────────────────────────────────────────────
#  Remote training script (uploaded and executed on the GPU box)
# ─────────────────────────────────────────────────────────────────────────────

_TRAIN_SCRIPT = '''#!/usr/bin/env python3
"""Auto-generated Orca QLoRA training script."""
import json, os, sys, time
from pathlib import Path

preset = os.environ.get("ORCA_PRESET", "cloud")
epochs = int(os.environ.get("ORCA_EPOCHS", "3"))
rank   = int(os.environ.get("ORCA_RANK", "128"))
base   = os.environ.get("ORCA_BASE_MODEL", "unsloth/Meta-Llama-3.1-8B-Instruct")
train_file = "data/orca_llama3_train.jsonl"
eval_file  = "data/orca_llama3_eval.jsonl"

print(f"[setup] preset={preset} epochs={epochs} rank={rank} base={base}")

from unsloth import FastLanguageModel
import torch

max_seq_length = 4096
print("[model] loading base model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=rank,
    lora_alpha=rank * 2,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

from datasets import Dataset
import json

def load_jsonl(path):
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except Exception:
                    pass
    return lines

def format_conv(ex):
    turns = ex.get("conversations", [])
    parts = []
    for t in turns:
        role = t.get("role", "")
        val  = t.get("value", "")
        if role == "system":
            parts.append(f"<|start_header_id|>system<|end_header_id|>\\n\\n{val}<|eot_id|>")
        elif role == "human":
            parts.append(f"<|start_header_id|>user<|end_header_id|>\\n\\n{val}<|eot_id|>")
        elif role == "gpt":
            parts.append(f"<|start_header_id|>assistant<|end_header_id|>\\n\\n{val}<|eot_id|>")
    return "".join(parts)

print("[data] loading dataset...")
raw_train = load_jsonl(train_file)
raw_eval  = load_jsonl(eval_file) if Path(eval_file).exists() else raw_train[:max(1, len(raw_train)//10)]

train_ds = Dataset.from_list([{"text": format_conv(ex)} for ex in raw_train])
eval_ds  = Dataset.from_list([{"text": format_conv(ex)} for ex in raw_eval])
print(f"[data] train={len(train_ds)} eval={len(eval_ds)}")

from trl import SFTTrainer
from transformers import TrainingArguments

batch = 8 if preset in ("cloud", "cloud_xl") else 4
grad_acc = 2 if preset in ("cloud", "cloud_xl") else 4

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_acc,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        output_dir="output",
        evaluation_strategy="steps",
        report_to="none",
    ),
)

print("[train] starting QLoRA training...")
t0 = time.time()
trainer.train()
elapsed = (time.time() - t0) / 60
print(f"[train] done in {elapsed:.1f} min | loss={trainer.state.log_history[-1].get('train_loss', 0):.4f}")

print("[merge] merging LoRA adapters...")
model.save_pretrained_merged("merged", tokenizer, save_method="merged_16bit")
print("[merge] saved to ./merged")

print("[gguf] converting to GGUF q4_k_m...")
import subprocess
result = subprocess.run(
    ["python", "-m", "llama_cpp.server", "--help"],
    capture_output=True,
)
# Use unsloth GGUF export
model.save_pretrained_gguf("gguf", tokenizer, quantization_method="q4_k_m")
print("[gguf] saved to ./gguf")
print("[done]")
'''

# ─────────────────────────────────────────────────────────────────────────────
#  SSH helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ssh(ssh_cmd: str) -> list[str]:
    """Convert 'ssh root@host -p 22 -i key' to arg list for subprocess."""
    parts = ssh_cmd.strip().split()
    if parts[0] == "ssh":
        parts = parts[1:]
    return parts


def _ssh_run(
    args: list[str],
    remote_cmd: str,
    timeout: int | None = None,
    on_line: Callable[[str], None] | None = None,
) -> int:
    """Run a command on the remote host, streaming stdout line by line."""
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15"] + args + [remote_cmd]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for line in iter(proc.stdout.readline, ""):  # type: ignore[union-attr]
            line = line.rstrip()
            if on_line:
                on_line(line)
            else:
                console.print(f"  [dim]{line}[/dim]")
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return 1
    return proc.returncode


def _rsync_up(args: list[str], local: str, remote: str) -> int:
    """rsync local → remote."""
    host_part = next((a for a in args if "@" in a), None)
    port_part = None
    for i, a in enumerate(args):
        if a == "-p" and i + 1 < len(args):
            port_part = args[i + 1]
            break
    key_part = None
    for i, a in enumerate(args):
        if a == "-i" and i + 1 < len(args):
            key_part = args[i + 1]
            break

    rsync_ssh = "ssh -o StrictHostKeyChecking=no"
    if port_part:
        rsync_ssh += f" -p {port_part}"
    if key_part:
        rsync_ssh += f" -i {key_part}"

    cmd = ["rsync", "-avz", "--progress", "-e", rsync_ssh, local, f"{host_part}:{remote}"]
    result = subprocess.run(cmd)
    return result.returncode


def _rsync_down(args: list[str], remote: str, local: str) -> int:
    """rsync remote → local."""
    host_part = next((a for a in args if "@" in a), None)
    port_part = None
    for i, a in enumerate(args):
        if a == "-p" and i + 1 < len(args):
            port_part = args[i + 1]
            break
    key_part = None
    for i, a in enumerate(args):
        if a == "-i" and i + 1 < len(args):
            key_part = args[i + 1]
            break

    rsync_ssh = "ssh -o StrictHostKeyChecking=no"
    if port_part:
        rsync_ssh += f" -p {port_part}"
    if key_part:
        rsync_ssh += f" -i {key_part}"

    cmd = ["rsync", "-avz", "--progress", "-e", rsync_ssh, f"{host_part}:{remote}", local]
    result = subprocess.run(cmd)
    return result.returncode


# ─────────────────────────────────────────────────────────────────────────────
#  CloudTrainer
# ─────────────────────────────────────────────────────────────────────────────

class CloudTrainer:
    """
    Orchestrates end-to-end QLoRA training on a rented GPU instance.

    Args:
        ssh: Full SSH connection string, e.g. "ssh root@1.2.3.4 -p 22 -i ~/.ssh/id_rsa"
        preset: Training preset (cloud|cloud_xl|prosumer)
        epochs: Number of training epochs
        model_name: Name for the resulting Ollama model
    """

    def __init__(
        self,
        ssh: str,
        preset: str = "cloud",
        epochs: int = 3,
        model_name: str = "orca",
    ):
        self.ssh = ssh
        self.ssh_args = _parse_ssh(ssh)
        self.preset = preset
        self.epochs = epochs
        self.model_name = model_name

        from orca.train.prepare import GPU_TIERS
        rank_map = {g["preset"]: g["lora_rank"] for g in GPU_TIERS}
        self.rank = rank_map.get(preset, 128)

        from orca.train.config import TrainingConfig
        cfg = TrainingConfig.preset(preset)
        self.base_model = cfg.base_model

    def run(self) -> dict:
        """Full pipeline: setup → upload → train → download → register."""
        results: dict = {}
        start = time.time()

        self._step("Checking connectivity")
        rc = _ssh_run(self.ssh_args, "echo OK", timeout=20)
        if rc != 0:
            raise RuntimeError("Cannot reach remote host. Check your SSH command.")
        console.print("  [green]✓[/green] Connected\n")

        self._step("Installing training dependencies")
        install_cmd = (
            "pip install -q 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git' "
            "trl transformers datasets peft bitsandbytes accelerate 2>&1 | tail -5"
        )
        _ssh_run(self.ssh_args, install_cmd, timeout=600, on_line=self._log)
        console.print("  [green]✓[/green] Dependencies installed\n")

        self._step("Creating remote workspace")
        _ssh_run(self.ssh_args, f"mkdir -p {REMOTE_WORKDIR}/data {REMOTE_WORKDIR}/output {REMOTE_WORKDIR}/merged {REMOTE_WORKDIR}/gguf")

        self._step("Uploading training data")
        train_file = FORMATTED_DIR / "orca_llama3_train.jsonl"
        eval_file  = FORMATTED_DIR / "orca_llama3_eval.jsonl"
        if not train_file.exists():
            raise FileNotFoundError(f"Train file not found: {train_file}\nRun: orca data format --format llama3 --split")
        rc = _rsync_up(self.ssh_args, str(train_file), f"{REMOTE_WORKDIR}/data/")
        if eval_file.exists():
            _rsync_up(self.ssh_args, str(eval_file), f"{REMOTE_WORKDIR}/data/")
        console.print("  [green]✓[/green] Data uploaded\n")

        self._step("Uploading training script")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(_TRAIN_SCRIPT)
            script_path = f.name
        _rsync_up(self.ssh_args, script_path, f"{REMOTE_WORKDIR}/train.py")
        console.print("  [green]✓[/green] Script uploaded\n")

        self._step(f"Training — {self.preset} preset | {self.epochs} epochs | rank {self.rank}")
        console.print()
        train_env = (
            f"ORCA_PRESET={self.preset} "
            f"ORCA_EPOCHS={self.epochs} "
            f"ORCA_RANK={self.rank} "
            f"ORCA_BASE_MODEL={self.base_model}"
        )
        rc = _ssh_run(
            self.ssh_args,
            f"cd {REMOTE_WORKDIR} && {train_env} python train.py 2>&1",
            timeout=7200,
            on_line=self._train_log,
        )
        if rc != 0:
            raise RuntimeError("Training failed. Check the log above for errors.")
        console.print("\n  [green]✓[/green] Training complete\n")

        self._step("Downloading merged model")
        local_merged = MODELS_DIR / f"{self.model_name}-merged"
        local_merged.mkdir(exist_ok=True)
        rc = _rsync_down(self.ssh_args, f"{REMOTE_WORKDIR}/merged/", str(local_merged) + "/")
        results["merged_path"] = str(local_merged)
        console.print(f"  [green]✓[/green] Model at {local_merged}\n")

        self._step("Downloading GGUF")
        local_gguf = MODELS_DIR / f"{self.model_name}.gguf"
        rc = _rsync_down(self.ssh_args, f"{REMOTE_WORKDIR}/gguf/", str(MODELS_DIR) + "/")
        # Find the q4_k_m file
        gguf_files = list(MODELS_DIR.glob("*q4_k_m*.gguf"))
        if gguf_files:
            results["gguf_path"] = str(gguf_files[0])
        console.print(f"  [green]✓[/green] GGUF downloaded\n")

        self._step(f"Registering as '{self.model_name}' in Ollama")
        self._register_ollama(results.get("gguf_path", ""))
        results["ollama_model"] = self.model_name
        console.print(f"  [green]✓[/green] Model registered: {self.model_name}\n")

        results["duration_min"] = round((time.time() - start) / 60, 1)
        self._print_summary(results)
        return results

    def _register_ollama(self, gguf_path: str) -> None:
        """Create a Modelfile and run ollama create."""
        if not gguf_path or not Path(gguf_path).exists():
            console.print("  [yellow]⚠[/yellow]  No GGUF found — skipping Ollama registration")
            console.print(f"     Run manually: [bold]orca train export <merged_path>[/bold]")
            return

        from orca.data.collector import ORCA_SYSTEM_PROMPT
        modelfile = MODELS_DIR / "Modelfile"
        modelfile.write_text(
            f'FROM {gguf_path}\n'
            f'SYSTEM """{ORCA_SYSTEM_PROMPT}"""\n'
            f'PARAMETER temperature 0.7\n'
            f'PARAMETER top_p 0.9\n'
            f'PARAMETER num_ctx 8192\n'
        )
        result = subprocess.run(
            ["ollama", "create", self.model_name, "-f", str(modelfile)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            console.print(f"  [yellow]⚠[/yellow]  ollama create failed: {result.stderr[:200]}")
            console.print(f"     Run manually: [bold]ollama create {self.model_name} -f {modelfile}[/bold]")

    def _step(self, label: str) -> None:
        console.print(Rule(f"[dim]{label}[/dim]"))

    def _log(self, line: str) -> None:
        if line:
            console.print(f"  [dim]{line}[/dim]")

    def _train_log(self, line: str) -> None:
        if not line:
            return
        if line.startswith("[train]") or line.startswith("[merge]") or line.startswith("[gguf]") or line.startswith("[done]"):
            console.print(f"  [cyan]{line}[/cyan]")
        elif "loss=" in line or "Loss:" in line:
            console.print(f"  [green]{line}[/green]")
        elif "Error" in line or "error" in line:
            console.print(f"  [red]{line}[/red]")
        else:
            console.print(f"  [dim]{line}[/dim]")

    def _print_summary(self, results: dict) -> None:
        console.print()
        console.print(Panel(
            f"[bold]Model:[/bold]    [green]{results.get('ollama_model', self.model_name)}[/green]\n"
            f"[bold]Merged:[/bold]   {results.get('merged_path', '—')}\n"
            f"[bold]GGUF:[/bold]     {results.get('gguf_path', '—')}\n"
            f"[bold]Duration:[/bold] {results.get('duration_min', 0)} min\n\n"
            f"[bold]Test it:[/bold]  [cyan]ollama run {self.model_name}[/cyan]\n"
            f"[bold]Use it:[/bold]   [cyan]ORCA_CORE_MODEL={self.model_name} orca core chat[/cyan]",
            title="[green bold]Training Complete[/green bold]",
            border_style="green",
        ))
