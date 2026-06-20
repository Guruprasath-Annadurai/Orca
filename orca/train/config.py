"""
Orca Training Configuration — QLoRA hyperparameters and model selection.

Presets designed for different GPU budgets:
- "laptop"   : RTX 3060/4060 (8GB VRAM) — 7B model, rank 8
- "prosumer" : RTX 4090 (24GB VRAM) — 7B model, rank 64
- "cloud"    : A100 40GB — 13B model, rank 128
- "cloud_xl" : A100 80GB — 70B model, rank 64
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from orca.config import ORCA_HOME

MODELS_DIR = ORCA_HOME / "models"
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class LoRAConfig:
    r: int = 64                      # LoRA rank (higher = more capacity, more VRAM)
    lora_alpha: int = 128            # scaling factor (usually 2x rank)
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",   # MLP layers
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    # Base model
    base_model: str = "unsloth/Meta-Llama-3.1-8B-Instruct"
    model_name: str = "orca-8b"

    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True

    # Training
    max_seq_length: int = 4096
    batch_size: int = 2
    gradient_accumulation_steps: int = 8   # effective batch = 16
    num_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True

    # Data
    data_format: str = "llama3"
    train_file: str = ""
    eval_file: str = ""
    dataset_text_field: str = "text"

    # Output
    output_dir: str = str(MODELS_DIR / "orca-8b-qlora")
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10

    # Wandb (optional)
    use_wandb: bool = False
    wandb_project: str = "orca-finetune"

    @classmethod
    def preset(cls, name: str) -> "TrainingConfig":
        cfg = cls()
        if name == "laptop":
            cfg.lora.r = 8
            cfg.lora.lora_alpha = 16
            cfg.batch_size = 1
            cfg.gradient_accumulation_steps = 16
            cfg.max_seq_length = 2048
            cfg.base_model = "unsloth/Meta-Llama-3.1-8B-Instruct"
        elif name == "prosumer":
            cfg.lora.r = 64
            cfg.lora.lora_alpha = 128
            cfg.batch_size = 4
            cfg.gradient_accumulation_steps = 4
        elif name == "cloud":
            cfg.lora.r = 128
            cfg.lora.lora_alpha = 256
            cfg.batch_size = 8
            cfg.gradient_accumulation_steps = 2
            cfg.base_model = "unsloth/Meta-Llama-3.1-8B-Instruct"
        elif name == "cloud_xl":
            cfg.base_model = "unsloth/Meta-Llama-3.1-70B-Instruct"
            cfg.lora.r = 64
            cfg.lora.lora_alpha = 128
            cfg.batch_size = 4
            cfg.load_in_4bit = True
            cfg.model_name = "orca-70b"
            cfg.output_dir = str(MODELS_DIR / "orca-70b-qlora")
        return cfg
