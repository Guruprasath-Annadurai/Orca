"""
Orca Fine-Tuner — QLoRA training with Unsloth (2x faster, 60% less VRAM).

Pipeline:
1. Load base model with 4-bit quantization
2. Attach LoRA adapters to attention + MLP layers
3. Train on curated Orca conversations
4. Save merged model ready for GGUF export

Run:
    orca train run --preset prosumer
    orca train run --preset cloud --epochs 5
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from orca.train.config import TrainingConfig, MODELS_DIR
from orca.config import ORCA_HOME

FORMATTED_DIR = ORCA_HOME / "training" / "formatted"


def _check_deps():
    missing = []
    for pkg in ["unsloth", "trl", "transformers", "datasets", "peft", "bitsandbytes"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"Missing training dependencies: {', '.join(missing)}\n"
            f"Run: pip install unsloth trl transformers datasets peft bitsandbytes accelerate"
        )


def train(cfg: TrainingConfig, on_log: Callable[[str], None] | None = None) -> dict:
    """
    Full QLoRA fine-tuning pipeline.
    Returns paths to saved model artifacts.
    """
    log = on_log or print
    _check_deps()

    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset

    log(f"[Train] Loading base model: {cfg.base_model}")
    log(f"[Train] LoRA rank: {cfg.lora.r} | 4-bit: {cfg.load_in_4bit} | seq_len: {cfg.max_seq_length}")

    # ── Step 1: Load model + tokenizer ────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.load_in_4bit,
    )

    # ── Step 2: Attach LoRA adapters ──────────────────────────────────────────
    log("[Train] Attaching LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        target_modules=cfg.lora.target_modules,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        use_gradient_checkpointing="unsloth",  # saves VRAM
        random_state=42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"[Train] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Step 3: Load dataset ───────────────────────────────────────────────────
    train_file = cfg.train_file or str(FORMATTED_DIR / f"orca_{cfg.data_format}_train.jsonl")
    eval_file = cfg.eval_file or str(FORMATTED_DIR / f"orca_{cfg.data_format}_eval.jsonl")

    if not Path(train_file).exists():
        raise FileNotFoundError(
            f"Training data not found: {train_file}\n"
            f"Run first: orca data curate && orca data format"
        )

    log(f"[Train] Loading dataset: {train_file}")
    data_files = {"train": train_file}
    if Path(eval_file).exists():
        data_files["validation"] = eval_file

    dataset = load_dataset("json", data_files=data_files)
    log(f"[Train] Train examples: {len(dataset['train'])}")

    # ── Step 4: Training arguments ─────────────────────────────────────────────
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps if "validation" in dataset else None,
        evaluation_strategy="steps" if "validation" in dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=True if "validation" in dataset else False,
        report_to="wandb" if cfg.use_wandb else "none",
        run_name=cfg.model_name if cfg.use_wandb else None,
        dataloader_num_workers=2,
        group_by_length=True,   # pack similar-length sequences → faster
        seed=42,
    )

    # ── Step 5: Train ─────────────────────────────────────────────────────────
    log(f"[Train] Starting training for {cfg.num_epochs} epoch(s)...")
    start = time.time()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        dataset_text_field=cfg.dataset_text_field,
        max_seq_length=cfg.max_seq_length,
        args=training_args,
        packing=True,  # pack multiple short examples → higher GPU utilization
    )

    trainer_stats = trainer.train()
    elapsed = time.time() - start
    log(f"[Train] Done in {elapsed/60:.1f} min | loss: {trainer_stats.training_loss:.4f}")

    # ── Step 6: Save LoRA adapters ────────────────────────────────────────────
    adapter_path = output_dir / "lora_adapters"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    log(f"[Train] LoRA adapters saved: {adapter_path}")

    # ── Step 7: Merge and save full model ─────────────────────────────────────
    log("[Train] Merging LoRA into base model (this takes a few minutes)...")
    merged_path = output_dir / "merged"
    model.save_pretrained_merged(
        str(merged_path),
        tokenizer,
        save_method="merged_16bit",
    )
    log(f"[Train] Merged model saved: {merged_path}")

    # Save training metadata
    meta = {
        "model_name": cfg.model_name,
        "base_model": cfg.base_model,
        "lora_rank": cfg.lora.r,
        "epochs": cfg.num_epochs,
        "train_loss": trainer_stats.training_loss,
        "duration_min": elapsed / 60,
        "adapter_path": str(adapter_path),
        "merged_path": str(merged_path),
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta
