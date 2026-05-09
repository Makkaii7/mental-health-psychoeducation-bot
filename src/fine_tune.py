"""
QLoRA fine-tuning: Unsloth (preferred on GPU) or HuggingFace + PEFT + TRL fallback.

Run from project root: ``python -m src.fine_tune`` or ``python src/fine_tune.py``

Unsloth requires a CUDA GPU. If Unsloth import or model load fails on Windows, set
``USE_HF_PEFT=1`` or the script auto-falls back to ``AutoModelForCausalLM`` + BitsAndBytes + ``get_peft_model``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer


def load_config(path: str | Path = "config/config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_model_hf_peft(
    model_name: str,
    max_seq_length: int,
    training_cfg: dict,
) -> tuple:
    """Standard HF QLoRA path (NF4) + PEFT LoRA."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    lora_config = LoraConfig(
        r=int(training_cfg["r"]),
        lora_alpha=int(training_cfg["lora_alpha"]),
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def _load_model_unsloth(model_name: str, max_seq_length: int, training_cfg: dict) -> tuple:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(training_cfg["r"]),
        lora_alpha=int(training_cfg["lora_alpha"]),
        lora_dropout=0.0,
        bias="none",
        target_modules=target_modules,
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", default="data/processed/train_hf")
    parser.add_argument("--eval_dataset", default="data/processed/val_hf")
    parser.add_argument("--output", default="checkpoints/lora_adapter")
    parser.add_argument(
        "--use-hf-peft",
        action="store_true",
        help="Force HuggingFace + BitsAndBytes + PEFT (skip Unsloth).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA is not available. QLoRA fine-tuning for Qwen3-8B requires an NVIDIA GPU "
            "with a CUDA-enabled PyTorch build.",
            file=sys.stderr,
        )
        sys.exit(2)

    cfg = load_config(args.config)
    model_name = cfg["model"]["name"]
    max_seq_length = int(cfg["model"]["max_seq_length"])
    t = cfg["training"]

    force_hf = args.use_hf_peft or os.environ.get("USE_HF_PEFT", "").lower() in ("1", "true", "yes")
    use_unsloth = not force_hf
    is_bf16_supp = torch.cuda.is_bf16_supported()

    if use_unsloth:
        try:
            from unsloth import is_bfloat16_supported as unsloth_bf16

            is_bf16_supp = unsloth_bf16()
            model, tokenizer = _load_model_unsloth(model_name, max_seq_length, t)
            print("[fine_tune] Using Unsloth FastLanguageModel path.")
        except Exception as e:  # pragma: no cover
            print(f"[fine_tune] Unsloth failed ({e!r}); falling back to HF PEFT.", file=sys.stderr)
            use_unsloth = False

    if not use_unsloth:
        model, tokenizer = _load_model_hf_peft(model_name, max_seq_length, t)
        print("[fine_tune] Using HuggingFace AutoModelForCausalLM + BitsAndBytes + get_peft_model.")

    train_dataset = load_from_disk(args.dataset)
    eval_path = Path(args.eval_dataset)
    eval_dataset = load_from_disk(str(eval_path)) if eval_path.exists() else None

    callbacks = []
    if eval_dataset is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

    sft_kwargs = dict(
        output_dir=args.output,
        num_train_epochs=float(t.get("epochs", 3)),
        per_device_train_batch_size=int(t["batch_size"]),
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        learning_rate=float(t["lr"]),
        logging_steps=10,
        optim="adamw_8bit" if use_unsloth else "adamw_torch",
        warmup_steps=10,
        lr_scheduler_type="linear",
        seed=42,
        bf16=is_bf16_supp,
        fp16=not is_bf16_supp,
        max_length=max_seq_length,
        dataset_text_field="text",
        report_to="none",
        save_steps=100,
        save_total_limit=3,
    )

    if eval_dataset is not None:
        sft_kwargs.update(
            eval_strategy="steps",
            eval_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    else:
        sft_kwargs["eval_strategy"] = "no"

    sft_config = SFTConfig(**sft_kwargs)

    trainer_kwargs: dict = dict(
        model=model,
        train_dataset=train_dataset,
        args=sft_config,
        callbacks=callbacks,
    )
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer = SFTTrainer(**trainer_kwargs)

    if torch.cuda.is_available():
        alloc_gb = torch.cuda.memory_allocated() / 1e9
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        props = torch.cuda.get_device_properties(0)
        print(
            f"[fine_tune] Model ready. VRAM allocated ~{alloc_gb:.2f} GB, "
            f"peak so far ~{peak_gb:.2f} GB / {props.total_memory / 1e9:.1f} GB device."
        )

    print("[fine_tune] Starting trainer.train() …")
    trainer.train()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"[fine_tune] Saved adapter to {args.output}")


if __name__ == "__main__":
    main()
