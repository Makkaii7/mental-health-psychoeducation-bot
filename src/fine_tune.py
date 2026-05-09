"""
QLoRA fine-tuning with Unsloth + PEFT + TRL (SFTTrainer).
Run on a CUDA GPU (e.g., RTX 5080). Requires `unsloth` compatible with your torch build.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported


def load_config(path: str | Path = "config/config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", default="data/processed/train_hf", help="HF Dataset on disk with `text` field")
    parser.add_argument("--output", default="checkpoints/lora_adapter")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model"]["name"]
    max_seq_length = int(cfg["model"]["max_seq_length"])
    t = cfg["training"]

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
        r=int(t["r"]),
        lora_alpha=int(t["lora_alpha"]),
        lora_dropout=0.0,
        bias="none",
        target_modules=target_modules,
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    train_dataset = load_from_disk(args.dataset)

    sft_config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=float(t.get("epochs", 3)),
        per_device_train_batch_size=int(t["batch_size"]),
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        learning_rate=float(t["lr"]),
        logging_steps=10,
        optim="adamw_8bit",
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        seed=42,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=sft_config,
    )
    trainer.train()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
