"""
Compare LoRA adapter output at different scaling factors.

Requires:
  - checkpoints/lora_adapter populated (e.g. ``hf download Makkaii/qwen3-8b-psychoed-lora --local-dir checkpoints/lora_adapter`` after ``hf auth login``)
  - CUDA recommended

The original one-liner reuses ``base`` after the first ``PeftModel.from_pretrained``, which wraps the
model in-place. This script loads the adapter once, snapshots ``module.scaling`` dicts, then for each
global factor resets from the snapshot and multiplies (so 1.0 / 0.5 / 0.3 / 0.15 are independent).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "checkpoints" / "lora_adapter"
MODEL_NAME = "Qwen/Qwen3-8B"


def _snapshot_scaling(model: PeftModel) -> list[tuple[object, dict[str, torch.Tensor | float]]]:
    snaps: list[tuple[object, dict[str, torch.Tensor | float]]] = []
    for _name, module in model.named_modules():
        sc = getattr(module, "scaling", None)
        if isinstance(sc, dict) and sc:
            snap: dict[str, torch.Tensor | float] = {}
            for k, v in sc.items():
                if isinstance(v, torch.Tensor):
                    snap[k] = v.detach().clone()
                else:
                    snap[k] = float(v)
            snaps.append((module, snap))
    return snaps


def _apply_global_scale(snaps: list[tuple[object, dict[str, torch.Tensor | float]]], factor: float) -> None:
    for module, orig in snaps:
        sc = module.scaling
        for k, v in orig.items():
            if isinstance(v, torch.Tensor):
                sc[k] = v * factor
            else:
                sc[k] = v * factor


def main() -> None:
    if not ADAPTER.is_dir() or not any(ADAPTER.iterdir()):
        print(
            "ERROR: checkpoints/lora_adapter is missing or empty.\n"
            "Authenticate and download, e.g.:\n"
            "  hf auth login\n"
            "  hf download Makkaii/qwen3-8b-psychoed-lora --local-dir checkpoints/lora_adapter",
            file=sys.stderr,
        )
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base, str(ADAPTER))
    try:
        model.set_adapter("default")
    except (ValueError, AttributeError):
        names = list(getattr(model, "peft_config", {}) or {})
        if names:
            model.set_adapter(names[0])

    snaps = _snapshot_scaling(model)
    if not snaps:
        print("WARNING: no LoRA scaling dicts found; adapter may be empty or incompatible.", file=sys.stderr)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a psychoeducation thinking partner. Help users think clearly about mental well-being. "
                "Be Socratic and non-directive. Never diagnose or prescribe. Use proper English."
            ),
        },
        {"role": "user", "content": "What do you know about depression?"},
    ]
    tmpl_kw: dict = {"tokenize": False, "add_generation_prompt": True}
    try:
        text = tokenizer.apply_chat_template(messages, **tmpl_kw, enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **tmpl_kw)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    for scale in (1.0, 0.5, 0.3, 0.15):
        _apply_global_scale(snaps, scale)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.5,
                do_sample=True,
                repetition_penalty=1.3,
                no_repeat_ngram_size=4,
                pad_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        print(f"=== SCALE {scale} ===")
        print(reply.strip())
        print()


if __name__ == "__main__":
    main()
