"""
End-to-end chatbot: safety routing, RAG context, and local LLM generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.rag_pipeline import format_context, load_vectorstore, retrieve
from src.safety import (
    StickySession,
    classify_tier,
    get_crisis_response,
    get_redirect_response,
)


def _load_system_prompt() -> str:
    root = Path(__file__).resolve().parents[1]
    p = root / "prompts" / "system_prompt.txt"
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def load_config(path: str | Path = "config/config.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(
    base_model_name: str | None = None,
    adapter_path: str | Path | None = "checkpoints/lora_adapter",
    config_path: str | Path = "config/config.yaml",
) -> tuple[Any, Any]:
    """
    Load base Qwen in 4-bit NF4 and attach fine-tuned LoRA adapter (if ``adapter_path`` exists).
    """
    cfg = load_config(config_path)
    name = base_model_name or cfg["model"]["name"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    adapter = Path(adapter_path) if adapter_path else None
    if adapter and adapter.is_dir() and any(adapter.iterdir()):
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, tokenizer


class ChatBot:
    def __init__(
        self,
        model,
        tokenizer,
        vectorstore,
        system_prompt: str | None = None,
        top_k: int = 3,
        max_new_tokens: int = 512,
        config_path: str | Path = "config/config.yaml",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.vectorstore = vectorstore
        self.system_prompt = system_prompt or _load_system_prompt()
        cfg = load_config(config_path)
        self.top_k = int(cfg.get("rag", {}).get("top_k", top_k))
        sticky_on = bool(cfg.get("safety", {}).get("enable_sticky_crisis", True))
        self.session = StickySession(enable_sticky=sticky_on)
        self.max_new_tokens = max_new_tokens

    def respond(self, user_message: str) -> str:
        if self.session.blocked_from_normal_chat():
            return get_crisis_response()

        tier = classify_tier(user_message)
        self.session.note_tier(tier)

        if tier == 4:
            return get_crisis_response()
        if tier == 3:
            return get_redirect_response()

        chunks = retrieve(self.vectorstore, user_message, k=self.top_k)
        context = format_context(chunks)
        return self._generate(user_message, context)

    def _generate(self, user_message: str, context: str) -> str:
        user_block = (
            f"Retrieved psychoeducation context (may be incomplete; do not invent facts beyond it):\n{context}\n\n"
            f"User message:\n{user_message}"
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_block},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()
