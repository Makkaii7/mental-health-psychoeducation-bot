"""
End-to-end chatbot: safety routing, RAG context, and local LLM generation.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.rag_pipeline import format_context, retrieve
from src.safety import (
    StickySession,
    Tier,
    classify_tier,
    get_crisis_keywords,
    get_crisis_response,
    get_redirect_response,
    get_tier2_system_addon,
)


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


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
    """Shared model/RAG; **StickySession must be supplied per user** (e.g. ``gr.State``)."""

    _MAX_HISTORY_TURNS = 5

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
        self.max_new_tokens = max_new_tokens
        self._crisis_keywords = get_crisis_keywords()
        self._tier2_addon = get_tier2_system_addon()

    def respond(
        self,
        user_message: str,
        session: StickySession,
        history: list | None = None,
    ) -> str:
        history = history or []
        if session.blocked_from_normal_chat():
            return get_crisis_response()

        tier = classify_tier(user_message, crisis_keywords=self._crisis_keywords)
        session.note_tier(tier)

        if tier == 4:
            return get_crisis_response()
        if tier == 3:
            return get_redirect_response()

        if self.vectorstore is None:
            chunks = []
        else:
            chunks = retrieve(self.vectorstore, user_message, k=self.top_k)
        context = format_context(chunks)
        return self._generate(user_message, context, tier, history)

    def _build_system_content(self, tier: Tier) -> str:
        base = self.system_prompt
        if tier == 2:
            return f"{base}\n\n---\nAdditional mode (in-scope with care):\n{self._tier2_addon}"
        return base

    def _history_slice(self, history: list) -> list[tuple[str, str]]:
        """Gradio history: list of (user, assistant) tuples."""
        if not history:
            return []
        tail = history[-self._MAX_HISTORY_TURNS :]
        out: list[tuple[str, str]] = []
        for turn in tail:
            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                out.append((str(turn[0]), str(turn[1])))
        return out

    def _generate(
        self,
        user_message: str,
        context: str,
        tier: Tier,
        history: list,
    ) -> str:
        ctx_block = context.strip() if context.strip() else ""
        if not ctx_block:
            ctx_note = (
                "Retrieved context: (none — no relevant passages were retrieved or the corpus is empty.)"
            )
        else:
            ctx_note = "Retrieved psychoeducation context:\n" + ctx_block

        user_block = f"{ctx_note}\n\nCurrent user message:\n{user_message}"

        messages: list[dict[str, str]] = [{"role": "system", "content": self._build_system_content(tier)}]
        for u_prev, a_prev in self._history_slice(history):
            messages.append({"role": "user", "content": u_prev})
            messages.append({"role": "assistant", "content": a_prev})
        messages.append({"role": "user", "content": user_block})

        try:
            template = self.tokenizer.get_chat_template()
        except (ValueError, TypeError):
            template = None
        chat_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if template and "enable_thinking" in template:
            chat_kwargs["enable_thinking"] = False

        prompt = self.tokenizer.apply_chat_template(messages, **chat_kwargs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if "enable_thinking" in inspect.signature(self.model.generate).parameters:
            gen_kwargs["enable_thinking"] = False
        with torch.inference_mode():
            out = self.model.generate(**gen_kwargs)
        gen = out[0, inputs["input_ids"].shape[-1] :]
        decoded = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        return strip_thinking(decoded)
