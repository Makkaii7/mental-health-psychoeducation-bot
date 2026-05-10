"""Measure respond() latency (5 messages) and peak GPU memory. Prints JSON lines to stdout."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from src.chatbot import ChatBot, load_model
from src.rag_pipeline import chunk_documents, create_vectorstore, load_corpus, load_vectorstore
from src.safety import StickySession


def main() -> None:
    cfg_path = ROOT / "config" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f) or {}
    rag_cfg = full_cfg.get("rag") or {}
    embed_model = rag_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    persist = ROOT / "chroma_db"
    chunk_size = int(rag_cfg.get("chunk_size", 500))
    overlap = int(rag_cfg.get("chunk_overlap", 50))

    model, tokenizer = load_model(config_path=cfg_path)
    vectorstore = None
    if persist.exists() and any(persist.iterdir()):
        vectorstore = load_vectorstore(embedding_model=embed_model, persist_directory=persist)
    else:
        docs = load_corpus(ROOT / "data" / "rag_corpus")
        if docs:
            chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=overlap)
            vectorstore = create_vectorstore(
                chunks, embedding_model=embed_model, persist_directory=persist
            )

    bot = ChatBot(model, tokenizer, vectorstore, config_path=cfg_path, max_new_tokens=256)

    cases = [
        ("tier1", "What does it mean to have good mental health?"),
        ("tier2", "I've been feeling really anxious for the past few weeks"),
        ("tier3", "Can you diagnose me with depression?"),
        ("tier4", "I want to kill myself"),
        ("greeting", "hi"),
    ]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    latencies: dict[str, float] = {}
    for name, msg in cases:
        sess = StickySession(enable_sticky=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = bot.respond(msg.strip(), session=sess, history=[])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies[name] = time.perf_counter() - t0

    avg_s = sum(latencies.values()) / len(latencies)
    peak_gb = None
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_reserved() / 1e9

    out = {"latencies_s": latencies, "avg_latency_s": avg_s, "peak_gpu_memory_reserved_gb": peak_gb}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
