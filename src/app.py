"""
Gradio web UI for the Mental Health Psychoeducation Bot.
Run from project root: ``python -m src.app``
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
from langchain_core.documents import Document

from src.chatbot import ChatBot, load_model
from src.rag_pipeline import chunk_documents, create_vectorstore, load_corpus, load_vectorstore


DISCLAIMER = (
    "**Disclaimer:** This is a student research prototype for psychoeducation-style reflection only. "
    "It is **not** medical advice, therapy, or crisis monitoring. If you may be in danger, "
    "contact Estijaba **800 1717**, HOPE Line **800 4673**, or Emergency **999**."
)


def build_chat_fn(bot: ChatBot):
    def respond(user_message: str, history: list):
        return bot.respond(user_message)

    return respond


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config" / "config.yaml"

    model, tokenizer = load_model(config_path=cfg_path)
    rag_cfg = {}
    if cfg_path.exists():
        import yaml

        with open(cfg_path, encoding="utf-8") as f:
            rag_cfg = (yaml.safe_load(f) or {}).get("rag") or {}
    embed_model = rag_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    persist = root / "chroma_db"
    chunk_size = int(rag_cfg.get("chunk_size", 500))
    overlap = int(rag_cfg.get("chunk_overlap", 50))
    if not persist.exists() or not any(persist.iterdir()):
        docs = load_corpus(root / "data" / "rag_corpus")
        if not docs:
            docs = [
                Document(
                    page_content=(
                        "Placeholder corpus: add trusted psychoeducation `.txt` or `.md` files "
                        "to `data/rag_corpus/` and rebuild the vector store."
                    ),
                    metadata={"source": "placeholder"},
                )
            ]
        chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=overlap)
        vectorstore = create_vectorstore(
            chunks,
            embedding_model=embed_model,
            persist_directory=persist,
        )
    else:
        vectorstore = load_vectorstore(
            embedding_model=embed_model,
            persist_directory=persist,
        )
    bot = ChatBot(model, tokenizer, vectorstore, config_path=cfg_path)

    demo = gr.ChatInterface(
        fn=build_chat_fn(bot),
        title="Mental Health Psychoeducation Bot",
        description=(
            f"{DISCLAIMER}\n\n"
            "This bot supports **Socratic psychoeducation**: reflective questions and grounded "
            "information from a curated corpus. It does **not** provide therapy, diagnoses, or "
            "medical instructions."
        ),
        examples=[
            "What does 'grounding' mean in anxiety psychoeducation?",
            "I've felt lonely lately — can we explore that gently?",
        ],
    )
    demo.launch()


if __name__ == "__main__":
    main()
