"""
Gradio web UI for the Mental Health Psychoeducation Bot.
Run from project root: ``python -m src.app``

Uses ``gr.State`` so each user has their own ``StickySession`` (not shared on the shared ``ChatBot``).
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import yaml

from src.chatbot import ChatBot, load_model
from src.rag_pipeline import chunk_documents, create_vectorstore, load_corpus, load_vectorstore
from src.safety import StickySession


DISCLAIMER = (
    "**Disclaimer:** This is a student research prototype for psychoeducation-style reflection only. "
    "It is **not** medical advice, therapy, or crisis monitoring. If you may be in danger, "
    "contact Estijaba **800 1717**, HOPE Line **800 4673**, or Emergency **999**."
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config" / "config.yaml"

    rag_cfg: dict = {}
    sticky_default = True
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f) or {}
        rag_cfg = full_cfg.get("rag") or {}
        sticky_default = bool((full_cfg.get("safety") or {}).get("enable_sticky_crisis", True))

    model, tokenizer = load_model(config_path=cfg_path)
    embed_model = rag_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    persist = root / "chroma_db"
    chunk_size = int(rag_cfg.get("chunk_size", 500))
    overlap = int(rag_cfg.get("chunk_overlap", 50))

    vectorstore = None
    if persist.exists() and any(persist.iterdir()):
        vectorstore = load_vectorstore(
            embedding_model=embed_model,
            persist_directory=persist,
        )
    else:
        docs = load_corpus(root / "data" / "rag_corpus")
        if docs:
            chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=overlap)
            vectorstore = create_vectorstore(
                chunks,
                embedding_model=embed_model,
                persist_directory=persist,
            )

    bot = ChatBot(model, tokenizer, vectorstore, config_path=cfg_path)

    def respond_fn(message: str, history: list, session: StickySession):
        if not message or not str(message).strip():
            return history, session
        reply = bot.respond(str(message).strip(), session=session, history=history or [])
        history = list(history or [])
        history.append((message, reply))
        return history, session

    def reset_fn():
        return [], StickySession(enable_sticky=sticky_default)

    with gr.Blocks() as demo:
        session_state = gr.State(lambda: StickySession(enable_sticky=sticky_default))
        gr.Markdown(DISCLAIMER)
        gr.Markdown(
            "# Mental Health Psychoeducation Bot\n\n"
            "Socratic psychoeducation with optional RAG over your curated corpus. "
            "Not therapy or diagnosis."
        )
        chatbot = gr.Chatbot(height=420, label="Conversation")
        with gr.Row():
            msg = gr.Textbox(
                label="Your message",
                placeholder="Share what feels okay to explore…",
                scale=4,
            )
            submit = gr.Button("Send", variant="primary")
            new_session_btn = gr.Button("New Session")

        gr.Examples(
            examples=[
                "What does 'grounding' mean in anxiety psychoeducation?",
                "I've felt lonely lately — can we explore that gently?",
            ],
            inputs=msg,
        )

        submit.click(respond_fn, [msg, chatbot, session_state], [chatbot, session_state]).then(
            lambda: "", outputs=msg
        )
        msg.submit(respond_fn, [msg, chatbot, session_state], [chatbot, session_state]).then(
            lambda: "", outputs=msg
        )
        new_session_btn.click(reset_fn, None, [chatbot, session_state])

    demo.launch()


if __name__ == "__main__":
    main()
