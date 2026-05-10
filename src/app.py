"""
Gradio web UI for the Mental Health Psychoeducation Bot.
Run from project root: ``python -m src.app``

Uses ``gr.State`` so each user has their own ``StickySession`` (not shared on the shared ``ChatBot``).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import gradio as gr
import yaml

from src.chatbot import ChatBot, load_model
from src.rag_pipeline import chunk_documents, create_vectorstore, load_corpus, load_vectorstore
from src.safety import StickySession


DISCLAIMER_MD = (
    "**Disclaimer:** This is a student research prototype for psychoeducation-style reflection only. "
    "It is **not** medical advice, therapy, or crisis monitoring. If you may be in danger, "
    "contact Estijaba **800 1717**, HOPE Line **800 4673**, or Emergency **999**."
)

APP_CSS = """
#mh-root {
    max-width: 800px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding: 0 1rem 2rem !important;
}
#mh-disclaimer {
    background: linear-gradient(135deg, #e8f2fc 0%, #dce9f7 50%, #f5f9fc 100%);
    border: 1px solid #c5d8eb;
    border-radius: 10px;
    padding: 1rem 1.15rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 2px rgba(26, 86, 181, 0.06);
}
.mh-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.35rem;
}
.mh-brand-mark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2.5rem;
    height: 2.5rem;
    border-radius: 10px;
    background: linear-gradient(145deg, #1a56b5, #0d3d82);
    color: #fff;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
    flex-shrink: 0;
}
.mh-brand-text {
    font-size: 1.55rem;
    font-weight: 650;
    color: #0d2847;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.mh-subtitle {
    color: #3d5a73;
    font-size: 0.98rem;
    margin: 0.25rem 0 1rem 0;
    line-height: 1.45;
}
#mh-chatbot {
    min-height: 480px !important;
}
#mh-typing {
    min-height: 1.5rem;
    color: #1a56b5;
    font-size: 0.9rem;
    font-style: italic;
    margin: 0.35rem 0 0.5rem 0;
}
#mh-footer {
    text-align: center;
    color: #5a7289;
    font-size: 0.82rem;
    margin-top: 1.5rem;
    padding-top: 1rem;
    border-top: 1px solid #e2e8f0;
}
footer { display: none !important; }
"""


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

    def respond_fn(
        message: str,
        history: list,
        session: StickySession,
    ) -> Iterator[tuple[list, StickySession, str]]:
        history = list(history or [])
        if not message or not str(message).strip():
            yield history, session, ""
            return
        yield history, session, "*Assistant is composing a reply…*"
        reply = bot.respond(str(message).strip(), session=session, history=history)
        history.append({"role": "user", "content": str(message).strip()})
        history.append({"role": "assistant", "content": reply})
        yield history, session, ""

    def reset_fn() -> tuple[list, StickySession, str]:
        return [], StickySession(enable_sticky=sticky_default), ""

    theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Source Sans 3"), "ui-sans-serif", "system-ui", "sans-serif"],
    )

    with gr.Blocks(theme=theme, css=APP_CSS) as demo:
        session_state = gr.State(lambda: StickySession(enable_sticky=sticky_default))
        with gr.Column(elem_id="mh-root"):
            gr.HTML(
                '<div class="mh-brand">'
                '<span class="mh-brand-mark">MH</span>'
                '<span class="mh-brand-text">Psychoeducation Assistant</span>'
                "</div>"
            )
            gr.Markdown(
                "Grounded in authoritative mental health resources. "
                "Socratic reflection — not therapy or diagnosis.",
                elem_classes=["mh-subtitle"],
            )
            gr.Markdown(DISCLAIMER_MD, elem_id="mh-disclaimer")

            chatbot = gr.Chatbot(
                height=520,
                label="Conversation",
                elem_id="mh-chatbot",
                show_copy_button=True,
            )
            typing_status = gr.Markdown("", elem_id="mh-typing")

            with gr.Row():
                msg = gr.Textbox(
                    label="Your message",
                    placeholder="Share what's on your mind...",
                    scale=5,
                    lines=2,
                    max_lines=6,
                )
                with gr.Column(scale=0, min_width=140):
                    submit = gr.Button("Send", variant="primary", scale=1)
                    new_session_btn = gr.Button("New Session", variant="secondary", scale=1)

            gr.Examples(
                examples=[
                    "What does 'grounding' mean in anxiety psychoeducation?",
                    "I've felt lonely lately — can we explore that gently?",
                ],
                inputs=msg,
            )

            gr.Markdown(
                "CODS 641 Final Project | Khalifa University | Built with Qwen3-8B",
                elem_id="mh-footer",
            )

        out_targets = [chatbot, session_state, typing_status]
        submit.click(respond_fn, [msg, chatbot, session_state], out_targets).then(
            lambda: "", outputs=msg
        )
        msg.submit(respond_fn, [msg, chatbot, session_state], out_targets).then(
            lambda: "", outputs=msg
        )
        new_session_btn.click(reset_fn, None, out_targets)

    demo.launch()


if __name__ == "__main__":
    main()
