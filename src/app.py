"""
Gradio web UI for the MindBridge psychoeducation chatbot.
Run from project root: ``python -m src.app``

Per-user ``StickySession`` lives in ``gr.State`` so multiple browsers don't share crisis state.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import gradio as gr
import yaml

from src.chatbot import ChatBot, load_model
from src.rag_pipeline import chunk_documents, create_vectorstore, load_corpus, load_vectorstore
from src.safety import StickySession


HEADER_HTML = """
<div class="mb-header">
  <div class="mb-brand">
    <span class="mb-brand-mark">🧠</span>
    <div class="mb-brand-text">
      <div class="mb-brand-title">MindBridge</div>
      <div class="mb-brand-subtitle">Your Psychoeducation Thinking Partner</div>
    </div>
  </div>
  <div class="mb-tagline">
    Explore mental well-being through guided reflection — not therapy, not diagnosis,
    just thoughtful conversation.
  </div>
</div>
"""

DISCLAIMER_HTML = """
<div class="mb-disclaimer">
  <span class="mb-disclaimer-icon">ℹ️</span>
  <div class="mb-disclaimer-text">
    <strong>This is a research prototype for psychoeducation-style reflection.</strong>
    It is not medical advice, therapy, or crisis monitoring. If you may be in danger, contact
    <strong>Estijaba&nbsp;800&nbsp;1717</strong>, <strong>HOPE&nbsp;Line&nbsp;800&nbsp;4673</strong>,
    or <strong>Emergency&nbsp;999</strong>.
  </div>
</div>
"""

FOOTER_HTML = """
<div class="mb-footer">
  <span class="mb-footer-brand">MindBridge</span>
  <span class="mb-footer-sep">·</span>
  CODS 641 Final Project
  <span class="mb-footer-sep">·</span>
  Khalifa University
  <span class="mb-footer-sep">·</span>
  Powered by Qwen3-8B + RAG
</div>
"""

APP_CSS = """
/* ───── page shell ─────────────────────────────────────────── */
.gradio-container, body {
    background-color: #f8f9fa !important;
    color: #0d2847 !important;
    font-family: 'Source Sans 3', 'Inter', ui-sans-serif, system-ui, sans-serif !important;
}
#mb-root {
    max-width: 820px !important;
    margin: 0 auto !important;
    padding: 1.5rem 1.25rem 2.5rem !important;
}

/* ───── header ─────────────────────────────────────────────── */
.mb-header { margin: 0 0 1.25rem 0; }
.mb-brand {
    display: flex;
    align-items: center;
    gap: 0.85rem;
}
.mb-brand-mark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 3rem; height: 3rem;
    border-radius: 14px;
    background: linear-gradient(145deg, #6fb5ec 0%, #4a90d9 100%);
    box-shadow: 0 4px 12px rgba(74, 144, 217, 0.25);
    font-size: 1.5rem;
    flex-shrink: 0;
}
.mb-brand-text { display: flex; flex-direction: column; line-height: 1.15; }
.mb-brand-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0d2847;
    letter-spacing: -0.025em;
}
.mb-brand-subtitle {
    font-size: 0.95rem;
    font-weight: 500;
    color: #4a90d9;
    margin-top: 0.1rem;
}
.mb-tagline {
    margin: 0.75rem 0 0.25rem 0;
    color: #3d5a73;
    font-size: 0.98rem;
    line-height: 1.5;
}

/* ───── disclaimer banner ──────────────────────────────────── */
.mb-disclaimer {
    display: flex;
    gap: 0.7rem;
    align-items: flex-start;
    background: linear-gradient(135deg, #eaf4fc 0%, #dbeafe 100%);
    border: 1px solid #c5dceb;
    border-left: 4px solid #4a90d9;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin: 1rem 0 1.25rem 0;
    color: #1f3a5f;
    font-size: 0.92rem;
    line-height: 1.5;
}
.mb-disclaimer-icon { font-size: 1.05rem; flex-shrink: 0; padding-top: 1px; }
.mb-disclaimer-text strong { color: #0d2847; font-weight: 600; }

/* ───── chat panel ─────────────────────────────────────────── */
#mb-chatbot {
    min-height: 550px !important;
    background: #ffffff !important;
    border: 1px solid #e2eaf2 !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 12px rgba(13, 40, 71, 0.04);
}
#mb-chatbot .message,
#mb-chatbot [data-testid="bot"],
#mb-chatbot [data-testid="user"] {
    border-radius: 12px !important;
    padding: 0.85rem 1rem !important;
    line-height: 1.55 !important;
    font-size: 0.97rem !important;
    border: 1px solid transparent !important;
}
/* user bubbles — light blue */
#mb-chatbot .user,
#mb-chatbot [data-testid="user"],
#mb-chatbot .message-row.user-row .message,
#mb-chatbot .role-user {
    background: #e3f2fd !important;
    color: #0d2847 !important;
    border-color: #d0e6f7 !important;
}
/* bot bubbles — white with soft border */
#mb-chatbot .bot,
#mb-chatbot [data-testid="bot"],
#mb-chatbot .message-row.bot-row .message,
#mb-chatbot .role-assistant {
    background: #ffffff !important;
    color: #0d2847 !important;
    border-color: #e2eaf2 !important;
    box-shadow: 0 1px 2px rgba(13, 40, 71, 0.03);
}

#mb-typing {
    min-height: 1.4rem;
    color: #4a90d9;
    font-size: 0.88rem;
    font-style: italic;
    margin: 0.5rem 0 0.25rem 0.25rem;
}

/* ───── input area ─────────────────────────────────────────── */
#mb-input textarea {
    border: 1px solid #d6e1ec !important;
    border-radius: 10px !important;
    background: #ffffff !important;
    font-size: 0.97rem !important;
    padding: 0.75rem 0.9rem !important;
    transition: border-color 0.15s ease;
}
#mb-input textarea:focus {
    border-color: #4a90d9 !important;
    box-shadow: 0 0 0 3px rgba(74, 144, 217, 0.12) !important;
    outline: none !important;
}
#mb-send-btn button {
    background: linear-gradient(145deg, #6fb5ec, #4a90d9) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 0.7rem 1.1rem !important;
    box-shadow: 0 2px 6px rgba(74, 144, 217, 0.25);
    transition: transform 0.05s ease, box-shadow 0.15s ease;
}
#mb-send-btn button:hover {
    box-shadow: 0 4px 10px rgba(74, 144, 217, 0.32);
    transform: translateY(-1px);
}
#mb-reset-btn button {
    background: #ffffff !important;
    border: 1.5px solid #d6e1ec !important;
    color: #4a90d9 !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.1rem !important;
}
#mb-reset-btn button:hover { background: #f1f7fd !important; border-color: #4a90d9 !important; }

/* ───── examples block ─────────────────────────────────────── */
.mb-examples-label {
    margin: 1.25rem 0 0.5rem 0.25rem;
    color: #3d5a73;
    font-size: 0.88rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
#mb-examples .gr-sample-textbox, #mb-examples button {
    background: #ffffff !important;
    border: 1px solid #d6e1ec !important;
    border-radius: 999px !important;
    padding: 0.45rem 0.95rem !important;
    color: #1f3a5f !important;
    font-size: 0.9rem !important;
    transition: all 0.15s ease;
}
#mb-examples .gr-sample-textbox:hover, #mb-examples button:hover {
    border-color: #4a90d9 !important;
    background: #f1f7fd !important;
    color: #0d2847 !important;
}

/* ───── footer ─────────────────────────────────────────────── */
.mb-footer {
    text-align: center;
    color: #6b8299;
    font-size: 0.82rem;
    margin-top: 2rem;
    padding-top: 1.25rem;
    border-top: 1px solid #e2eaf2;
}
.mb-footer-brand { color: #4a90d9; font-weight: 600; }
.mb-footer-sep { margin: 0 0.4rem; color: #b8c5d3; }
footer { display: none !important; }
"""

EXAMPLE_PROMPTS = [
    "What is anxiety and how does it affect daily life?",
    "I've been feeling stressed lately — can we explore that?",
    "What's the difference between sadness and depression?",
    "How can I build better mental health habits?",
]


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
        yield history, session, "MindBridge is composing a reply…"
        reply = bot.respond(str(message).strip(), session=session, history=history)
        history.append({"role": "user", "content": str(message).strip()})
        history.append({"role": "assistant", "content": reply})
        yield history, session, ""

    def reset_fn() -> tuple[list, StickySession, str]:
        return [], StickySession(enable_sticky=sticky_default), ""

    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.Color(
            c50="#eaf4fc", c100="#d0e6f7", c200="#b1d4ef", c300="#8cc0e5",
            c400="#6fb5ec", c500="#4a90d9", c600="#3a7bc0", c700="#2c629c",
            c800="#1f4a78", c900="#143356", c950="#0d2847",
        ),
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Source Sans 3"), "ui-sans-serif", "system-ui", "sans-serif"],
    )

    with gr.Blocks(title="MindBridge — Psychoeducation Bot") as demo:
        session_state = gr.State(lambda: StickySession(enable_sticky=sticky_default))

        with gr.Column(elem_id="mb-root"):
            gr.HTML(HEADER_HTML)
            gr.HTML(DISCLAIMER_HTML)

            chatbot = gr.Chatbot(
                height=550,
                label="Conversation",
                elem_id="mb-chatbot",
                show_label=False,
                avatar_images=None,
            )
            typing_status = gr.Markdown("", elem_id="mb-typing")

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="What's on your mind? Share whatever feels comfortable...",
                    scale=5,
                    lines=2,
                    max_lines=6,
                    show_label=False,
                    elem_id="mb-input",
                    container=False,
                )
                with gr.Column(scale=0, min_width=150):
                    submit = gr.Button("Send", variant="primary", elem_id="mb-send-btn")
                    new_session_btn = gr.Button(
                        "New Session", variant="secondary", elem_id="mb-reset-btn"
                    )

            gr.HTML('<div class="mb-examples-label">Try a starter prompt</div>')
            gr.Examples(
                examples=EXAMPLE_PROMPTS,
                inputs=msg,
                elem_id="mb-examples",
                label="",
            )

            gr.HTML(FOOTER_HTML)

        out_targets = [chatbot, session_state, typing_status]
        submit.click(respond_fn, [msg, chatbot, session_state], out_targets).then(
            lambda: "", outputs=msg
        )
        msg.submit(respond_fn, [msg, chatbot, session_state], out_targets).then(
            lambda: "", outputs=msg
        )
        new_session_btn.click(reset_fn, None, out_targets)

    demo.launch(theme=theme, css=APP_CSS)


if __name__ == "__main__":
    main()
