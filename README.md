# Mental-Health Psychoeducation Bot (CODS 641 Final Project)

Hybrid **QLoRA + RAG** chatbot for mental health psychoeducation: **Qwen3-8B** fine-tuned on filtered **CounselChat**, with **RAG** over curated psychoeducation documents and a **four-tier safety** layer (keyword crisis routing, out-of-scope redirect, “with care” tone, sticky crisis session).

## Overview

- **Fine-tuning:** Filter CounselChat for psychoeducation-appropriate Q&A, format to ChatML-style strings, train LoRA adapters with Unsloth + TRL `SFTTrainer` (4-bit NF4).
- **RAG:** Chunk corpus → embed with `sentence-transformers/all-MiniLM-L6-v2` → ChromaDB; retrieve top-`k` chunks into the prompt.
- **Safety:** Tier 1 (in scope) → Tier 2 (in scope, grounded) → Tier 3 (redirect) → Tier 4 (crisis resources + sticky session).
- **UI:** Gradio `ChatInterface` (`python -m src.app`).

## Architecture

```text
User message
    → Safety (tiers + sticky crisis)
    → [if in scope] RAG retrieve (Chroma)
    → LLM (Qwen3-8B + LoRA, 4-bit) with system prompt + context
    → Response
```

Key modules: `src/data_prep.py`, `src/fine_tune.py`, `src/rag_pipeline.py`, `src/safety.py`, `src/chatbot.py`, `src/evaluate.py`, `src/app.py`. Configuration: `config/config.yaml`. Prompts: `prompts/`.

## Setup

1. Python 3.10+ recommended; **CUDA** strongly recommended for training and inference.
2. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

3. **Unsloth** / **bitsandbytes** builds are platform-specific; if install fails on Windows, follow Unsloth’s current install notes or run training on Linux/WSL with a supported GPU stack.

4. Prepare data (from project root):

```python
from pathlib import Path
from datasets import Dataset
from src.data_prep import (
    filter_psychoeducation,
    format_chatml,
    load_counselchat,
    save_splits,
    split_dataset,
    dataframe_to_hf_dataset,
)

df = load_counselchat()
df = filter_psychoeducation(df)
df = format_chatml(df)
train_df, val_df, test_df = split_dataset(df)
save_splits(train_df, val_df, test_df, Path("data/processed"))
dataframe_to_hf_dataset(train_df).save_to_disk("data/processed/train_hf")
```

5. **RAG corpus:** Add `.txt` / `.md` psychoeducation files under `data/rag_corpus/` (see report for curation policy). The demo app will build `chroma_db/` on first launch if missing.

6. **Train** (after `train_hf` exists):

```bash
python src/fine_tune.py --dataset data/processed/train_hf --output checkpoints/lora_adapter
```

## Usage

- **Gradio demo** (builds placeholder Chroma index if the DB is empty):

```bash
python -m src.app
```

- **Programmatic:** construct `ChatBot` in `src/chatbot.py` after `load_model()` and vector store creation (see `src/app.py`).

## Evaluation

- **RAG:** `src/evaluate.py` → `evaluate_rag()` (Ragas: faithfulness, answer relevancy, context precision; may require additional Ragas LLM provider configuration per upstream docs).
- **Safety:** `evaluate_safety()` with `RED_TEAM_CASES`.
- **Efficiency:** `measure_latency()`, `measure_memory()`.

## References

- CounselChat / mental-health dialogue data (Hugging Face; verify dataset card and license before use).
- Qwen3 model card (`Qwen/Qwen3-8B`) and license terms.
- Unsloth, PEFT, TRL, LangChain, ChromaDB, Ragas, Gradio documentation.

---

**Course:** CODS 641 (NLP & IR), Khalifa University — final project, due **Monday May 11, 2026**.
