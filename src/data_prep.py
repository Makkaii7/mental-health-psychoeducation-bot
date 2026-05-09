"""
CounselChat loading, psychoeducation-oriented filtering, ChatML formatting, and splits.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split


# Default Hugging Face dataset id (override via env or argument if your mirror differs).
DEFAULT_COUNSELCHAT_ID = "nbroad/CounselChat"

# Heuristic patterns suggesting deep therapy / prescriptive clinical tone (exclude from SFT).
_THERAPY_EXCLUSION_PATTERNS = [
    r"\btransference\b",
    r"\bcounter-?transference\b",
    r"\bEMDR\b",
    r"\bCBT\s+homework\b",
    r"\binterpretation\b.*\bunconscious\b",
    r"\bdiagnos(e|is)\b",
    r"\bICD-?\d+\b",
    r"\bDSM-?\d+\b",
    r"\bprescrib(e|ing)\b",
    r"\bmedication\b.*\b(start|stop|dose|taper)\b",
    r"\byou should\b.*\b(leave|divorce|confront|quit)\b",
]

_THERAPY_EXCLUSION_RE = re.compile(
    "|".join(f"(?:{p})" for p in _THERAPY_EXCLUSION_PATTERNS),
    re.IGNORECASE,
)

# Prefer shorter, educational Q&A over long monologue-style therapist replies.
_MAX_RESPONSE_CHARS = 2200
_MIN_RESPONSE_CHARS = 40


def load_counselchat(
    dataset_id: str = DEFAULT_COUNSELCHAT_ID,
    split: str | None = None,
) -> pd.DataFrame:
    """
    Load CounselChat from Hugging Face and normalize to columns: question, answer (str).
    """
    ds = load_dataset(dataset_id) if split is None else load_dataset(dataset_id, split=split)

    if isinstance(ds, DatasetDict):
        # Prefer train if present; else first split.
        if "train" in ds:
            frame = ds["train"].to_pandas()
        else:
            first_key = next(iter(ds.keys()))
            frame = ds[first_key].to_pandas()
    else:
        frame = ds.to_pandas()

    col_map = {c.lower(): c for c in frame.columns}
    q_col = col_map.get("question") or col_map.get("user") or col_map.get("input")
    a_col = col_map.get("answer") or col_map.get("response") or col_map.get("output")
    if not q_col or not a_col:
        raise ValueError(
            f"Could not infer question/answer columns from: {list(frame.columns)}. "
            "Adjust load_counselchat() for your dataset schema."
        )

    out = pd.DataFrame(
        {
            "question": frame[q_col].astype(str).str.strip(),
            "answer": frame[a_col].astype(str).str.strip(),
        }
    )
    out = out[(out["question"].str.len() > 0) & (out["answer"].str.len() > 0)]
    return out.reset_index(drop=True)


def filter_psychoeducation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep rows that look like psychoeducation-style Q&A; drop deep therapy / prescriptive rows.
    """
    def _keep_row(answer: str, question: str) -> bool:
        text = f"{question}\n{answer}"
        if _THERAPY_EXCLUSION_RE.search(text):
            return False
        if len(answer) > _MAX_RESPONSE_CHARS or len(answer) < _MIN_RESPONSE_CHARS:
            return False
        # Penalize very directive imperatives in therapist voice.
        if answer.lower().count("you must ") + answer.lower().count("you need to ") > 2:
            return False
        return True

    mask = df.apply(lambda r: _keep_row(r["answer"], r["question"]), axis=1)
    return df.loc[mask].reset_index(drop=True)


def _default_system_prompt() -> str:
    root = Path(__file__).resolve().parents[1]
    p = root / "prompts" / "system_prompt.txt"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return (
        "You are a psychoeducation thinking partner, not a therapist. "
        "Use Socratic questions; never diagnose or prescribe."
    )


def format_chatml(
    df: pd.DataFrame,
    system_prompt: str | None = None,
    text_column: str = "text",
) -> pd.DataFrame:
    """
    Add a `text` column in ChatML-style multi-turn format for SFT (system + user + assistant).
    """
    sys = system_prompt or _default_system_prompt()
    rows: list[str] = []
    for _, r in df.iterrows():
        user = r["question"].replace("<|im_start|>", "").replace("<|im_end|>", "")
        assistant = r["answer"].replace("<|im_start|>", "").replace("<|im_end|>", "")
        chatml = (
            f"<|im_start|>system\n{sys}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant}<|im_end|>"
        )
        rows.append(chatml)
    out = df.copy()
    out[text_column] = rows
    return out


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratify-free random split 80/10/10 (adjust if you add labels for stratification)."""
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    train_df, temp_df = train_test_split(
        df, test_size=(1.0 - train_ratio), random_state=seed
    )
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(temp_df, test_size=(1.0 - val_size), random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: str | Path = "data/processed",
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)


def dataframe_to_hf_dataset(df: pd.DataFrame) -> Dataset:
    """Convert pandas DataFrame (with `text` column) to HF Dataset for TRL."""
    return Dataset.from_pandas(df, preserve_index=False)
