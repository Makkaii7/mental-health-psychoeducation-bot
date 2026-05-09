"""
CounselChat loading, deduplication, psychoeducation-oriented filtering, ChatML formatting,
and question-level splits (no train/test question leakage).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split


# Default Hugging Face dataset id (override via env or argument if your mirror differs).
DEFAULT_COUNSELCHAT_ID = "nbroad/CounselChat"

# Legacy / overlap with new rules (kept for extra clinical-signal drops on Q+A combined text).
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

_MIN_ANSWER_CHARS = 100
_MAX_ANSWER_CHARS = 2000

_DIRECTIVE_PHRASES = (
    "you should",
    "you must",
    "you need to",
    "you have to",
    "i recommend",
    "i suggest",
    "i would advise",
)

_TOPIC_CLINICAL_RE = re.compile(
    r"(diagnos|medication|substance|addiction)",
    re.IGNORECASE,
)

_BONUS_TERMS = (
    "?",
    "think about",
    "consider",
    "reflect",
    "notice",
    "what do you",
    "how does",
    "many people",
    "it's common",
    "its common",
    "research shows",
    "one way to think about",
)

_PENALTY_TERMS = (
    "i diagnose",
    "your condition",
    "your disorder",
    "take medication",
    "prescribe",
    "in my clinical experience",
    "as your therapist",
)


def _directive_hit_count(answer: str) -> int:
    al = answer.lower()
    return sum(al.count(p) for p in _DIRECTIVE_PHRASES)


def _psychoeducation_fit_score(answer: str) -> int:
    al = answer.lower()
    score = 0
    for t in _BONUS_TERMS:
        if t in al:
            score += 1
    for t in _PENALTY_TERMS:
        if t in al:
            score -= 2
    return score


def _purely_informational_answer(answer: str) -> bool:
    """Loose heuristic: long-ish neutral/educational tone without strong prescription."""
    al = answer.lower()
    if len(answer) < 250:
        return False
    if _directive_hit_count(answer) >= 2:
        return False
    if any(p in al for p in ("research", "studies show", "according to", "psychoeducation", "coping skill")):
        return True
    return _psychoeducation_fit_score(answer) >= 4


def _topic_clinical_drop(topic: str, answer: str) -> bool:
    """
    Return True if this row should be REMOVED due to clinical topic slug,
    unless answer looks purely informational.
    """
    t = (topic or "").lower()
    if not _TOPIC_CLINICAL_RE.search(t):
        return False
    if _purely_informational_answer(answer):
        return False
    return True


def _row_passes_filter(
    answer: str,
    question: str,
    topic: str,
    *,
    require_fit_score: bool,
) -> bool:
    qa = f"{question}\n{answer}"
    if _THERAPY_EXCLUSION_RE.search(qa):
        return False
    if len(answer) < _MIN_ANSWER_CHARS or len(answer) > _MAX_ANSWER_CHARS:
        return False
    if _directive_hit_count(answer) >= 3:
        return False
    if _topic_clinical_drop(topic, answer):
        return False
    if require_fit_score and _psychoeducation_fit_score(answer) < 0:
        return False
    return True


def load_counselchat(
    dataset_id: str = DEFAULT_COUNSELCHAT_ID,
    split: str | None = None,
) -> pd.DataFrame:
    """
    Load CounselChat from Hugging Face.

    Normalized columns: ``question``, ``answer``, ``upvotes`` (int), ``topic`` (str).
    Drops rows with empty question or answer after strip.
    """
    ds = load_dataset(dataset_id) if split is None else load_dataset(dataset_id, split=split)

    if isinstance(ds, DatasetDict):
        if "train" in ds:
            frame = ds["train"].to_pandas()
        else:
            first_key = next(iter(ds.keys()))
            frame = ds[first_key].to_pandas()
    else:
        frame = ds.to_pandas()

    col_map = {c.lower(): c for c in frame.columns}
    q_col = (
        col_map.get("question")
        or col_map.get("user")
        or col_map.get("input")
        or col_map.get("questiontext")
    )
    a_col = (
        col_map.get("answer")
        or col_map.get("response")
        or col_map.get("output")
        or col_map.get("answertext")
    )
    if not q_col or not a_col:
        raise ValueError(
            f"Could not infer question/answer columns from: {list(frame.columns)}. "
            "Adjust load_counselchat() for your dataset schema."
        )

    up_col = col_map.get("upvotes")
    topic_col = col_map.get("topic")

    out = pd.DataFrame(
        {
            "question": frame[q_col].astype(str).str.strip(),
            "answer": frame[a_col].astype(str).str.strip(),
        }
    )
    if up_col:
        out["upvotes"] = pd.to_numeric(frame[up_col], errors="coerce").fillna(-1).astype(int)
    else:
        out["upvotes"] = -1
    if topic_col:
        out["topic"] = frame[topic_col].fillna("").astype(str).str.strip()
    else:
        out["topic"] = ""

    out = out[(out["question"].str.len() > 0) & (out["answer"].str.len() > 0)]
    return out.reset_index(drop=True)


def deduplicate_best_answer_per_question(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per ``question``: highest ``upvotes``; ties / missing upvotes → longest answer
    among rows that pass the structural filter **without** the psychoeducation fit score (tie-break).
    """
    if df.empty:
        return df

    def _pick_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.reset_index(drop=True)
        u = g["upvotes"].astype(int)
        max_u = int(u.max())
        if max_u < 0:
            candidates = g
        else:
            candidates = g.loc[u == max_u].copy()
        if len(candidates) == 1:
            return candidates
        weak_ok = candidates.apply(
            lambda r: _row_passes_filter(
                r["answer"],
                r["question"],
                str(r["topic"]) if "topic" in r.index else "",
                require_fit_score=False,
            ),
            axis=1,
        )
        passed = candidates.loc[weak_ok]
        if len(passed):
            idx = passed["answer"].str.len().idxmax()
            return passed.loc[[idx]]
        idx = candidates["answer"].str.len().idxmax()
        return candidates.loc[[idx]]

    parts = [_pick_group(g) for _, g in df.groupby("question", sort=False)]
    return pd.concat(parts, ignore_index=True)


def filter_psychoeducation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strong psychoeducation-oriented filter (length, directives, topic, fit score).
    """
    mask = df.apply(
        lambda r: _row_passes_filter(
            r["answer"],
            r["question"],
            str(r["topic"]) if "topic" in r.index else "",
            require_fit_score=True,
        ),
        axis=1,
    )
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
    tokenizer: Any | None = None,
    model_name: str | None = None,
    system_prompt: str | None = None,
    text_column: str = "text",
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Add a ``text`` column using the model tokenizer's ``apply_chat_template`` (matches inference).
    """
    from transformers import AutoTokenizer

    root = Path(__file__).resolve().parents[1]
    cfg_file = Path(config_path) if config_path else root / "config" / "config.yaml"
    if tokenizer is None:
        if model_name is None and cfg_file.is_file():
            import yaml

            cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
            model_name = (cfg.get("model") or {}).get("name", "Qwen/Qwen3-8B")
        if model_name is None:
            model_name = "Qwen/Qwen3-8B"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    sys = system_prompt or _default_system_prompt()
    rows: list[str] = []
    for _, r in df.iterrows():
        user = str(r["question"]).strip()
        assistant = str(r["answer"]).strip()
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        rows.append(text)
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
    """
    Split by **unique question text** (80/10/10): all rows for a question live in one split only.
    """
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    questions = pd.Series(df["question"].unique())
    q_train, q_temp = train_test_split(
        questions,
        test_size=(1.0 - train_ratio),
        random_state=seed,
    )
    val_size = val_ratio / (val_ratio + test_ratio)
    q_val, q_test = train_test_split(
        q_temp,
        test_size=(1.0 - val_size),
        random_state=seed,
    )
    train_set = set(q_train)
    val_set = set(q_val)
    test_set = set(q_test)
    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set
    if overlap_tv or overlap_tt or overlap_vt:
        raise RuntimeError(f"Question split overlap: {overlap_tv | overlap_tt | overlap_vt}")

    def _assign(q: str) -> str:
        if q in train_set:
            return "train"
        if q in val_set:
            return "val"
        if q in test_set:
            return "test"
        raise KeyError(q)

    split_col = df["question"].map(_assign)
    train_df = df.loc[split_col == "train"].reset_index(drop=True)
    val_df = df.loc[split_col == "val"].reset_index(drop=True)
    test_df = df.loc[split_col == "test"].reset_index(drop=True)
    return train_df, val_df, test_df


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


def run_full_preprocessing_pipeline(
    dataset_id: str = DEFAULT_COUNSELCHAT_ID,
    out_dir: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    load → dedup → filter → format_chatml → question-level split → save CSV + HF datasets.
    Returns a dict of counts and sample rows for logging.
    """
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(out_dir) if out_dir else root / "data" / "processed"
    cfg_path = Path(config_path) if config_path else root / "config" / "config.yaml"

    raw = load_counselchat(dataset_id)
    n_after_load = len(raw)
    deduped = deduplicate_best_answer_per_question(raw)
    n_after_dedup = len(deduped)
    filtered = filter_psychoeducation(deduped)
    n_after_filter = len(filtered)

    fmt = format_chatml(filtered.copy(), config_path=cfg_path)
    train_df, val_df, test_df = split_dataset(fmt)
    save_splits(train_df, val_df, test_df, out_dir)

    def _for_hf(d: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ("question", "answer", "text", "topic", "upvotes") if c in d.columns]
        return d[cols].copy()

    dataframe_to_hf_dataset(_for_hf(train_df)).save_to_disk(str(out_dir / "train_hf"))
    dataframe_to_hf_dataset(_for_hf(val_df)).save_to_disk(str(out_dir / "val_hf"))
    dataframe_to_hf_dataset(_for_hf(test_df)).save_to_disk(str(out_dir / "test_hf"))

    # leakage check: question sets disjoint
    qt = set(train_df["question"])
    qv = set(val_df["question"])
    qe = set(test_df["question"])
    leakage = (qt & qv) | (qt & qe) | (qv & qe)

    kf = set(zip(filtered["question"], filtered["answer"]))
    filter_removed = deduped[~deduped.apply(lambda r: (r["question"], r["answer"]) in kf, axis=1)]
    n_removed = len(filter_removed)
    removed_sample = filter_removed.sample(n=min(3, n_removed), random_state=42) if n_removed else pd.DataFrame()
    kept_sample = filtered.sample(n=min(3, len(filtered)), random_state=41) if len(filtered) else pd.DataFrame()

    def _short_rows(frame: pd.DataFrame) -> list[dict]:
        out = []
        for _, r in frame.iterrows():
            out.append(
                {
                    "topic": (str(r["topic"]) if "topic" in r.index else ""),
                    "upvotes": int(r["upvotes"]) if "upvotes" in r.index else -1,
                    "question": str(r["question"])[:220] + ("..." if len(str(r["question"])) > 220 else ""),
                    "answer": str(r["answer"])[:280] + ("..." if len(str(r["answer"])) > 280 else ""),
                }
            )
        return out

    return {
        "n_after_load": n_after_load,
        "n_after_dedup": n_after_dedup,
        "n_after_filter": n_after_filter,
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "leakage_questions": list(leakage),
        "kept_examples_preview": _short_rows(kept_sample),
        "removed_examples_preview": _short_rows(removed_sample),
        "dedup_removed_count": n_after_load - n_after_dedup,
        "filter_removed_count": n_after_dedup - n_after_filter,
    }
