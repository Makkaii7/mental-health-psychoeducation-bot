"""One-off exploration for CounselChat (run from project root)."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset

from src.data_prep import (
    dataframe_to_hf_dataset,
    filter_psychoeducation,
    format_chatml,
    load_counselchat,
    save_splits,
    split_dataset,
)

DATASET_ID = os.environ.get("COUNSELCHAT_ID", "nbertagnolli/counsel-chat")


def main() -> None:
    print("=" * 70)
    print("STEP 1: Load dataset")
    print("=" * 70)
    ds = load_dataset(DATASET_ID)
    print(ds)
    train = ds["train"]
    n_rows = len(train)
    print(f"\nRows in train: {n_rows}")
    print(f"Columns (features): {train.column_names}")
    print("\n--- First row (raw) ---")
    print(train[0])
    print("\n--- Example rows 1, 100, 500 (truncated) ---")
    for idx in [1, min(100, n_rows - 1), min(500, n_rows - 1)]:
        r = train[idx]
        qt = (r.get("questionText") or r.get("question") or "")[:400]
        at = (r.get("answerText") or r.get("answer") or "")[:400]
        print(f"\n[Row {idx}] topic={r.get('topic', 'N/A')}")
        print("Q:", qt.replace("\n", " ") + ("..." if len(str(r.get('questionText',''))) > 400 else ""))
        print("A:", at.replace("\n", " ") + ("..." if len(str(r.get('answerText',''))) > 400 else ""))

    print("\n" + "=" * 70)
    print("STEP 2: Explore (via normalized dataframe)")
    print("=" * 70)
    df = load_counselchat(DATASET_ID)
    print(f"Normalized rows: {len(df)}")
    n_unique_q = df["question"].nunique()
    print(f"Unique questions (exact string match on normalized 'question' column): {n_unique_q}")
    per_q = df.groupby("question").size()
    print(f"Answers per question - mean: {per_q.mean():.3f}, median: {per_q.median():.1f}, max: {per_q.max()}")
    lens = df["answer"].str.len()
    print(
        f"Answer length (chars) - min: {lens.min()}, max: {lens.max()}, "
        f"mean: {lens.mean():.1f}, median: {lens.median():.1f}"
    )
    qs = np.quantile(lens, [0.1, 0.25, 0.5, 0.75, 0.9])
    print(
        "Quantiles p10/p25/p50/p75/p90:",
        {k: float(v) for k, v in zip(["p10", "p25", "p50", "p75", "p90"], qs)},
    )
    # topic from original if we merged — reload raw for topic
    raw = train.to_pandas()
    if "topic" in raw.columns:
        tc = raw["topic"].fillna("(missing)").astype(str)
        print(f"\nTopic column 'topic' - {tc.nunique()} unique values")
        print(tc.value_counts().head(15))
    else:
        print("\nNo 'topic' column.")

    # Heuristic picks for display (not filter)
    def score_reflective(row):
        a = row["answer"].lower()
        q = row["question"].lower()
        s = 0
        for w in ["consider", "reflect", "explore", "understand", "feel", "sense", "often", "sometimes"]:
            if w in a:
                s += 1
        if "?" in a:
            s += 2
        if len(row["answer"]) < 1200:
            s += 1
        if "diagnos" in a or "prescrib" in a or "you must" in a or "emdr" in a:
            s -= 10
        return s

    def score_therapy(row):
        a = row["question"] + "\n" + row["answer"]
        from src import data_prep as dp

        if dp._THERAPY_EXCLUSION_RE.search(a):
            return 100
        s = 0
        al = row["answer"].lower()
        for w in ["you must", "you need to", "i recommend you", "take medication", "diagnosis", "dsm"]:
            if w in al:
                s += 3
        if len(row["answer"]) > 1800:
            s += 2
        return s

    df["_refl"] = df.apply(score_reflective, axis=1)
    df["_ther"] = df.apply(score_therapy, axis=1)
    good = df.sort_values("_refl", ascending=False).head(15)
    bad = df.sort_values("_ther", ascending=False).head(15)

    print("\n--- 5 examples: good psychoeducation-ish (heuristic pick) ---")
    for i, (_, r) in enumerate(good.head(5).iterrows(), 1):
        print(f"\n[{i}] len={len(r['answer'])}")
        print("Q:", r["question"][:350].replace("\n", " "))
        print("A:", r["answer"][:500].replace("\n", " ") + ("..." if len(r["answer"]) > 500 else ""))

    print("\n--- 5 examples: deep therapy-ish (heuristic / regex flags) ---")
    for i, (_, r) in enumerate(bad.head(5).iterrows(), 1):
        print(f"\n[{i}] len={len(r['answer'])}")
        print("Q:", r["question"][:350].replace("\n", " "))
        print("A:", r["answer"][:500].replace("\n", " ") + ("..." if len(r["answer"]) > 500 else ""))

    print("\n" + "=" * 70)
    print("STEP 3: filter_psychoeducation (src.data_prep)")
    print("=" * 70)
    df_clean = df.drop(columns=["_refl", "_ther"], errors="ignore").copy()
    before = len(df_clean)
    kept_df = filter_psychoeducation(df_clean.copy())
    after = len(kept_df)
    kc = set(zip(kept_df["question"], kept_df["answer"]))
    removed_df = df_clean[~df_clean.apply(lambda r: (r["question"], r["answer"]) in kc, axis=1)].reset_index(
        drop=True
    )
    pct_removed = 100 * (before - after) / before if before else 0
    print(f"Rows before: {before}")
    print(f"Rows after:  {after}")
    print(f"Removed: {before - after} ({pct_removed:.1f}%)")
    if pct_removed > 60:
        print("FLAG: filter removes MORE than 60% - likely too aggressive for this corpus.")
    elif pct_removed < 10:
        print("FLAG: filter removes LESS than 10% - likely too lenient.")
    else:
        print("Removal rate within 10–60% band (no flag).")

    print("\n--- 5 KEPT (random sample) ---")
    sample_k = kept_df.sample(min(5, len(kept_df)), random_state=42) if len(kept_df) else kept_df
    for i, (_, r) in enumerate(sample_k.iterrows(), 1):
        print(f"\n[{i}] len={len(r['answer'])}")
        print("Q:", r["question"][:300].replace("\n", " "))
        print("A:", r["answer"][:400].replace("\n", " ") + ("..." if len(r["answer"]) > 400 else ""))

    print("\n--- 5 REMOVED (random sample) ---")
    sample_r = removed_df.sample(min(5, len(removed_df)), random_state=43) if len(removed_df) else removed_df
    for i, (_, r) in enumerate(sample_r.iterrows(), 1):
        print(f"\n[{i}] len={len(r['answer'])}")
        print("Q:", r["question"][:300].replace("\n", " "))
        print("A:", r["answer"][:400].replace("\n", " ") + ("..." if len(r["answer"]) > 400 else ""))

    print("\n" + "=" * 70)
    print("STEP 4: format_chatml -> split -> save")
    print("=" * 70)
    cfg_path = ROOT / "config" / "config.yaml"
    fmt = format_chatml(kept_df.copy(), config_path=cfg_path)
    train_df, val_df, test_df = split_dataset(fmt)
    save_splits(train_df, val_df, test_df, ROOT / "data" / "processed")
    dataframe_to_hf_dataset(train_df).save_to_disk(str(ROOT / "data" / "processed" / "train_hf"))
    dataframe_to_hf_dataset(val_df).save_to_disk(str(ROOT / "data" / "processed" / "val_hf"))
    dataframe_to_hf_dataset(test_df).save_to_disk(str(ROOT / "data" / "processed" / "test_hf"))
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print("\n--- ChatML example 1 (full `text` field) ---")
    print(train_df.iloc[0]["text"])
    print("\n--- ChatML example 2 (full `text` field) ---")
    print(train_df.iloc[1]["text"])


if __name__ == "__main__":
    main()
