"""Generate report/architecture.png — IEEE-style system diagram (matplotlib)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "architecture.png"


def _box(ax, xy, w, h, text, *, fc, ec="black", lw=1.0, tc="white", fontsize=8):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.06",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        mutation_aspect=0.5,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=tc,
        wrap=True,
        linespacing=1.15,
    )


def _arrow(ax, xy_start, xy_end, *, color="0.25", lw=1.2):
    arr = FancyArrowPatch(
        xy_start,
        xy_end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        edgecolor=color,
        facecolor=color,
        shrinkA=2,
        shrinkB=2,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(arr)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "axes.linewidth": 0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )

    navy = "#1c355e"
    crisis_red = "#b2182b"
    ok_green = "#1b7837"
    tier3_blue = "#2166ac"
    txt_light = "white"
    txt_dark = "#111111"

    fig, ax = plt.subplots(figsize=(7.2, 10.2), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # --- Top stack (shared) ---
    _box(ax, (3.1, 12.55), 3.8, 0.62, "User Message", fc=navy, tc=txt_light, fontsize=9)
    _arrow(ax, (5.0, 12.55), (5.0, 12.05))
    _box(
        ax,
        (1.9, 11.15),
        6.2,
        0.78,
        "Safety Classifier\n(regex + keyword phrases, 4 tiers)",
        fc=navy,
        tc=txt_light,
        fontsize=8.5,
    )

    # Fan-out from classifier
    _arrow(ax, (3.3, 11.15), (1.85, 10.35))
    _arrow(ax, (5.0, 11.15), (5.0, 10.35))
    _arrow(ax, (6.7, 11.15), (8.15, 10.35))

    # --- Three columns ---
    cx_l, cx_m, cx_r = 0.35, 4.15, 7.55
    col_w = 2.55

    # Tier 4 (left, red)
    _box(
        ax,
        (cx_l, 9.55),
        col_w,
        0.52,
        "[Tier 4 — Crisis]",
        fc=crisis_red,
        tc=txt_light,
        fontsize=8,
    )
    _arrow(ax, (1.625, 9.55), (1.625, 9.05), color=crisis_red)
    _box(
        ax,
        (cx_l, 8.15),
        col_w,
        1.05,
        "Hardcoded crisis response\n+ UAE resources\n(Estijaba, HOPE, 999)",
        fc=crisis_red,
        tc=txt_light,
        fontsize=7.5,
    )
    _arrow(ax, (1.625, 8.15), (1.625, 7.65), color=crisis_red)
    _box(
        ax,
        (cx_l, 7.0),
        col_w,
        0.48,
        "Sticky crisis mode ON",
        fc=crisis_red,
        tc=txt_light,
        fontsize=8,
    )

    # Tier 3 (center)
    _box(
        ax,
        (cx_m, 9.55),
        col_w,
        0.52,
        "[Tier 3 — Out of scope]",
        fc=tier3_blue,
        tc=txt_light,
        fontsize=8,
    )
    _arrow(ax, (5.425, 9.55), (5.425, 9.05))
    _box(
        ax,
        (cx_m, 8.35),
        col_w,
        0.85,
        "Hardcoded redirect response\n(no diagnosis / no directives)",
        fc=tier3_blue,
        tc=txt_light,
        fontsize=7.5,
    )

    # Tier 1/2 (right, green accent chain)
    _box(
        ax,
        (cx_r, 9.55),
        col_w,
        0.52,
        "[Tier 1 / 2 — In scope]",
        fc=ok_green,
        tc=txt_light,
        fontsize=8,
    )
    _arrow(ax, (8.825, 9.55), (8.825, 9.12), color=ok_green)
    _box(
        ax,
        (cx_r, 8.28),
        col_w,
        1.12,
        "RAG retrieval — ChromaDB\n17 docs → 341 chunks\ncosine similarity",
        fc=ok_green,
        tc=txt_light,
        fontsize=7.5,
    )
    _arrow(ax, (8.825, 8.28), (8.825, 7.78), color=ok_green)
    _box(
        ax,
        (cx_r, 7.18),
        col_w,
        0.95,
        "Tier-specific system prompt\n(Tier 2: duration / intensity)",
        fc=ok_green,
        tc=txt_light,
        fontsize=7.5,
    )
    _arrow(ax, (8.825, 7.18), (8.825, 6.72), color=ok_green)
    _box(
        ax,
        (cx_r, 5.92),
        col_w,
        0.68,
        "Qwen3-8B + QLoRA\n4-bit NF4 (~43.6M adapter)",
        fc=navy,
        tc=txt_light,
        fontsize=7.5,
    )
    _arrow(ax, (8.825, 5.92), (8.825, 5.45))
    _box(
        ax,
        (cx_r, 4.68),
        col_w,
        0.68,
        "Post-processing\n(strip thinking tags)",
        fc=navy,
        tc=txt_light,
        fontsize=7.5,
    )

    # Merge arrows to response bar
    y_resp = 2.25
    _arrow(ax, (1.625, 7.0), (4.2, y_resp + 0.55), color="0.35")
    _arrow(ax, (5.425, 8.35), (5.0, y_resp + 0.55))
    _arrow(ax, (8.825, 4.68), (5.8, y_resp + 0.55), color="0.35")

    _box(
        ax,
        (2.35, y_resp),
        5.3,
        0.62,
        "Response to User",
        fc=navy,
        tc=txt_light,
        fontsize=9,
    )

    # Legend-style note (minimal)
    ax.text(
        5.0,
        0.85,
        "Tier 4 path (red) bypasses LLM; Tier 3 bypasses RAG + LLM; Tier 1/2 runs full generation.",
        ha="center",
        va="center",
        fontsize=7,
        color=txt_dark,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
