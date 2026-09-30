#!/usr/bin/env python3
"""
FSDP1 vs FSDP2 parameter layout comparison.
Left: FSDP1 flattens multiple parameters into one FlatParameter, then shards.
Right: FSDP2 shards each parameter individually on dimension 0.
"""
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Ensure shared directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

COLORS = {
    "param1": "#3498db",
    "param2": "#e74c3c",
    "param3": "#2ecc71",
    "flat": "#9b59b6",
    "ranks": ["#3498db", "#e74c3c", "#2ecc71", "#f1c40f"],
    "text": "#2c3e50",
    "arrow": "#7f8c8d",
}

LABELS = {
    "en": {
        "title_fsdp1": "FSDP1: FlatParameter",
        "title_fsdp2": "FSDP2: Per-Parameter Sharding",
        "step1_title": "1. Original Parameters",
        "w1_shape": "W1\n(4096,1024)",
        "w2_shape": "W2\n(1024,)",
        "w3_shape": "W3\n(1024,4096)",
        "arrow_flatten": "Flatten &\nConcatenate",
        "step2_fsdp1": "2. FlatParameter",
        "w1_label": "W1",
        "w2_label": "W2",
        "w3_label": "W3",
        "arrow_shard": "Shard across\nN ranks",
        "step3_fsdp1": "3. Sharded (N=4)",
        "rank_label": "R{i}",
        "note_fsdp1": "Each rank holds 1/N of FlatParameter",
        "arrow_shard_dim0": "Shard each\non dim 0",
        "step2_fsdp2": "2. Per-Parameter Sharding (N=4)",
        "w1_sharded": "W1\n(1024,1024)\n×4",
        "w2_sharded": "W2\n(256,)\n×4",
        "w3_sharded": "W3\n(256,4096)\n×4",
        "note_fsdp2": "Each rank holds dim-0 shard of every parameter",
        "legend_rank": "Rank {i}",
    },
    "zh": {
        "title_fsdp1": "FSDP1: FlatParameter",
        "title_fsdp2": "FSDP2: 逐参数切分",
        "step1_title": "1. 原始参数",
        "w1_shape": "W1\n(4096,1024)",
        "w2_shape": "W2\n(1024,)",
        "w3_shape": "W3\n(1024,4096)",
        "arrow_flatten": "展平并拼接\n(Flatten & Concat)",
        "step2_fsdp1": "2. FlatParameter 扁平张量",
        "w1_label": "W1",
        "w2_label": "W2",
        "w3_label": "W3",
        "arrow_shard": "在 N 个卡间分片",
        "step3_fsdp1": "3. 均匀切分 (N=4)",
        "rank_label": "R{i}",
        "note_fsdp1": "每个 Rank 持有 FlatParameter 的 1/N",
        "arrow_shard_dim0": "每个参数沿\ndim 0 切分",
        "step2_fsdp2": "2. 逐参数切分 (N=4)",
        "w1_sharded": "W1\n(1024,1024)\n×4",
        "w2_sharded": "W2\n(256,)\n×4",
        "w3_sharded": "W3\n(256,4096)\n×4",
        "note_fsdp2": "每个 Rank 持有所有参数的 dim-0 分片",
        "legend_rank": "Rank {i}",
    },
}


def draw_rect(ax, x, y, w, h, color, label="", edgecolor="black", fontsize=13, alpha=1.0):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        linewidth=1.5, edgecolor=edgecolor, facecolor=color, alpha=alpha
    )
    ax.add_patch(rect)
    if label:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", 
                fontsize=fontsize, fontweight="bold", color="white")


def panel_fsdp1(ax, text: dict):
    ax.set_xlim(-0., 5.6)
    ax.set_ylim(-1, 5.5)
    ax.axis("off")
    ax.set_title(text["title_fsdp1"], fontsize=16, fontweight="bold", pad=10)

    # Step 1: Original parameters (different shapes)
    ax.text(2.75, 5.2, text["step1_title"], ha="center", fontsize=14, style="italic")
    draw_rect(ax, 0.2, 4.0, 1.2, 0.8, COLORS["param1"], text["w1_shape"], fontsize=13)
    draw_rect(ax, 1.6, 4.0, 0.8, 0.8, COLORS["param2"], text["w2_shape"], fontsize=13)
    draw_rect(ax, 2.6, 4.0, 1.5, 0.8, COLORS["param3"], text["w3_shape"], fontsize=13)

    # Arrow down
    ax.annotate("", xy=(2.75, 3.0), xytext=(2.75, 3.8),
                arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=2))
    ax.text(4.5, 3.4, text["arrow_flatten"], fontsize=13, ha="left", color=COLORS["arrow"])

    # Step 2: FlatParameter (one long tensor)
    ax.text(2.75, 2.8, text["step2_fsdp1"], ha="center", fontsize=14, style="italic")
    # Draw as segmented bar showing original params concatenated
    total_w = 5.0
    w1, w2, w3 = 1.8, 0.5, 2.7  # proportional to param sizes
    draw_rect(ax, 0.25, 1.8, w1, 0.6, COLORS["param1"], text["w1_label"], fontsize=13)
    draw_rect(ax, 0.25 + w1, 1.8, w2, 0.6, COLORS["param2"], text["w2_label"], fontsize=13)
    draw_rect(ax, 0.25 + w1 + w2, 1.8, w3, 0.6, COLORS["param3"], text["w3_label"], fontsize=13)

    # Arrow down
    ax.annotate("", xy=(2.75, 0.9), xytext=(2.75, 1.6),
                arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=2))
    ax.text(4.5, 1.2, text["arrow_shard"], fontsize=13, ha="left", color=COLORS["arrow"])

    # Step 3: Sharded FlatParameter (4 ranks)
    ax.text(2.75, 0.7, text["step3_fsdp1"], ha="center", fontsize=14, style="italic")
    shard_w = total_w / 4
    for i in range(4):
        draw_rect(ax, 0.25 + i * shard_w, -0.3, shard_w - 0.05, 0.5, COLORS["ranks"][i], text["rank_label"].format(i=i), fontsize=13)
    ax.text(2.75, -1.0, text["note_fsdp1"], ha="center", fontsize=13, color=COLORS["text"])


def panel_fsdp2(ax, text: dict):
    ax.set_xlim(-0., 5.4)
    ax.set_ylim(-1, 5.5)
    ax.axis("off")
    ax.set_title(text["title_fsdp2"], fontsize=16, fontweight="bold", pad=10)

    # Step 1: Original parameters
    ax.text(2.75, 5.2, text["step1_title"], ha="center", fontsize=14, style="italic")
    draw_rect(ax, 0.2, 4.0, 1.2, 0.8, COLORS["param1"], text["w1_shape"], fontsize=13)
    draw_rect(ax, 1.6, 4.0, 0.8, 0.8, COLORS["param2"], text["w2_shape"], fontsize=13)
    draw_rect(ax, 2.6, 4.0, 1.5, 0.8, COLORS["param3"], text["w3_shape"], fontsize=13)

    # Arrow down
    ax.annotate("", xy=(2.75, 3.0), xytext=(2.75, 3.8),
                arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=2))
    ax.text(4.5, 3.4, text["arrow_shard_dim0"], fontsize=13, ha="left", color=COLORS["arrow"])

    # Step 2: Each parameter sharded on dim 0
    ax.text(2.75, 2.8, text["step2_fsdp2"], ha="center", fontsize=14, style="italic")

    # W1: (4096,1024) -> 4 x (1024,1024)
    for i in range(4):
        draw_rect(ax, 0.2 + i*0.3, 1.8 - i*0.15, 0.28, 0.5, COLORS["ranks"][i], "", fontsize=9)
    ax.text(0.75, 0.8, text["w1_sharded"], ha="center", fontsize=12)

    # W2: (1024,) -> 4 x (256,)
    for i in range(4):
        draw_rect(ax, 1.8 + i*0.18, 1.8 - i*0.15, 0.16, 0.5, COLORS["ranks"][i], "", fontsize=9)
    ax.text(2.15, 0.8, text["w2_sharded"], ha="center", fontsize=12)

    # W3: (1024,4096) -> 4 x (256,4096)
    for i in range(4):
        draw_rect(ax, 2.8 + i*0.35, 1.8 - i*0.15, 0.33, 0.5, COLORS["ranks"][i], "", fontsize=9)
    ax.text(3.5, 0.8, text["w3_sharded"], ha="center", fontsize=12)

    # Legend for ranks
    ax.text(2.75, 0.3, text["note_fsdp2"], ha="center", fontsize=13, color=COLORS["text"])

    # Small legend
    for i in range(4):
        draw_rect(ax, 0.5 + i*1.2, -0.5, 0.3, 0.3, COLORS["ranks"][i], "", fontsize=9)
        ax.text(0.5 + i*1.2 + 0.45, -0.35, text["legend_rank"].format(i=i), fontsize=13, va="center")


def draw(text: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    panel_fsdp1(ax1, text)
    panel_fsdp2(ax2, text)

    plt.tight_layout(pad=1.0)
    return fig


if __name__ == "__main__":
    localized_figure(draw, "fsdp1_vs_fsdp2_layout", LABELS, __file__, pad_inches=0.08)
