#!/usr/bin/env python3
"""
reshard_after_forward comparison: True (ZeRO-3 style) vs False (ZeRO-2 style).
Shows memory vs communication tradeoff during forward and backward passes.
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
    "params_sharded": "#3498db",
    "params_full": "#2ecc71",
    "grads": "#e74c3c",
    "compute": "#f1c40f",
    "comm": "#9b59b6",
    "text": "#2c3e50",
    "arrow": "#7f8c8d",
    "memory_bar": "#3498db",
    "comm_bar": "#e74c3c",
}

LABELS = {
    "en": {
        "title_true": "reshard_after_forward=True",
        "title_false": "reshard_after_forward=False",
        "fwd": "Fwd",
        "bwd": "Bwd",
        "ag": "AG",
        "l1": "L1",
        "l2": "L2",
        "free": "free",
        "keep": "keep",
        "rs": "RS",
    },
    "zh": {
        "title_true": "reshard_after_forward=True（ZeRO-3 模式）",
        "title_false": "reshard_after_forward=False（ZeRO-2 模式）",
        "fwd": "Fwd",
        "bwd": "Bwd",
        "ag": "AG",
        "l1": "L1",
        "l2": "L2",
        "free": "free",
        "keep": "keep",
        "rs": "RS",
    },
}


def draw_timeline_block(ax, x, y, w, h, color, label="", fontsize=9):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01",
        linewidth=1, edgecolor="black", facecolor=color
    )
    ax.add_patch(rect)
    if label:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", 
                fontsize=fontsize, fontweight="bold", color="white")


def panel_reshard_true(ax, text: dict):
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(0.5, 5)
    ax.axis("off")
    ax.set_title(text["title_true"], fontsize=16, fontweight="bold", pad=10)

    # Timeline
    y_fwd = 3.5
    y_bwd = 1.5
    block_h = 0.7

    ax.text(-0.3, y_fwd + 0.3, text["fwd"], fontsize=13, fontweight="bold", ha="right")
    ax.text(-0.3, y_bwd + 0.3, text["bwd"], fontsize=13, fontweight="bold", ha="right")

    # Forward: all-gather -> compute -> free (reshard)
    draw_timeline_block(ax, 0, y_fwd, 1.8, block_h, COLORS["comm"], text["ag"], 11)
    draw_timeline_block(ax, 2.0, y_fwd, 2.2, block_h, COLORS["compute"], text["l1"], 11)
    ax.text(4.4, y_fwd + 0.35, text["free"], fontsize=10, color=COLORS["arrow"], style="italic")
    draw_timeline_block(ax, 5.0, y_fwd, 1.8, block_h, COLORS["comm"], text["ag"], 11)
    draw_timeline_block(ax, 7.0, y_fwd, 2.2, block_h, COLORS["compute"], text["l2"], 11)

    # Backward: all-gather -> compute grad -> reduce-scatter
    draw_timeline_block(ax, 0, y_bwd, 1.5, block_h, COLORS["comm"], text["ag"], 11)
    draw_timeline_block(ax, 1.6, y_bwd, 1.5, block_h, COLORS["compute"], text["l2"], 11)
    draw_timeline_block(ax, 3.2, y_bwd, 1.5, block_h, COLORS["grads"], text["rs"], 11)
    draw_timeline_block(ax, 4.8, y_bwd, 1.5, block_h, COLORS["comm"], text["ag"], 11)
    draw_timeline_block(ax, 6.4, y_bwd, 1.5, block_h, COLORS["compute"], text["l1"], 11)
    draw_timeline_block(ax, 8.0, y_bwd, 1.5, block_h, COLORS["grads"], text["rs"], 11)


def panel_reshard_false(ax, text: dict):
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(0.5, 5)
    ax.axis("off")
    ax.set_title(text["title_false"], fontsize=16, fontweight="bold", pad=10)

    y_fwd = 3.5
    y_bwd = 1.5
    block_h = 0.7

    ax.text(-0.3, y_fwd + 0.3, text["fwd"], fontsize=13, fontweight="bold", ha="right")
    ax.text(-0.3, y_bwd + 0.3, text["bwd"], fontsize=13, fontweight="bold", ha="right")

    # Forward: all-gather -> compute -> KEEP (no reshard)
    draw_timeline_block(ax, 0, y_fwd, 1.8, block_h, COLORS["comm"], text["ag"], 11)
    draw_timeline_block(ax, 2.0, y_fwd, 2.2, block_h, COLORS["compute"], text["l1"], 11)
    ax.text(4.4, y_fwd + 0.35, text["keep"], fontsize=10, color="#2ecc71", style="italic", fontweight="bold")
    draw_timeline_block(ax, 5.0, y_fwd, 1.8, block_h, COLORS["comm"], text["ag"], 11)
    draw_timeline_block(ax, 7.0, y_fwd, 2.2, block_h, COLORS["compute"], text["l2"], 11)

    # Backward: NO all-gather needed (params still in memory) -> compute grad -> reduce-scatter
    draw_timeline_block(ax, 0, y_bwd, 2.2, block_h, COLORS["compute"], text["l2"], 11)
    draw_timeline_block(ax, 2.4, y_bwd, 1.8, block_h, COLORS["grads"], text["rs"], 11)
    draw_timeline_block(ax, 4.4, y_bwd, 2.2, block_h, COLORS["compute"], text["l1"], 11)
    draw_timeline_block(ax, 6.8, y_bwd, 1.8, block_h, COLORS["grads"], text["rs"], 11)


def draw(text: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    panel_reshard_true(ax1, text)
    panel_reshard_false(ax2, text)

    plt.tight_layout(pad=1.0)
    return fig


if __name__ == "__main__":
    localized_figure(draw, "reshard_after_forward", LABELS, __file__, pad_inches=0.08)
