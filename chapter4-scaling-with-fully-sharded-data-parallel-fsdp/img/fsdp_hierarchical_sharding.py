#!/usr/bin/env python3
"""
Hierarchical sharding in FSDP: wrapping at different levels.
Shows how applying fully_shard to individual blocks vs the whole model
creates different FSDP units and communication boundaries.
"""
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure shared directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

COLORS = {
    "embedding": "#95a5a6",
    "block": "#3498db",
    "block_sharded": "#2980b9",
    "lm_head": "#95a5a6",
    "fsdp_boundary": "#e74c3c",
    "text": "#2c3e50",
    "arrow": "#7f8c8d",
    "ranks": ["#3498db", "#e74c3c", "#2ecc71", "#f1c40f"],
}

LABELS = {
    "en": {
        "title_hierarchical": "Hierarchical Sharding",
        "title_flat": "Flat Sharding",
        "embed": "Embed",
        "block": "Block {i}",
        "head": "Head",
    },
    "zh": {
        "title_hierarchical": "分层分片 (Hierarchical Sharding)",
        "title_flat": "扁平切分 (Flat Sharding)",
        "embed": "Embed (嵌入层)",
        "block": "Block {i}",
        "head": "Head (输出头)",
    },
}


def draw_module(ax, x, y, w, h, color, label, fontsize=10, edgecolor="black", linestyle="-", linewidth=1.5):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        linewidth=linewidth, edgecolor=edgecolor, facecolor=color, linestyle=linestyle
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", 
            fontsize=fontsize, fontweight="bold", color="white")


def draw_fsdp_boundary(ax, x, y, w, h, label=""):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03",
        linewidth=2.5, edgecolor=COLORS["fsdp_boundary"], facecolor="none", linestyle="--"
    )
    ax.add_patch(rect)
    if label:
        ax.text(x + w + 0.1, y + h/2, label, fontsize=9, color=COLORS["fsdp_boundary"], 
                va="center", fontweight="bold")


def panel_hierarchical(ax, text: dict):
    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.2, 6)
    ax.axis("off")
    ax.set_title(text["title_hierarchical"], fontsize=18, fontweight="bold", pad=10)

    # Model structure - larger blocks
    x_start = 0.8
    block_w, block_h = 2.5, 0.8
    spacing = 0.2

    # Embedding (not sharded in this example)
    draw_module(ax, x_start, 5.0, block_w, block_h, COLORS["embedding"], text["embed"], fontsize=13)

    # Transformer blocks (each is an FSDP unit)
    for i in range(4):
        y_pos = 4.0 - i * (block_h + spacing)
        draw_module(ax, x_start, y_pos, block_w, block_h, COLORS["block"], text["block"].format(i=i), fontsize=13)
        draw_fsdp_boundary(ax, x_start - 0.15, y_pos - 0.1, block_w + 0.3, block_h + 0.2, "")

    # LM Head (not sharded)
    draw_module(ax, x_start, 0.0, block_w, block_h, COLORS["lm_head"], text["head"], fontsize=13)


def panel_flat(ax, text: dict):
    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.2, 6)
    ax.axis("off")
    ax.set_title(text["title_flat"], fontsize=18, fontweight="bold", pad=10)

    x_start = 0.8
    block_w, block_h = 2.5, 0.8
    spacing = 0.2

    # All modules in one FSDP unit
    draw_module(ax, x_start, 5.0, block_w, block_h, COLORS["embedding"], text["embed"], fontsize=13)
    for i in range(4):
        y_pos = 4.0 - i * (block_h + spacing)
        draw_module(ax, x_start, y_pos, block_w, block_h, COLORS["block"], text["block"].format(i=i), fontsize=13)
    draw_module(ax, x_start, 0.0, block_w, block_h, COLORS["lm_head"], text["head"], fontsize=13)

    # Single FSDP boundary around everything
    draw_fsdp_boundary(ax, x_start - 0.25, -0.15, block_w + 0.5, 5.8, "")


def draw(text: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    panel_hierarchical(ax1, text)
    panel_flat(ax2, text)

    plt.tight_layout(pad=1.0)
    return fig


if __name__ == "__main__":
    localized_figure(draw, "fsdp_hierarchical_sharding", LABELS, __file__, pad_inches=0.08)
