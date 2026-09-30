#!/usr/bin/env python3
"""
Device Mesh visualization: 1D mesh for FSDP vs 2D mesh for HSDP (Hybrid Sharding).
Left: 1D mesh - all GPUs in one dimension, full sharding across all.
Right: 2D mesh - shard within node (dim 1), replicate across nodes (dim 0).
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
    "gpu": "#3498db",
    "node": "#ecf0f1",
    "node_border": "#bdc3c7",
    "highlight": "#e74c3c",
    "text": "#2c3e50",
    "arrow": "#7f8c8d",
    "shard_colors": ["#3498db", "#e74c3c", "#2ecc71", "#f1c40f", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"],
}

LABELS = {
    "en": {
        "title_1d": "1D Mesh (FSDP)",
        "title_2d": "2D Mesh (HSDP)",
        "rank": "R{i}",
        "node": "N{node_idx}",
    },
    "zh": {
        "title_1d": "1D 网格 (FSDP)",
        "title_2d": "2D 网格 (HSDP)",
        "rank": "R{i}",
        "node": "N{node_idx}",
    },
}


def draw_gpu(ax, x, y, label, color="#3498db", size=0.4):
    rect = patches.FancyBboxPatch(
        (x - size/2, y - size/2), size, size,
        boxstyle="round,pad=0.02",
        linewidth=1.5, edgecolor="black", facecolor=color
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=11, fontweight="bold", color="white")


def panel_1d_mesh(ax, text: dict):
    ax.set_xlim(-0.5, 9)
    ax.set_ylim(-0.5, 2.5)
    ax.axis("off")
    ax.set_title(text["title_1d"], fontsize=20, fontweight="bold", pad=10)

    # Draw 4 GPUs in a row - larger and centered
    for i in range(4):
        draw_gpu(ax, 1.5 + i * 1.8, 1.0, text["rank"].format(i=i), COLORS["shard_colors"][i], size=0.8)

    # Double-headed arrow showing single dimension
    ax.annotate("", xy=(1.0, 0.2), xytext=(7.9, 0.2),
                arrowprops=dict(arrowstyle="<->", color=COLORS["arrow"], lw=2.5))


def panel_2d_mesh(ax, text: dict):
    ax.set_xlim(-0.5, 9)
    ax.set_ylim(-0.5, 3.5)
    ax.axis("off")
    ax.set_title(text["title_2d"], fontsize=20, fontweight="bold", pad=10)

    # Draw 2 nodes x 4 GPUs - larger
    node_w, node_h = 5.5, 1.2
    gpu_spacing = 1.3

    for node_idx in range(2):
        node_y = 2.2 - node_idx * 1.6
        # Node box
        node_rect = patches.FancyBboxPatch(
            (0.8, node_y - 0.4), node_w, node_h,
            boxstyle="round,pad=0.05",
            linewidth=2, edgecolor=COLORS["node_border"], facecolor=COLORS["node"], alpha=0.5
        )
        ax.add_patch(node_rect)
        ax.text(0.5, node_y + 0.2, text["node"].format(node_idx=node_idx), ha="right", fontsize=14, fontweight="bold")

        # GPUs in node
        for gpu_idx in range(4):
            color_idx = gpu_idx  # Same color for same position across nodes (replicated)
            draw_gpu(ax, 1.2 + gpu_idx * gpu_spacing, node_y + 0.2,
                     text["rank"].format(i=node_idx*4 + gpu_idx), COLORS["shard_colors"][color_idx], size=0.55)

    # Annotations for dimensions - simplified
    ax.annotate("", xy=(0.0, 2.4), xytext=(0.0, 0.6),
                arrowprops=dict(arrowstyle="<->", color="#e74c3c", lw=2.5))

    ax.annotate("", xy=(1.0, 2.9), xytext=(6.1, 2.9),
                arrowprops=dict(arrowstyle="<->", color="#2ecc71", lw=2.5))


def draw(text: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    panel_1d_mesh(ax1, text)
    panel_2d_mesh(ax2, text)

    plt.tight_layout(pad=1.0)
    return fig


if __name__ == "__main__":
    localized_figure(draw, "device_mesh", LABELS, __file__, pad_inches=0.08)
