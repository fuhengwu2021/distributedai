"""
Single-node vs multi-node DDP: one machine with 2 GPUs vs two machines
with 2 GPUs each (RANK and LOCAL_RANK layout).
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "node": "Node {node_idx}",
        "gpu": "GPU{i}",
        "rank": "RANK {r}",
        "local_rank": "LOCAL_RANK {i}",
        "single_caption": "Single-node (1 machine, 2 GPUs)",
        "multi_caption": "Multi-node (2 machines, 4 GPUs)",
    },
    "zh": {
        "node": "节点 {node_idx}",
        "gpu": "GPU{i}",
        "rank": "RANK {r}",
        "local_rank": "LOCAL_RANK {i}",
        "single_caption": "单机架构（1 台主机，2 块 GPU）",
        "multi_caption": "多机架构（2 台主机，4 块 GPU）",
    },
}


def draw(text: dict) -> plt.Figure:
    gpus_per_node = 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax in axes:
        ax.set_axis_off()
        ax.set_aspect("equal")

    # --- Single node: one box with 2 GPUs ---
    ax = axes[0]
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 4)
    node_w = 2.6
    node = mpatches.FancyBboxPatch((0.5, 0.5), node_w, 3, boxstyle="round,pad=0.1",
                                    facecolor="#e8f4f8", edgecolor="black", linewidth=1.5)
    ax.add_patch(node)
    ax.text(0.5 + node_w / 2, 2.75, text["node"].format(node_idx=0), ha="center", va="center", fontsize=12, fontweight="bold")
    for i in range(gpus_per_node):
        x = 0.7 + i * 1.3
        ax.add_patch(mpatches.FancyBboxPatch((x, 1.0), 0.9, 0.9, boxstyle="round,pad=0.05",
                                              facecolor="#99bcff", edgecolor="black", linewidth=1))
        ax.text(x + 0.45, 1.45, text["gpu"].format(i=i), ha="center", va="center", fontsize=10)
        ax.text(x + 0.45, 1.05, text["rank"].format(r=i), ha="center", va="bottom", fontsize=10)
        ax.text(x + 0.45, 0.72, text["local_rank"].format(i=i), ha="center", va="top", fontsize=9, color="blue")
    ax.text(0.5 + node_w / 2, 0.25, text["single_caption"], ha="center", va="center", fontsize=11)

    # --- Multi-node: two boxes, 2 GPUs each ---
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    for node_idx in range(2):
        bx = 0.3 + node_idx * 4.8
        node = mpatches.FancyBboxPatch((bx - 0.15, 0.5), 4.2, 3, boxstyle="round,pad=0.1",
                                       facecolor="#e8c0f8", edgecolor="black", linewidth=1.5)
        ax.add_patch(node)
        ax.text(bx + 1.82, 3., text["node"].format(node_idx=node_idx), ha="center", va="center", fontsize=12, fontweight="bold")
        for i in range(gpus_per_node):
            x = bx + 0.35 + i * 2.35
            r = node_idx * gpus_per_node + i
            ax.add_patch(mpatches.FancyBboxPatch((x, 1.0), 1.05, 0.9, boxstyle="round,pad=0.05",
                                                  facecolor="#eeffff", edgecolor="black", linewidth=1))
            ax.text(x + 0.495, 1.54, text["gpu"].format(i=i), ha="center", va="center", fontsize=9)
            ax.text(x + 0.495, 1.12, text["rank"].format(r=r), ha="center", va="bottom", fontsize=9)
            ax.text(x + 0.435, 0.75, text["local_rank"].format(i=i), ha="center", va="top", fontsize=9, color="blue")
    ax.text(5.0, 0.025, text["multi_caption"], ha="center", va="center", fontsize=11)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    localized_figure(draw, "single_node_vs_multi_node", LABELS, __file__)
