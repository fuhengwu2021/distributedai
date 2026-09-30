"""
DDP environment variables: RANK, LOCAL_RANK, WORLD_SIZE for 2 nodes × 2 GPUs.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "world_size": "WORLD_SIZE = 4",
        "node": "Node {node_idx}",
        "rank": "RANK={r}",
        "local_rank": "LOCAL_RANK={gpu}",
        "gpu": "GPU{gpu}",
    },
    "zh": {
        "world_size": "WORLD_SIZE = 4",
        "node": "节点 {node_idx}",
        "rank": "RANK={r}",
        "local_rank": "LOCAL_RANK={gpu}",
        "gpu": "GPU{gpu}",
    },
}


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_axis_off()
    ax.set_xlim(0.6, 8)
    ax.set_ylim(0.25, 3.5)

    # WORLD_SIZE annotation
    ax.text(4, 3.35, text["world_size"], ha="center", va="center", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd", edgecolor="gray"))

    # Two nodes, 2 GPUs each
    for node_idx in range(2):
        bx = 0.8 + node_idx * 3.8
        node = mpatches.FancyBboxPatch((bx, 0.4), 3.2, 2.6, boxstyle="round,pad=0.08",
                                        facecolor="#e8f4f8", edgecolor="black", linewidth=1.2)
        ax.add_patch(node)
        ax.text(bx + 1.6, 2.85, text["node"].format(node_idx=node_idx), ha="center", va="center", fontsize=12, fontweight="bold")
        for gpu in range(2):
            x = bx + 0.4 + gpu * 1.35
            r = node_idx * 2 + gpu
            ax.add_patch(mpatches.FancyBboxPatch((x, 1.0), 1.2, 1.2, boxstyle="round,pad=0.05",
                                                  facecolor="#99bcff", edgecolor="black", linewidth=1))
            ax.text(x + 0.55, 1.95, text["rank"].format(r=r), ha="center", va="center", fontsize=12)
            ax.text(x + 0.6, 1.5, text["local_rank"].format(gpu=gpu), ha="center", va="center", fontsize=11, color="blue")
            ax.text(x + 0.55, 1.1, text["gpu"].format(gpu=gpu), ha="center", va="center", fontsize=12)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    localized_figure(draw, "ddp_env_vars_multi", LABELS, __file__)
