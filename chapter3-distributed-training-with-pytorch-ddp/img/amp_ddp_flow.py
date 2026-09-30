"""
AMP + DDP flow: scale loss → backward → DDP AllReduce → unscale → optimizer step.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "steps": [
            "Scale loss",
            "Backward",
            "DDP AllReduce",
            "Unscale",
            "Optimizer step",
        ],
    },
    "zh": {
        "steps": [
            "损失缩放",
            "反向求导",
            "DDP AllReduce",
            "解缩放",
            "优化器步进",
        ],
    },
}


def draw(text: dict) -> plt.Figure:
    n_boxes = 5
    w = 1.35
    gap = 0.64
    x_start = 0.3
    total_w = x_start + n_boxes * w + (n_boxes - 1) * gap + 0.3

    fig, ax = plt.subplots(figsize=(total_w, 1.05))
    ax.set_axis_off()
    ax.set_xlim(0, total_w)
    ax.set_ylim(0.6, 2.0)

    labels = text["steps"]
    colors = ["#ffdb99", "#99bcff", "#b19cd9", "#ffdb99", "#e69183"]

    for i, (label, color) in enumerate(zip(labels, colors)):
        x = x_start + i * (w + gap)
        ax.add_patch(mpatches.FancyBboxPatch((x, 0.8), w, 1.0, boxstyle="round,pad=0.028",
                                              facecolor=color, edgecolor="black", linewidth=1))
        ax.text(x + w / 2, 1.3, label, ha="center", va="center", fontsize=11)
        if i < len(labels) - 1:
            # Arrow from right edge of this box to left edge of next box (full gap)
            ax.annotate("", xy=(x + w + gap - 0.0005, 1.3), xytext=(x + w, 1.3),
                        arrowprops=dict(arrowstyle="->", lw=1.5))

    plt.tight_layout(pad=0.00001)
    return fig


if __name__ == '__main__':
    localized_figure(draw, "amp_ddp_flow", LABELS, __file__, pad_inches=0)
