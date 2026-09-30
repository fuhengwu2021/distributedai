#!/usr/bin/env python3
"""
Layered communication and system stack for distributed deep learning.

This figure illustrates the strict top-down dependency structure from
high-level frameworks down to the physical communication layer.
"""

import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Ensure shared directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "layers": [
            "Framework Layer",
            "Messaging Layer",
            "Collective Operations Layer",
            "Data Transfer Layer",
            "Topology Layer",
            "Link Layer",
            "Physical Layer",
        ]
    },
    "zh": {
        "layers": [
            "框架层",
            "消息传递层",
            "集合通信操作层",
            "数据传输层",
            "网络拓扑层",
            "物理链路层",
            "物理硬件层",
        ]
    }
}


def draw(text: dict) -> plt.Figure:
    layers = text["layers"]
    LAYER_STEP = 0.9  # vertical spacing between layer centers
    y_positions = [i * LAYER_STEP for i in range(len(layers))][::-1]
    ARROW_END_GAP_FRAC = 0.1

    fig, ax = plt.subplots(figsize=(3, 9))

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.45, (len(layers) - 1) * LAYER_STEP + 0.45)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    layer_texts = []
    for y, layer in zip(y_positions, layers):
        t = ax.text(
            0.5,
            y,
            layer,
            ha="center",
            va="center",
            fontsize=14,
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor="white",
                edgecolor="black",
                linewidth=1.5,
            ),
        )
        layer_texts.append(t)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_data = ax.transData.inverted()

    for i in range(len(layer_texts) - 1):
        upper = layer_texts[i]
        lower = layer_texts[i + 1]
        bb_upper = upper.get_window_extent(renderer).transformed(to_data)
        bb_lower = lower.get_window_extent(renderer).transformed(to_data)
        gap = bb_upper.y0 - bb_lower.y1
        pad = gap * ARROW_END_GAP_FRAC
        ax.add_patch(
            FancyArrowPatch(
                (0.5, bb_lower.y1 + pad),
                (0.5, bb_upper.y0 - pad),
                arrowstyle="->",
                mutation_scale=14,
                linewidth=1.6,
                shrinkA=0,
                shrinkB=0,
                clip_on=False,
                transform=ax.transData,
            )
        )

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    localized_figure(draw, "distai-stack", LABELS, __file__)
