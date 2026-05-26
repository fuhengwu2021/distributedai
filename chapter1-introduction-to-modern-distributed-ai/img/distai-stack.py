"""
Layered communication and system stack for distributed deep learning.

This figure illustrates the strict top-down dependency structure from
high-level frameworks down to the physical communication layer.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os
import sys

# Add shared directory to path for math4ai imports
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "shared",
    ),
)
from math4ai import configure_math_fonts, save_figure

# Configure matplotlib fonts (MANDATORY)
configure_math_fonts()

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
layers = [
    "Framework Layer",
    "Messaging Layer",
    "Collective Operations Layer",
    "Data Transfer Layer",
    "Topology Layer",
    "Link Layer",
    "Physical Layer",
]

LAYER_STEP = 0.9  # vertical spacing between layer centers
y_positions = [i * LAYER_STEP for i in range(len(layers))][::-1]

# Leave a fraction of the inter-box gap clear at each arrow end (not inside boxes)
ARROW_END_GAP_FRAC = 0.1

# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
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

# Arrows: lower box top → upper box bottom (bbox-measured, minimal gap)
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
save_figure(__file__)
