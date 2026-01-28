"""
Layered communication and system stack for distributed deep learning.

This figure illustrates the strict top-down dependency structure from
high-level frameworks down to the physical communication layer.
"""

import matplotlib.pyplot as plt
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
from math4ai import configure_math_fonts

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

y_positions = list(range(len(layers)))[::-1]

# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3, 9))

# Draw layer boxes
for y, layer in zip(y_positions, layers):
    ax.text(
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

# Draw arrows (bottom → top)
# Connect from top of lower box to bottom of higher box
# y_positions is [6,5,4,3,2,1,0] (top to bottom in list, but y=0 is bottom visually)
# For bottom-to-top arrows, we go from lower y to higher y
for i in range(len(y_positions) - 1):
    lower_y = y_positions[i + 1]    # Lower y value (bottom box, e.g., y=0)
    higher_y = y_positions[i]       # Higher y value (top box, e.g., y=1)
    # Lower box top (approximate, accounting for bbox padding ~0.45)
    lower_box_top = lower_y + 0.3
    # Higher box bottom
    higher_box_bottom = higher_y - 0.3
    ax.annotate(
        "",
        xy=(0.5, higher_box_bottom),  # Destination: bottom of higher box
        xytext=(0.5, lower_box_top),   # Source: top of lower box
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.6,
        ),
    )

# ---------------------------------------------------------------------
# Styling (clean schematic)
# ---------------------------------------------------------------------
ax.set_xlim(0, 1)
ax.set_ylim(-0.5, len(layers) - 0.5)
ax.set_xticks([])
ax.set_yticks([])
ax.axis("off")

plt.tight_layout()

# ---------------------------------------------------------------------
# Save figure (standard Math4AIML pattern)
# ---------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.splitext(os.path.basename(__file__))[0]
output_path = os.path.join(script_dir, f"{script_name}.png")

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    pad_inches=0.03,
)
print(f"Saved figure to: {output_path}")

plt.close()

