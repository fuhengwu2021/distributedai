"""
Visualize the Ring AllReduce algorithm used by NCCL for gradient synchronization.
Shows 4 ranks in a ring; each rank sends/receives chunks along the ring in two
phases (scatter-reduce then allgather) so all ranks get the full reduced result.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import os

# Optional: use math4ai for LaTeX-style fonts and save (if shared is available)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from math4ai import configure_math_fonts, save_figure
configure_math_fonts()
# ---------------------------
# Setup: 4 ranks on a ring
# ---------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect("equal")
ax.axis("off")

num_ranks = 4
radius = 1.8
node_radius = 0.42
arrow_radius = radius + 0.35

# Node positions (rank 0 at top, then clockwise)
angles = np.linspace(90, 90 - 360, num_ranks, endpoint=False) * np.pi / 180
xs = radius * np.cos(angles)
ys = radius * np.sin(angles)

colors = ["#0096d6", "#d62728", "#6aa84f", "#ffd11a"]

# ---------------------------
# Draw ring (light gray circle)
# ---------------------------
ring = Circle((0, 0), radius + node_radius * 1.2, fill=False,
              edgecolor="lightgray", linewidth=1, linestyle="--")
ax.add_patch(ring)

# ---------------------------
# Draw nodes (ranks)
# ---------------------------
for i in range(num_ranks):
    circ = Circle((xs[i], ys[i]), node_radius, facecolor=colors[i],
                  edgecolor="black", linewidth=1.5, zorder=3)
    ax.add_patch(circ)
    ax.text(xs[i], ys[i], f"{i}", ha="center", va="center", fontsize=16,
            fontweight="bold", color="white", zorder=4)

# Rank labels below nodes
for i in range(num_ranks):
    r = radius - node_radius - 0.38
    ax.text(r * np.cos(angles[i]), r * np.sin(angles[i]), f"rank {i}",
            ha="center", va="center", fontsize=13)

# ---------------------------
# Draw arrows along the ring (clockwise: 0 -> 1 -> 2 -> 3 -> 0)
# Start/end at circle edges so arrows touch the node borders
# ---------------------------
for i in range(num_ranks):
    j = (i + 1) % num_ranks
    # Unit vector from center i toward center j
    dx = xs[j] - xs[i]
    dy = ys[j] - ys[i]
    d = np.hypot(dx, dy)
    if d > 0:
        ux, uy = dx / d, dy / d
        ax_start = xs[i] + node_radius * ux
        ay_start = ys[i] + node_radius * uy
        ax_end = xs[j] - node_radius * ux
        ay_end = ys[j] - node_radius * uy
        arrow = FancyArrowPatch(
            (ax_start, ay_start), (ax_end, ay_end),
            arrowstyle="->", mutation_scale=20, linewidth=2, color="gray", zorder=5
        )
        ax.add_patch(arrow)

# ---------------------------
# Limits and save (tight bounds, no extra border)
# ---------------------------
# Content extent: outer dashed ring at radius + node_radius*1.2; small margin for arrows
lim = radius + node_radius * 1.2 + 0.12
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
plt.tight_layout(pad=0.1)
save_figure(__file__, pad_inches=0)
