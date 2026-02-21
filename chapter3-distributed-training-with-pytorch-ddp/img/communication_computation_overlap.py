"""
Overlap of backward computation and AllReduce: compute and communication
run concurrently so AllReduce for one bucket overlaps compute for the next.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from math4ai import save_figure

fig, ax = plt.subplots(figsize=(10, 3))
ax.set_axis_off()
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)

# Two horizontal lanes: Backward compute (top), AllReduce (bottom)
y_compute = 2.5
y_comm = 1.0
h = 0.6

# Backward compute blocks (three segments)
compute_blocks = [(0.5, 2.2), (2.8, 2.0), (5.2, 2.2)]
for i, (x, w) in enumerate(compute_blocks):
    rect = mpatches.FancyBboxPatch((x, y_compute), w, h, boxstyle="round,pad=0.02",
                                    facecolor="#99bcff", edgecolor="black", linewidth=1)
    ax.add_patch(rect)
    ax.text(x + w / 2, y_compute + h / 2, "Backward", ha="center", va="center", fontsize=11)

# AllReduce blocks (overlapping with gaps between compute blocks)
allreduce_blocks = [(2.0, 1.8), (4.2, 1.8), (6.4, 1.8)]
for i, (x, w) in enumerate(allreduce_blocks):
    rect = mpatches.FancyBboxPatch((x, y_comm), w, h, boxstyle="round,pad=0.02",
                                    facecolor="#b19cd9", edgecolor="black", linewidth=1)
    ax.add_patch(rect)
    ax.text(x + w / 2, y_comm + h / 2, "AllReduce", ha="center", va="center", fontsize=11)

# Lane labels
ax.text(-0.3, y_compute + h / 2, "Compute", fontsize=11, va="center")
ax.text(-0.3, y_comm + h / 2, "Comm", fontsize=11, va="center")

# Time axis
ax.plot([0.2, 9.5], [0.3, 0.3], "k-", linewidth=0.8)
ax.text(9.7, 0.3, "Time", fontsize=11, va="center")

plt.tight_layout()
save_figure(__file__)
