"""
DataLoader with multiple workers: main process, index queue, worker processes,
result queue, and prefetch overlap with training.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from math4ai import save_figure

fig, ax = plt.subplots(figsize=(10, 4))
ax.set_axis_off()
ax.set_xlim(0, 11)
ax.set_ylim(0.3, 3)

def box(ax, x, y, w, h, label, facecolor="#e6f0ff"):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                          facecolor=facecolor, edgecolor="black", linewidth=1))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)

# Main process (training; workers prefetch overlap)
box(ax, 0.5, 1.35, 2.3, 1.35, "Main process\n(training)\n(while workers prefetch)", "#ffdb99")
# Index queue (batch indices)
box(ax, 3.2, 1.8, 1.6, 0.6, "Index queue\n(batch indices)", "#d4edda")
# Result queue (batch tensors)
box(ax, 3.2, 0.8, 1.6, 0.6, "Result queue\n(batch tensors)", "#d4edda")
# Workers (each has its own Dataset copy)
for i in range(3):
    box(ax, 5.2 + i * 1.9, 0.5, 1.7, 1.0, f"Worker {i}\n(Dataset copy)", "#cce5ff")

# Arrows (flow: Main → index queue → workers → result queue → Main)
ax.annotate("", xy=(3.2, 2.0), xytext=(2.8, 2.0), arrowprops=dict(arrowstyle="->", lw=1.2))  # Main → Index queue
ax.annotate("", xy=(2.8, 1.7), xytext=(3.2, 1.0), arrowprops=dict(arrowstyle="->", lw=1.2))  # Result queue → Main
for i in range(3):
    ax.annotate("", xy=(5.2 + i * 1.9 + 0.35, 1.0), xytext=(4.8, 2.0), arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"))  # Index queue → Worker
    ax.annotate("", xy=(4.8, 1.0), xytext=(5.2 + i * 1.9 + 0.35, 0.5), arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"))  # Worker → Result queue

plt.tight_layout()
save_figure(__file__)
