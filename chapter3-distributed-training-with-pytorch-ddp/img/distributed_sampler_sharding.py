"""
DistributedSampler sharding: dataset split into non-overlapping shards,
one per rank (e.g. rank 0 gets indices 0..249, rank 1 gets 250..499, ...).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from math4ai import save_figure

fig, ax = plt.subplots(figsize=(10, 2.5))
ax.set_axis_off()
ax.set_xlim(1.4, 9.6)
ax.set_ylim(1, 3)

# Dataset bar (full length)
dataset_w = 8
dataset_x = 1.5
ax.add_patch(mpatches.FancyBboxPatch((dataset_x, 1.2), dataset_w, 0.7, boxstyle="round,pad=0.02",
                                      facecolor="lightgray", edgecolor="black", linewidth=1))
ax.text(dataset_x + dataset_w / 2 + 0.03, 2.55, "Dataset [0 .. N-1]", ha="center", va="center", fontsize=13)

# Four shards
n_ranks = 4
colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#f90024"]
shard_w = dataset_w / n_ranks
for i in range(n_ranks):
    x = dataset_x + i * shard_w
    ax.add_patch(mpatches.Rectangle((x, 1.2), shard_w, 0.7, facecolor=colors[i], edgecolor="black", linewidth=1))
    ax.text(x + shard_w / 2, 1.55, f"Rank {i}", ha="center", va="center", fontsize=13)

# DistributedSampler label
ax.text(dataset_x + dataset_w / 2, 2.7, "DistributedSampler", ha="center", va="bottom", fontsize=13, fontweight="bold")
# Common start: one point below "Dataset [0 .. N-1]"
x_start = dataset_x + dataset_w / 2
y_start = 2.45
# Four arrows: same start, each to the center of its rank shard
for i in range(n_ranks):
    x_end = dataset_x + (i + 0.5) * shard_w
    y_end = 1.65
    ax.annotate("", xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", lw=1.1, color=colors[i]))

plt.tight_layout()
save_figure(__file__)
