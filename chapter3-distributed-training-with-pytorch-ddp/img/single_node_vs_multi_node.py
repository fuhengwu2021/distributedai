"""
Single-node vs multi-node DDP: one machine with 2 GPUs vs two machines
with 2 GPUs each (RANK and LOCAL_RANK layout).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from math4ai import save_figure

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
ax.text(0.5 + node_w / 2, 2.75, "Node 0", ha="center", va="center", fontsize=12, fontweight="bold")
for i in range(gpus_per_node):
    x = 0.7 + i * 1.3
    ax.add_patch(mpatches.FancyBboxPatch((x, 1.0), 0.9, 0.9, boxstyle="round,pad=0.05",
                                          facecolor="#99bcff", edgecolor="black", linewidth=1))
    ax.text(x + 0.45, 1.45, f"GPU{i}", ha="center", va="center", fontsize=10)
    ax.text(x + 0.45, 1.05, f"RANK {i}", ha="center", va="bottom", fontsize=10)
    ax.text(x + 0.45, 0.72, f"LOCAL_RANK {i}", ha="center", va="top", fontsize=9, color="blue")
ax.text(0.5 + node_w / 2, 0.25, "Single-node (1 machine, 2 GPUs)", ha="center", va="center", fontsize=11)

# --- Multi-node: two boxes, 2 GPUs each ---
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
for node_idx in range(2):
    bx = 0.3 + node_idx * 4.8
    node = mpatches.FancyBboxPatch((bx-0.15, 0.5), 4.2, 3, boxstyle="round,pad=0.1",
                                   facecolor="#e8c0f8", edgecolor="black", linewidth=1.5)
    ax.add_patch(node)
    ax.text(bx + 1.82, 3., f"Node {node_idx}", ha="center", va="center", fontsize=12, fontweight="bold")
    for i in range(gpus_per_node):
        x = bx + 0.35 + i * 2.35
        r = node_idx * gpus_per_node + i
        ax.add_patch(mpatches.FancyBboxPatch((x, 1.0), 1.05, 0.9, boxstyle="round,pad=0.05",
                                              facecolor="#eeffff", edgecolor="black", linewidth=1))
        ax.text(x + 0.495, 1.54, f"GPU{i}", ha="center", va="center", fontsize=9)
        ax.text(x + 0.495, 1.12, f"RANK {r}", ha="center", va="bottom", fontsize=9)
        ax.text(x + 0.435, 0.75, f"LOCAL_RANK {i}", ha="center", va="top", fontsize=9, color="blue")
ax.text(5.0, 0.025, "Multi-node (2 machines, 4 GPUs)", ha="center", va="center", fontsize=11)

plt.tight_layout()
save_figure(__file__)
