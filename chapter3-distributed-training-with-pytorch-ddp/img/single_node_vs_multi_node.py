"""
Single-node vs multi-node DDP: one machine with 4 GPUs vs two machines
with 4 GPUs each (RANK and LOCAL_RANK layout).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from math4ai import save_figure

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax in axes:
    ax.set_axis_off()
    ax.set_aspect("equal")

# --- Single node: one box with 4 GPUs ---
ax = axes[0]
ax.set_xlim(0, 5)
ax.set_ylim(0, 4)
node = mpatches.FancyBboxPatch((0.3, 0.5), 4.4, 3, boxstyle="round,pad=0.1",
                                facecolor="#e8f4f8", edgecolor="black", linewidth=1.5)
ax.add_patch(node)
ax.text(2.5, 3.5, "Node 0", ha="center", va="center", fontsize=12, fontweight="bold")
for i in range(4):
    x = 0.8 + i * 1.05
    ax.add_patch(mpatches.FancyBboxPatch((x, 1.0), 0.8, 0.9, boxstyle="round,pad=0.05",
                                          facecolor="#99bcff", edgecolor="black", linewidth=1))
    ax.text(x + 0.4, 1.45, f"GPU{i}", ha="center", va="center", fontsize=10)
    ax.text(x + 0.4, 1.05, f"RANK {i}", ha="center", va="bottom", fontsize=9)
    ax.text(x + 0.4, 0.72, f"LOCAL_RANK {i}", ha="center", va="top", fontsize=8, color="gray")
ax.text(2.5, 0.25, "Single-node (1 machine, 4 GPUs)", ha="center", va="center", fontsize=11)

# --- Multi-node: two boxes, 4 GPUs each ---
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
for node_idx in range(2):
    bx = 0.3 + node_idx * 4.8
    node = mpatches.FancyBboxPatch((bx, 0.5), 4.2, 3, boxstyle="round,pad=0.1",
                                   facecolor="#e8f4f8", edgecolor="black", linewidth=1.5)
    ax.add_patch(node)
    ax.text(bx + 2.1, 3.5, f"Node {node_idx}", ha="center", va="center", fontsize=12, fontweight="bold")
    for i in range(4):
        x = bx + 0.5 + i * 0.9
        r = node_idx * 4 + i
        ax.add_patch(mpatches.FancyBboxPatch((x, 1.0), 0.7, 0.9, boxstyle="round,pad=0.05",
                                              facecolor="#99bcff", edgecolor="black", linewidth=1))
        ax.text(x + 0.35, 1.42, f"GPU{i}", ha="center", va="center", fontsize=9)
        ax.text(x + 0.35, 1.02, f"RANK {r}", ha="center", va="bottom", fontsize=8)
        ax.text(x + 0.35, 0.72, f"LOCAL_RANK {i}", ha="center", va="top", fontsize=7, color="gray")
ax.text(5.0, 0.25, "Multi-node (2 machines, 8 GPUs)", ha="center", va="center", fontsize=11)

plt.tight_layout()
save_figure(__file__)
