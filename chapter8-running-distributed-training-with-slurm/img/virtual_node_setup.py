"""
Virtual node setup: Multiple slurmd daemons on a single physical machine
simulating a multi-node cluster, with GPU mapping.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared', 'math4ai'))
from figure_utils import save_figure

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_axis_off()
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)

# Colors
physical_color = "#f5f5f5"
virtual_color = "#e8f4f8"
gpu_active_color = "#99bcff"
gpu_inactive_color = "#e0e0e0"
daemon_color = "#c3e6cb"

# Title
ax.text(5.0, 5.7, "Virtual Multi-Node Cluster on Single Physical Machine", 
        ha="center", va="center", fontsize=13, fontweight="bold")

# Physical machine box (large outer box)
physical_box = mpatches.FancyBboxPatch((0.3, 0.3), 9.4, 5.0, boxstyle="round,pad=0.1",
                                        facecolor=physical_color, edgecolor="black", linewidth=2)
ax.add_patch(physical_box)
ax.text(5.0, 5.05, "Physical Machine (8 GPUs)", ha="center", va="center", 
        fontsize=11, fontweight="bold")

# GPU row at bottom
gpu_y = 0.6
for i in range(8):
    x = 0.7 + i * 1.1
    # GPUs 6 and 7 are active (used by virtual nodes)
    if i in [6, 7]:
        color = gpu_active_color
        edge_width = 2
    else:
        color = gpu_inactive_color
        edge_width = 1
    
    gpu_box = mpatches.FancyBboxPatch((x, gpu_y), 0.9, 0.8, boxstyle="round,pad=0.03",
                                       facecolor=color, edgecolor="black", linewidth=edge_width)
    ax.add_patch(gpu_box)
    ax.text(x + 0.45, gpu_y + 0.4, f"GPU {i}", ha="center", va="center", fontsize=9)
    ax.text(x + 0.45, gpu_y + 0.15, f"/dev/nvidia{i}", ha="center", va="center", 
            fontsize=7, family="monospace", color="#666666")

# Virtual Node 0 (node6)
vnode0_box = mpatches.FancyBboxPatch((0.8, 2.0), 3.8, 2.5, boxstyle="round,pad=0.08",
                                      facecolor=virtual_color, edgecolor="#007bff", linewidth=1.5)
ax.add_patch(vnode0_box)
ax.text(2.7, 4.2, "Virtual Node: node6", ha="center", va="center", 
        fontsize=10, fontweight="bold", color="#007bff")

# slurmd daemon for node6
daemon0_box = mpatches.FancyBboxPatch((1.2, 2.8), 1.6, 1.0, boxstyle="round,pad=0.05",
                                       facecolor=daemon_color, edgecolor="black", linewidth=1)
ax.add_patch(daemon0_box)
ax.text(2.0, 3.5, "slurmd", ha="center", va="center", fontsize=9, fontweight="bold", family="monospace")
ax.text(2.0, 3.1, "Port: 17016", ha="center", va="center", fontsize=8)

# Config info for node6
ax.text(3.6, 3.4, "Gres=gpu:1", ha="center", va="center", fontsize=9, family="monospace")
ax.text(3.6, 3.0, "→ GPU 6", ha="center", va="center", fontsize=9, fontweight="bold")

# Arrow from node6 to GPU 6
ax.annotate("", xy=(7.35, 1.4), xytext=(3.0, 2.0),
            arrowprops=dict(arrowstyle="->", color="#007bff", lw=1.5,
                           connectionstyle="arc3,rad=0.2"))

# Virtual Node 1 (node7)
vnode1_box = mpatches.FancyBboxPatch((5.4, 2.0), 3.8, 2.5, boxstyle="round,pad=0.08",
                                      facecolor=virtual_color, edgecolor="#28a745", linewidth=1.5)
ax.add_patch(vnode1_box)
ax.text(7.3, 4.2, "Virtual Node: node7", ha="center", va="center", 
        fontsize=10, fontweight="bold", color="#28a745")

# slurmd daemon for node7
daemon1_box = mpatches.FancyBboxPatch((5.8, 2.8), 1.6, 1.0, boxstyle="round,pad=0.05",
                                       facecolor=daemon_color, edgecolor="black", linewidth=1)
ax.add_patch(daemon1_box)
ax.text(6.6, 3.5, "slurmd", ha="center", va="center", fontsize=9, fontweight="bold", family="monospace")
ax.text(6.6, 3.1, "Port: 17017", ha="center", va="center", fontsize=8)

# Config info for node7
ax.text(8.2, 3.4, "Gres=gpu:1", ha="center", va="center", fontsize=9, family="monospace")
ax.text(8.2, 3.0, "→ GPU 7", ha="center", va="center", fontsize=9, fontweight="bold")

# Arrow from node7 to GPU 7
ax.annotate("", xy=(8.45, 1.4), xytext=(7.6, 2.0),
            arrowprops=dict(arrowstyle="->", color="#28a745", lw=1.5,
                           connectionstyle="arc3,rad=-0.2"))

# Legend/note at bottom
ax.text(5.0, 1.65, "gres.conf mapping: NodeName=node6 → /dev/nvidia6, NodeName=node7 → /dev/nvidia7",
        ha="center", va="center", fontsize=8, family="monospace", 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd", edgecolor="#ffc107"))

plt.tight_layout()
save_figure(__file__)
