"""
Virtual node setup: Multiple slurmd daemons on a single physical machine
simulating a multi-node cluster, with GPU mapping.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math4ai import save_figure

fig, ax = plt.subplots(figsize=(12, 5.5))
ax.set_axis_off()
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.4)

# Colors
physical_color = "#f5f5f5"
virtual_color = "#e8f4f8"
gpu_active_color = "#99bcff"
gpu_inactive_color = "#e0e0e0"
daemon_color = "#c3e6cb"

# Title
#ax.text(5.0, 5.7, "Virtual Multi-Node Cluster on Single Physical Machine", 
        #ha="center", va="center", fontsize=13, fontweight="bold")

# Physical machine box (large outer box)
physical_box = mpatches.FancyBboxPatch((0.3, 0.53), 9.4, 4.1, boxstyle="round,pad=0.1",
                                        facecolor=physical_color, edgecolor="black", linewidth=2)
ax.add_patch(physical_box)
ax.text(5.0, 5.05, "Physical Machine (8 GPUs)", ha="center", va="center", 
        fontsize=13, fontweight="bold")

# GPU row at bottom - balanced layout: GPU 0, 1, ..., 5, 6, 7
gpu_y = 0.6
gpu_width = 1.2
gpu_spacing = 1.4  # Wider spacing for balanced layout

# Define positions: 6 elements (GPU 0, 1, "...", 5, 6, 7) evenly distributed
# Total width needed: 6 * gpu_width + 5 * gap = ~9.0 (within 0.5 to 9.5)
gpu_items = [0, 1, "...", 5, 6, 7]
start_x = 0.7
for idx, item in enumerate(gpu_items):
    x = start_x + idx * gpu_spacing
    
    if item == "...":
        ax.text(x + gpu_width/2, gpu_y + 0.4, "...", ha="center", va="center", 
                fontsize=16, fontweight="bold", color="#666666")
        continue
    
    i = item  # GPU index
    # GPUs 6 and 7 are active (used by virtual nodes)
    if i in [6, 7]:
        color = gpu_active_color
        edge_width = 2
    else:
        color = gpu_inactive_color
        edge_width = 1
    
    gpu_box = mpatches.FancyBboxPatch((x, gpu_y), gpu_width, 0.8, boxstyle="round,pad=0.03",
                                       facecolor=color, edgecolor="black", linewidth=edge_width)
    ax.add_patch(gpu_box)
    ax.text(x + gpu_width/2, gpu_y + 0.4, f"GPU {i}", ha="center", va="center", fontsize=13)
    ax.text(x + gpu_width/2, gpu_y + 0.15, f"/dev/nvidia{i}", ha="center", va="center", 
            fontsize=13, family="monospace", color="#666666")

# Virtual Node 0 (node6)
vnode0_box = mpatches.FancyBboxPatch((0.8, 2.0), 3.8, 2.5, boxstyle="round,pad=0.08",
                                      facecolor=virtual_color, edgecolor="#007bff", linewidth=1.5)
ax.add_patch(vnode0_box)
ax.text(2.7, 4.2, "Virtual Node: node6", ha="center", va="center", 
        fontsize=13, fontweight="bold", color="#007bff")

# slurmd daemon for node6
daemon0_box = mpatches.FancyBboxPatch((1.2, 2.8), 1.6, 1.0, boxstyle="round,pad=0.05",
                                       facecolor=daemon_color, edgecolor="black", linewidth=1)
ax.add_patch(daemon0_box)
ax.text(2.0, 3.5, "slurmd", ha="center", va="center", fontsize=13, fontweight="bold", family="monospace")
ax.text(2.0, 3.1, "Port: 17016", ha="center", va="center", fontsize=13)

# Config info for node6
ax.text(3.6, 3.4, "Gres=gpu:1", ha="center", va="center", fontsize=13, family="monospace")
ax.text(3.6, 3.0, "→ GPU 6", ha="center", va="center", fontsize=13, fontweight="bold")

# Arrow from node6 to GPU 6 (GPU 6 is at index 4: start_x + 4*gpu_spacing = 0.7 + 4*1.4 = 6.3)
ax.annotate("", xy=(6.75, 1.4), xytext=(3.0, 2.0),
            arrowprops=dict(arrowstyle="->", color="#007bff", lw=1.5,
                           connectionstyle="arc3,rad=-0.15"))

# Virtual Node 1 (node7)
vnode1_box = mpatches.FancyBboxPatch((5.4, 2.0), 3.8, 2.5, boxstyle="round,pad=0.08",
                                      facecolor=virtual_color, edgecolor="#28a745", linewidth=1.5)
ax.add_patch(vnode1_box)
ax.text(7.3, 4.2, "Virtual Node: node7", ha="center", va="center", 
        fontsize=13, fontweight="bold", color="#28a745")

# slurmd daemon for node7
daemon1_box = mpatches.FancyBboxPatch((5.8, 2.8), 1.6, 1.0, boxstyle="round,pad=0.05",
                                       facecolor=daemon_color, edgecolor="black", linewidth=1)
ax.add_patch(daemon1_box)
ax.text(6.6, 3.5, "slurmd", ha="center", va="center", fontsize=13, fontweight="bold", family="monospace")
ax.text(6.6, 3.1, "Port: 17017", ha="center", va="center", fontsize=13)

# Config info for node7
ax.text(8.2, 3.4, "Gres=gpu:1", ha="center", va="center", fontsize=13, family="monospace")
ax.text(8.2, 3.0, "→ GPU 7", ha="center", va="center", fontsize=13, fontweight="bold")

# Arrow from node7 to GPU 7 (GPU 7 is at index 5: start_x + 5*gpu_spacing = 0.7 + 5*1.4 = 7.7)
ax.annotate("", xy=(8.15, 1.4), xytext=(7.6, 2.0),
            arrowprops=dict(arrowstyle="->", color="#28a745", lw=1.5,
                           connectionstyle="arc3,rad=-0.15"))

# Legend/note at bottom
ax.text(5.0, 0.15, "gres.conf mapping: NodeName=node6 → /dev/nvidia6, NodeName=node7 → /dev/nvidia7",
        ha="center", va="center", fontsize=13, family="monospace", 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd", edgecolor="#ffc107"))

plt.tight_layout()
save_figure(__file__)
