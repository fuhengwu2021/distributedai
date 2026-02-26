"""
Multi-node distributed training with SLURM: showing torchrun/srun launching
processes across nodes with NCCL communication.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared', 'math4ai'))
from figure_utils import save_figure

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_axis_off()
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.5)

# Colors
node_color = "#e8f4f8"
gpu_color = "#99bcff"
process_color = "#d4edda"
nccl_color = "#ffc107"
slurm_color = "#f8d7da"

# Title
ax.text(5.0, 6.2, "Multi-Node Distributed Training with SLURM", 
        ha="center", va="center", fontsize=13, fontweight="bold")

# SLURM controller at top
slurm_box = mpatches.FancyBboxPatch((3.5, 5.0), 3.0, 0.9, boxstyle="round,pad=0.08",
                                     facecolor=slurm_color, edgecolor="black", linewidth=1.5)
ax.add_patch(slurm_box)
ax.text(5.0, 5.6, "SLURM", ha="center", va="center", fontsize=11, fontweight="bold")
ax.text(5.0, 5.2, "srun / torchrun", ha="center", va="center", fontsize=9, family="monospace")

# Two nodes
for node_idx, (node_x, node_name) in enumerate([(0.3, "Node 0 (MASTER_ADDR)"), (5.2, "Node 1")]):
    # Node box
    node_box = mpatches.FancyBboxPatch((node_x, 0.5), 4.5, 4.0, boxstyle="round,pad=0.1",
                                        facecolor=node_color, edgecolor="black", linewidth=1.5)
    ax.add_patch(node_box)
    ax.text(node_x + 2.25, 4.25, node_name, ha="center", va="center", 
            fontsize=10, fontweight="bold")
    
    # Two GPUs per node
    for gpu_idx in range(2):
        gpu_x = node_x + 0.4 + gpu_idx * 2.1
        
        # GPU box
        gpu_box = mpatches.FancyBboxPatch((gpu_x, 0.8), 1.8, 3.0, boxstyle="round,pad=0.05",
                                           facecolor=gpu_color, edgecolor="black", linewidth=1)
        ax.add_patch(gpu_box)
        
        # Global rank calculation
        global_rank = node_idx * 2 + gpu_idx
        
        # Process box inside GPU
        proc_box = mpatches.FancyBboxPatch((gpu_x + 0.15, 1.5), 1.5, 1.8, boxstyle="round,pad=0.05",
                                            facecolor=process_color, edgecolor="black", linewidth=0.8)
        ax.add_patch(proc_box)
        
        # Labels
        ax.text(gpu_x + 0.9, 3.55, f"GPU {gpu_idx}", ha="center", va="center", fontsize=9)
        ax.text(gpu_x + 0.9, 2.9, f"RANK={global_rank}", ha="center", va="center", 
                fontsize=9, fontweight="bold")
        ax.text(gpu_x + 0.9, 2.5, f"LOCAL_RANK={gpu_idx}", ha="center", va="center", 
                fontsize=8, color="#0066cc")
        ax.text(gpu_x + 0.9, 2.1, "train.py", ha="center", va="center", 
                fontsize=8, family="monospace")
        ax.text(gpu_x + 0.9, 1.0, f"Process {global_rank}", ha="center", va="center", fontsize=8)

# SLURM arrows to nodes
ax.annotate("", xy=(2.55, 4.5), xytext=(4.2, 5.0),
            arrowprops=dict(arrowstyle="->", color="#666666", lw=1.5,
                           connectionstyle="arc3,rad=0.2"))
ax.annotate("", xy=(7.45, 4.5), xytext=(5.8, 5.0),
            arrowprops=dict(arrowstyle="->", color="#666666", lw=1.5,
                           connectionstyle="arc3,rad=-0.2"))

# NCCL communication (horizontal double arrow between nodes)
ax.annotate("", xy=(5.2, 2.5), xytext=(4.8, 2.5),
            arrowprops=dict(arrowstyle="<->", color=nccl_color, lw=3))
ax.text(5.0, 2.9, "NCCL", ha="center", va="center", fontsize=9, fontweight="bold",
        color="#856404")
ax.text(5.0, 2.15, "AllReduce", ha="center", va="center", fontsize=8, color="#856404")

# WORLD_SIZE annotation
ax.text(5.0, 0.2, "WORLD_SIZE = 4 (2 nodes × 2 GPUs)", ha="center", va="center", 
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd", edgecolor="gray"))

plt.tight_layout()
save_figure(__file__)
