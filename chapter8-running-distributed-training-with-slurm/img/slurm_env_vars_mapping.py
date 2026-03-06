"""
SLURM environment variables mapping to PyTorch distributed concepts.
Shows how SLURM_PROCID -> RANK, SLURM_LOCALID -> LOCAL_RANK, etc.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math4ai import save_figure

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_axis_off()
ax.set_xlim(0.3,9.7)
#ax.set_ylim(0, 4.9)

# Colors
slurm_color = "#e8f4f8"
pytorch_color = "#d4edda"
arrow_color = "#666666"

# Title
#ax.text(5.0, 5.2, "SLURM → PyTorch Environment Variable Mapping", 
#        ha="center", va="center", fontsize=13, fontweight="bold")

# SLURM side (left)
slurm_box = mpatches.FancyBboxPatch((0.5, 0.5), 3.5, 4.2, boxstyle="round,pad=0.1",
                                     facecolor=slurm_color, edgecolor="black", linewidth=1.5)
ax.add_patch(slurm_box)
ax.text(2.25, 4.4, "SLURM Variables", ha="center", va="center", fontsize=12, fontweight="bold")

slurm_vars = [
    ("SLURM_PROCID", "Global process ID"),
    ("SLURM_LOCALID", "Local ID within node"),
    ("SLURM_NTASKS", "Total number of tasks"),
    ("SLURM_NODEID", "Node index (0 to N-1)"),
    ("SLURM_JOB_NODELIST", "List of allocated nodes"),
]

for i, (var, desc) in enumerate(slurm_vars):
    y = 3.8 - i * 0.7
    ax.text(0.7, y, var, ha="left", va="center", fontsize=13, fontweight="bold", family="monospace")
    ax.text(0.7, y - 0.25, desc, ha="left", va="center", fontsize=12, color="#555555")

# PyTorch side (right)
pytorch_box = mpatches.FancyBboxPatch((6.0, 0.5), 3.5, 4.2, boxstyle="round,pad=0.1",
                                       facecolor=pytorch_color, edgecolor="black", linewidth=1.5)
ax.add_patch(pytorch_box)
ax.text(7.75, 4.4, "PyTorch Variables", ha="center", va="center", fontsize=12, fontweight="bold")

pytorch_vars = [
    ("RANK", "Global rank"),
    ("LOCAL_RANK", "Rank within node"),
    ("WORLD_SIZE", "Total processes"),
    ("—", "(derived from above)"),
    ("MASTER_ADDR", "Master node address"),
]

for i, (var, desc) in enumerate(pytorch_vars):
    y = 3.8 - i * 0.7
    ax.text(6.2, y, var, ha="left", va="center", fontsize=13, fontweight="bold", family="monospace")
    ax.text(6.2, y - 0.25, desc, ha="left", va="center", fontsize=12, color="#555555")

# Arrows showing mapping
arrow_mappings = [
    (3, 3.8, 5.9, 3.8),   # SLURM_PROCID -> RANK
    (3, 3.1, 5.9, 3.1),   # SLURM_LOCALID -> LOCAL_RANK
    (3, 2.4, 5.9, 2.4),   # SLURM_NTASKS -> WORLD_SIZE
    (3, 1.0, 5.9, 1.0),   # SLURM_JOB_NODELIST -> MASTER_ADDR
]

for x1, y1, x2, y2 in arrow_mappings:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.5))

# Center label
ax.text(5, 1.7, "export or torchrun", ha="center", va="center", fontsize=13, 
        color=arrow_color, style="italic")

# Example at bottom
example_box = mpatches.FancyBboxPatch((0.5, -0.3), 9.0, 0.6, boxstyle="round,pad=0.05",
                                       facecolor="#fff3cd", edgecolor="#ffc107", linewidth=1)
ax.add_patch(example_box)
ax.text(5.0, 0.0, "Example: SLURM_PROCID=3, SLURM_NTASKS=8 → RANK=3, WORLD_SIZE=8",
        ha="center", va="center", fontsize=13, family="monospace")

ax.set_ylim(-0.5, 4.9)
plt.tight_layout()
save_figure(__file__)
