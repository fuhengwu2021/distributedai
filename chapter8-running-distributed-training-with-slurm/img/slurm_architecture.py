"""
SLURM Architecture: slurmctld, slurmd, slurmdbd and job flow.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from math4ai import save_figure

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_axis_off()
ax.set_xlim(0.2, 9.6)
ax.set_ylim(0, 6.2)

# Colors
head_color = "#e8f4f8"
compute_color = "#d4edda"
db_color = "#fff3cd"
user_color = "#f8d7da"
arrow_color = "#666666"

# User/Client box (left side)
user_box = mpatches.FancyBboxPatch((0.3, 2.5), 1.72, 1.45, boxstyle="round,pad=0.08",
                                    facecolor=user_color, edgecolor="black", linewidth=1.2)
ax.add_patch(user_box)
ax.text(1.2, 3.5, "User", ha="center", va="center", fontsize=11, fontweight="bold")
ax.text(1.2, 3.1, "sbatch\nsrun\nsqueue", ha="center", va="center", fontsize=12, family="monospace")

# Head Node with slurmctld
head_box = mpatches.FancyBboxPatch((2.85, 2.0), 2.25, 2.4, boxstyle="round,pad=0.08",
                                    facecolor=head_color, edgecolor="black", linewidth=1.2)
ax.add_patch(head_box)
ax.text(4.0, 4.2, "Head Node", ha="center", va="center", fontsize=11, fontweight="bold")

# slurmctld daemon box
ctld_box = mpatches.FancyBboxPatch((3.1, 2.3), 1.8, 1.5, boxstyle="round,pad=0.05",
                                    facecolor="#b8daff", edgecolor="black", linewidth=1)
ax.add_patch(ctld_box)
ax.text(4.0, 3.4, "slurmctld", ha="center", va="center", fontsize=13, fontweight="bold", family="monospace")
ax.text(4.0, 2.9, "Job Queue\nScheduler\nResource Mgr", ha="center", va="center", fontsize=12)

# Database with slurmdbd (bottom)
db_box = mpatches.FancyBboxPatch((3.3, 0.3), 1.4, 1.2, boxstyle="round,pad=0.05",
                                  facecolor=db_color, edgecolor="black", linewidth=1)
ax.add_patch(db_box)
ax.text(4.0, 1.1, "slurmdbd", ha="center", va="center", fontsize=12, fontweight="bold", family="monospace")
ax.text(4.0, 0.7, "Accounting", ha="center", va="center", fontsize=12)

# Compute Nodes (right side)
for i, (y_pos, node_name) in enumerate([(4.2, "Compute Node 0"), (2.2, "Compute Node 1"), (0.2, "Compute Node N")]):
    if i == 2:  # Add dots before last node
        ax.text(7.8, 1.6, "...", ha="center", va="center", fontsize=16, fontweight="bold")
    
    node_box = mpatches.FancyBboxPatch((6.0, y_pos), 3.5, 1.8, boxstyle="round,pad=0.08",
                                        facecolor=compute_color, edgecolor="black", linewidth=1.2)
    ax.add_patch(node_box)
    ax.text(7.75, y_pos + 1.55, node_name, ha="center", va="center", fontsize=13, fontweight="bold")
    
    # slurmd daemon
    slurmd_box = mpatches.FancyBboxPatch((6.3, y_pos + 0.2), 1.3, 1.0, boxstyle="round,pad=0.05",
                                          facecolor="#c3e6cb", edgecolor="black", linewidth=1)
    ax.add_patch(slurmd_box)
    ax.text(6.95, y_pos + 0.9, "slurmd", ha="center", va="center", fontsize=12, fontweight="bold", family="monospace")
    ax.text(6.95, y_pos + 0.5, "Tasks", ha="center", va="center", fontsize=12)
    
    # GPU boxes
    for j in range(2):
        gpu_box = mpatches.FancyBboxPatch((7.9 + j * 0.75, y_pos + 0.3), 0.65, 0.8, boxstyle="round,pad=0.03",
                                           facecolor="#99bcff", edgecolor="black", linewidth=0.8)
        ax.add_patch(gpu_box)
        ax.text(8.22 + j * 0.75, y_pos + 0.7, f"GPU{j}", ha="center", va="center", fontsize=12)

# Arrows
# User -> slurmctld
ax.annotate("", xy=(2.8, 3.25), xytext=(2.1, 3.25),
            arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.5))
ax.text(2.45, 3.5, "submit", ha="center", va="center", fontsize=10, color=arrow_color)

# slurmctld -> slurmd (multiple arrows)
for y_offset in [4.8, 2.8]:
    ax.annotate("", xy=(5.925, y_offset), xytext=(5.2, 3.25),
                arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.2,
                               connectionstyle="arc3,rad=-0.01"))

ax.text(5.4, 4.8, "allocate &\nlaunch", ha="center", va="center", fontsize=12, color=arrow_color)

# slurmctld <-> slurmdbd
ax.annotate("", xy=(4.0, 1.5), xytext=(4.0, 2.3),
            arrowprops=dict(arrowstyle="<->", color=arrow_color, lw=1.2))

# Title
#ax.text(5.0, 6.2, "SLURM Architecture", ha="center", va="center", fontsize=14, fontweight="bold")

plt.tight_layout()
save_figure(__file__)
