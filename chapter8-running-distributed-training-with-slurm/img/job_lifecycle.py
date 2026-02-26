"""
SLURM job lifecycle: PENDING -> RUNNING -> COMPLETING -> COMPLETED
with possible states like FAILED, CANCELLED, TIMEOUT.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared', 'math4ai'))
from figure_utils import save_figure

fig, ax = plt.subplots(figsize=(10, 4))
ax.set_axis_off()
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.5)

# Colors
pending_color = "#fff3cd"
running_color = "#d4edda"
completing_color = "#cce5ff"
completed_color = "#c3e6cb"
failed_color = "#f8d7da"
arrow_color = "#666666"

# Title
ax.text(5.0, 4.2, "SLURM Job State Lifecycle", ha="center", va="center", 
        fontsize=13, fontweight="bold")

# Main flow states (horizontal)
states = [
    (0.5, 2.2, "PENDING", pending_color, "Waiting for\nresources"),
    (2.7, 2.2, "RUNNING", running_color, "Executing\non nodes"),
    (4.9, 2.2, "COMPLETING", completing_color, "Cleaning up\nprocesses"),
    (7.1, 2.2, "COMPLETED", completed_color, "Finished\nsuccessfully"),
]

for x, y, label, color, desc in states:
    box = mpatches.FancyBboxPatch((x, y), 1.8, 1.4, boxstyle="round,pad=0.08",
                                   facecolor=color, edgecolor="black", linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + 0.9, y + 1.0, label, ha="center", va="center", 
            fontsize=10, fontweight="bold")
    ax.text(x + 0.9, y + 0.4, desc, ha="center", va="center", fontsize=8)

# Arrows between main states
for i in range(3):
    x_start = states[i][0] + 1.8
    x_end = states[i+1][0]
    y = 2.9
    ax.annotate("", xy=(x_end, y), xytext=(x_start, y),
                arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2))

# Alternative end states (bottom)
alt_states = [
    (4.9, 0.4, "FAILED", failed_color, "Error occurred"),
    (7.1, 0.4, "CANCELLED", "#e2e3e5", "User cancelled"),
    (2.7, 0.4, "TIMEOUT", "#ffeeba", "Time limit hit"),
]

for x, y, label, color, desc in alt_states:
    box = mpatches.FancyBboxPatch((x, y), 1.8, 1.0, boxstyle="round,pad=0.08",
                                   facecolor=color, edgecolor="black", linewidth=1)
    ax.add_patch(box)
    ax.text(x + 0.9, y + 0.65, label, ha="center", va="center", 
            fontsize=9, fontweight="bold")
    ax.text(x + 0.9, y + 0.25, desc, ha="center", va="center", fontsize=7)

# Arrows to alternative states
# RUNNING -> FAILED
ax.annotate("", xy=(5.8, 1.4), xytext=(5.8, 2.2),
            arrowprops=dict(arrowstyle="->", color="#dc3545", lw=1.5))

# RUNNING -> CANCELLED
ax.annotate("", xy=(8.0, 1.4), xytext=(4.5, 2.5),
            arrowprops=dict(arrowstyle="->", color="#6c757d", lw=1.2,
                           connectionstyle="arc3,rad=-0.3"))

# RUNNING -> TIMEOUT
ax.annotate("", xy=(3.6, 1.4), xytext=(3.6, 2.2),
            arrowprops=dict(arrowstyle="->", color="#856404", lw=1.2))

# sbatch annotation
ax.text(0.3, 3.8, "sbatch/srun", ha="left", va="center", fontsize=9, 
        family="monospace", style="italic")
ax.annotate("", xy=(1.4, 3.6), xytext=(1.4, 3.8),
            arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.5))

plt.tight_layout()
save_figure(__file__)
