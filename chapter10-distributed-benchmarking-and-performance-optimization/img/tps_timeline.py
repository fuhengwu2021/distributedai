"""
Generate Tokens Per Second (TPS) timeline diagram.
Shows how TPS is calculated across multiple concurrent requests.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

from math4ai import save_figure

# Set up figure with white background
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Remove axes
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
ax.axis('off')

# Colors
line_color = '#6B7280'
circle_color = '#9CA3AF'
text_color = '#374151'
dashed_color = '#9CA3AF'

# Y positions
y_main = 4.0
y_spacing = 0.6

# Main timeline
ax.plot([0.5, 11.5], [y_main, y_main], color=line_color, linewidth=1.5, zorder=1)

# Timeline markers
markers = [
    (0.5, 'T_start'),
    (1.5, 'Tx'),
    (9.5, 'Ty'),
    (11.5, 'T_end')
]

for x, label in markers:
    # Circle marker
    circle = Circle((x, y_main), 0.08, facecolor='white', edgecolor=circle_color, linewidth=1.5, zorder=2)
    ax.add_patch(circle)
    # Label above
    ax.text(x, y_main + 0.3, label, ha='center', va='bottom', fontsize=10, color=text_color)

# Dashed vertical lines at Tx and Ty
ax.plot([1.5, 1.5], [y_main, 0.5], color=dashed_color, linewidth=1, linestyle='--', zorder=0)
ax.plot([9.5, 9.5], [y_main, 0.5], color=dashed_color, linewidth=1, linestyle='--', zorder=0)

# Request lines (L1, L2, ..., Ln-1, Ln)
requests = [
    (1.5, 3.0, 'L1', y_main - y_spacing),
    (2.0, 4.0, 'L2', y_main - 2*y_spacing),
    (7.5, 9.0, 'Ln-1', y_main - 3*y_spacing),
    (8.5, 10.5, 'Ln', y_main - 4*y_spacing),
]

for x_start, x_end, label, y in requests:
    # Vertical line down from main timeline
    ax.plot([x_start, x_start], [y_main, y], color=line_color, linewidth=1, zorder=1)
    # Horizontal line for request duration
    ax.plot([x_start, x_end], [y, y], color=line_color, linewidth=1, zorder=1)
    # Circle at start (on main timeline)
    circle_start = Circle((x_start, y_main), 0.08, facecolor='white', edgecolor=circle_color, linewidth=1.5, zorder=2)
    ax.add_patch(circle_start)
    # Circle at end of horizontal line
    circle_end = Circle((x_end, y), 0.08, facecolor='white', edgecolor=circle_color, linewidth=1.5, zorder=2)
    ax.add_patch(circle_end)
    # Label
    ax.text(x_end + 0.2, y + 0.15, label, ha='left', va='bottom', fontsize=10, color=text_color)

# Ellipsis in the middle
ax.text(5.5, y_main - 2.5*y_spacing, '...', ha='center', va='center', fontsize=14, color=text_color)

# Title
ax.text(6.0, 4.8, 'Tokens Per Second (TPS) Timeline', 
        ha='center', va='center', fontsize=12, color=text_color, fontweight='bold')

plt.tight_layout()
save_figure(__file__)
