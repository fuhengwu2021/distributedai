"""
Generate TTFT (Time to First Token) pipeline diagram.
Shows the stages from input to first output token.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

from math4ai import save_figure

# Set up figure with white background
fig, ax = plt.subplots(1, 1, figsize=(14, 4))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Remove axes
ax.set_xlim(0, 14)
ax.set_ylim(0, 4)
ax.axis('off')

# Colors
purple = '#8B5CF6'
pink = '#F5A5A5'
green_light = '#90EE90'
green_text = '#16A34A'
red_arrow = '#DC2626'
text_dark = '#1a1a1a'

def draw_box(ax, x, y, width, height, text, facecolor, textcolor='white', fontsize=11):
    """Draw a rounded rectangle with text."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=facecolor,
        edgecolor=facecolor,
        linewidth=2
    )
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', fontsize=fontsize,
            color=textcolor, fontweight='bold')

def draw_arrow(ax, x_start, x_end, y):
    """Draw a horizontal arrow."""
    ax.annotate('', xy=(x_end, y), xytext=(x_start, y),
                arrowprops=dict(arrowstyle='->', color=red_arrow, lw=2.5))

# Y positions
y_center = 1.5
box_height = 1.0

# Stage 1: Tokenization
draw_box(ax, 0.5, y_center, 1.8, box_height, 'Tokenization', purple)

# Arrow 1
draw_arrow(ax, 2.4, 3.2, y_center + box_height/2)

# Stage 2: Main processing box (contains Prefill and Decode)
# Outer container
outer_box = FancyBboxPatch(
    (3.3, y_center - 0.3), 5.0, box_height + 0.6,
    boxstyle="round,pad=0.02,rounding_size=0.2",
    facecolor='#f5f5f5',
    edgecolor='#404040',
    linewidth=2
)
ax.add_patch(outer_box)

# Prefill box inside
prefill_box = FancyBboxPatch(
    (3.6, y_center), 2.2, box_height,
    boxstyle="round,pad=0.02,rounding_size=0.15",
    facecolor=pink,
    edgecolor=pink,
    linewidth=2
)
ax.add_patch(prefill_box)
ax.text(3.6 + 2.2/2, y_center + box_height/2 + 0.15, 'Initial Prompt', 
        ha='center', va='center', fontsize=10, color='#1a1a1a', fontweight='bold')
ax.text(3.6 + 2.2/2, y_center + box_height/2 - 0.15, 'Processing', 
        ha='center', va='center', fontsize=10, color='#1a1a1a', fontweight='bold')
ax.text(3.6 + 2.2/2, y_center + box_height/2 - 0.45, '(Prefill)', 
        ha='center', va='center', fontsize=9, color='#1a1a1a', fontweight='normal')

# Decode box inside (smaller, green)
decode_box = FancyBboxPatch(
    (6.0, y_center + 0.15), 0.5, box_height - 0.3,
    boxstyle="round,pad=0.02,rounding_size=0.1",
    facecolor=green_light,
    edgecolor=green_light,
    linewidth=2
)
ax.add_patch(decode_box)

# Label below decode box
ax.text(6.25, y_center - 0.15, 'decode/generation', 
        ha='center', va='top', fontsize=9, color=green_text, fontweight='normal')

# Arrow 2
draw_arrow(ax, 8.4, 9.2, y_center + box_height/2)

# Stage 3: De-Tokenization
draw_box(ax, 9.3, y_center, 2.0, box_height, 'De-Tokenization', purple)

# Arrow 3
draw_arrow(ax, 11.4, 12.2, y_center + box_height/2)

# Stage 4: First Output Token (text only)
ax.text(13.0, y_center + box_height/2, 'First Output Token', 
        ha='center', va='center', fontsize=12, color=green_text, fontweight='bold')

# Title
ax.text(7.0, 3.5, 'Time to First Token (TTFT)', 
        ha='center', va='center', fontsize=14, color=text_dark, fontweight='bold')

plt.tight_layout()
save_figure(__file__)
