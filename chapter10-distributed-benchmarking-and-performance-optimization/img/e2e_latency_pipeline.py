"""
Generate End-to-End Request Latency (e2e_latency) pipeline diagram.
Shows the complete request lifecycle from tokenization to final output.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np

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
gray_border = '#404040'

def draw_box(ax, x, y, width, height, text, facecolor, textcolor='white', fontsize=11, multiline=None):
    """Draw a rounded rectangle with text."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=facecolor,
        edgecolor=facecolor,
        linewidth=2
    )
    ax.add_patch(box)
    if multiline:
        for i, line in enumerate(multiline):
            offset = (len(multiline) - 1) / 2 - i
            ax.text(x + width/2, y + height/2 + offset * 0.25, line, 
                    ha='center', va='center', fontsize=fontsize,
                    color=textcolor, fontweight='bold')
    else:
        ax.text(x + width/2, y + height/2, text, 
                ha='center', va='center', fontsize=fontsize,
                color=textcolor, fontweight='bold')

def draw_arrow(ax, x_start, x_end, y):
    """Draw a horizontal arrow."""
    ax.annotate('', xy=(x_end, y), xytext=(x_start, y),
                arrowprops=dict(arrowstyle='->', color=red_arrow, lw=2.5))

# Y positions
y_center = 2.0
box_height = 1.0

# Stage 1: Tokenization
draw_box(ax, 0.5, y_center - box_height/2, 1.8, box_height, 'Tokenization', purple)

# Arrow 1
draw_arrow(ax, 2.4, 3.2, y_center)

# Stage 2: Main processing box (contains Prefill and Decode tokens)
# Outer container
outer_box = FancyBboxPatch(
    (3.3, y_center - box_height/2 - 0.3), 5.5, box_height + 0.6,
    boxstyle="round,pad=0.02,rounding_size=0.2",
    facecolor='#f5f5f5',
    edgecolor=gray_border,
    linewidth=2
)
ax.add_patch(outer_box)

# Prefill box inside
prefill_box = FancyBboxPatch(
    (3.6, y_center - box_height/2), 2.2, box_height,
    boxstyle="round,pad=0.02,rounding_size=0.15",
    facecolor=pink,
    edgecolor=pink,
    linewidth=2
)
ax.add_patch(prefill_box)
ax.text(3.6 + 2.2/2, y_center + 0.15, 'Initial Prompt', 
        ha='center', va='center', fontsize=10, color=text_dark, fontweight='bold')
ax.text(3.6 + 2.2/2, y_center - 0.1, 'Processing', 
        ha='center', va='center', fontsize=10, color=text_dark, fontweight='bold')
ax.text(3.6 + 2.2/2, y_center - 0.35, '(Prefill)', 
        ha='center', va='center', fontsize=9, color=text_dark, fontweight='normal')

# Multiple decode tokens inside (6 tokens)
token_start_x = 6.0
token_width = 0.35
token_spacing = 0.45
num_tokens = 6

for i in range(num_tokens):
    token_x = token_start_x + i * token_spacing
    token_box = FancyBboxPatch(
        (token_x, y_center - box_height/2 + 0.15), token_width, box_height - 0.3,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=green_light,
        edgecolor=green_light,
        linewidth=1.5
    )
    ax.add_patch(token_box)

# Label below decode tokens
ax.text(token_start_x + (num_tokens - 1) * token_spacing / 2 + token_width/2, 
        y_center - box_height/2 - 0.15, 'decode/generation', 
        ha='center', va='top', fontsize=9, color=green_text, fontweight='normal')

# Arrow 2
draw_arrow(ax, 8.9, 9.7, y_center)

# Stage 3: De-Tokenization
draw_box(ax, 9.8, y_center - box_height/2, 2.0, box_height, 'De-Tokenization', purple)

# Arrow 3
draw_arrow(ax, 11.9, 12.7, y_center)

# Output arrow indicator
ax.annotate('', xy=(13.5, y_center), xytext=(12.7, y_center),
            arrowprops=dict(arrowstyle='->', color=red_arrow, lw=2.5))

# Title
ax.text(7.0, 3.7, 'End-to-End Request Latency (e2e_latency)', 
        ha='center', va='center', fontsize=14, color=text_dark, fontweight='bold')

plt.tight_layout()
plt.savefig('img/e2e_latency_pipeline.png', dpi=150, facecolor='white', 
            edgecolor='none', bbox_inches='tight', pad_inches=0.2)
plt.savefig('img/e2e_latency_pipeline.pdf', facecolor='white', 
            edgecolor='none', bbox_inches='tight', pad_inches=0.2)
plt.close()

print("Generated: img/e2e_latency_pipeline.png and img/e2e_latency_pipeline.pdf")
