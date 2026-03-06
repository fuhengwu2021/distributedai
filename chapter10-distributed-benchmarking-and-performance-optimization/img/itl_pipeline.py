"""
Generate Inter-token Latency (ITL) pipeline diagram.
Shows the decode/generation phase producing multiple output tokens.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np

from math4ai import save_figure

# Set up figure with white background
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Remove axes
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
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
y_upper = 4.5
box_height = 1.0

# Stage 1: Tokenization
draw_box(ax, 0.5, y_upper - box_height/2, 1.8, box_height, 'Tokenization', purple)

# Arrow 1
draw_arrow(ax, 2.4, 3.2, y_upper)

# Stage 2: Main processing box (contains Prefill and Decode tokens)
# Outer container
outer_box = FancyBboxPatch(
    (3.3, y_upper - box_height/2 - 0.3), 5.5, box_height + 0.6,
    boxstyle="round,pad=0.02,rounding_size=0.2",
    facecolor='#f5f5f5',
    edgecolor=gray_border,
    linewidth=2
)
ax.add_patch(outer_box)

# Prefill box inside
prefill_box = FancyBboxPatch(
    (3.6, y_upper - box_height/2), 2.2, box_height,
    boxstyle="round,pad=0.02,rounding_size=0.15",
    facecolor=pink,
    edgecolor=pink,
    linewidth=2
)
ax.add_patch(prefill_box)
ax.text(3.6 + 2.2/2, y_upper + 0.15, 'Initial Prompt', 
        ha='center', va='center', fontsize=10, color=text_dark, fontweight='bold')
ax.text(3.6 + 2.2/2, y_upper - 0.1, 'Processing', 
        ha='center', va='center', fontsize=10, color=text_dark, fontweight='bold')
ax.text(3.6 + 2.2/2, y_upper - 0.35, '(Prefill)', 
        ha='center', va='center', fontsize=9, color=text_dark, fontweight='normal')

# Multiple decode tokens inside (6 tokens)
token_start_x = 6.0
token_width = 0.35
token_spacing = 0.45
num_tokens = 6

for i in range(num_tokens):
    token_x = token_start_x + i * token_spacing
    token_box = FancyBboxPatch(
        (token_x, y_upper - box_height/2 + 0.15), token_width, box_height - 0.3,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=green_light,
        edgecolor=green_light,
        linewidth=1.5
    )
    ax.add_patch(token_box)

# Label below decode tokens
ax.text(token_start_x + (num_tokens - 1) * token_spacing / 2 + token_width/2, 
        y_upper - box_height/2 - 0.15, 'decode/generation', 
        ha='center', va='top', fontsize=9, color=green_text, fontweight='normal')

# Arrow 2
draw_arrow(ax, 8.9, 9.7, y_upper)

# Stage 3: De-Tokenization
draw_box(ax, 9.8, y_upper - box_height/2, 2.0, box_height, 'De-Tokenization', purple)

# Arrow 3
draw_arrow(ax, 11.9, 12.7, y_upper)

# Output arrow indicator
ax.annotate('', xy=(13.5, y_upper), xytext=(12.7, y_upper),
            arrowprops=dict(arrowstyle='->', color=red_arrow, lw=2.5))

# ============ Lower section: Output tokens with ITL ============
y_lower = 1.5
output_token_width = 0.5
output_token_height = 1.2
output_token_spacing = 0.7
output_start_x = 7.0

# Output tokens label
ax.text(output_start_x + 3 * output_token_spacing, y_lower + output_token_height + 0.4, 
        'Output Tokens', ha='center', va='bottom', fontsize=11, color=text_dark, fontweight='bold')

# Draw output tokens
for i in range(num_tokens):
    token_x = output_start_x + i * output_token_spacing
    token_box = FancyBboxPatch(
        (token_x, y_lower), output_token_width, output_token_height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=green_light,
        edgecolor='#6B8E6B',
        linewidth=1.5
    )
    ax.add_patch(token_box)

# ITL brackets between tokens
for i in range(num_tokens - 1):
    x1 = output_start_x + i * output_token_spacing + output_token_width
    x2 = output_start_x + (i + 1) * output_token_spacing
    y_bracket = y_lower - 0.3
    
    # Horizontal line with end caps
    ax.plot([x1 + 0.05, x2 - 0.05], [y_bracket, y_bracket], color=text_dark, linewidth=1.5)
    # Left cap
    ax.plot([x1 + 0.05, x1 + 0.05], [y_bracket - 0.08, y_bracket + 0.08], color=text_dark, linewidth=1.5)
    # Right cap
    ax.plot([x2 - 0.05, x2 - 0.05], [y_bracket - 0.08, y_bracket + 0.08], color=text_dark, linewidth=1.5)

# ITL label
ax.text(output_start_x + 2.5 * output_token_spacing + output_token_width/2, y_lower - 0.6, 
        'Inter-token Latency (ITL)', ha='center', va='top', fontsize=10, color=text_dark, fontweight='normal')

# Title
ax.text(7.0, 5.7, 'Inter-token Latency (ITL)', 
        ha='center', va='center', fontsize=14, color=text_dark, fontweight='bold')

plt.tight_layout()
save_figure(__file__)

