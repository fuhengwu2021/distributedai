"""
RadixAttention Radix Tree Example

Shows how three requests with a common prefix share KV cache through
a radix tree structure. The shared prefix "You are helpful. " is stored
once and reused by all three requests.
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import os
import sys

# Add shared directory to path for math4ai imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from math4ai import configure_math_fonts, save_figure

configure_math_fonts()

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0.7, 11.1)
ax.set_ylim(0, 6.7)
ax.set_aspect('equal')
ax.axis('off')

# Colors
root_color = '#E3F2FD'
shared_color = '#C8E6C9'
unique_color = '#FFF9C4'
arrow_color = '#666666'

def draw_node(ax, x, y, width, height, label, facecolor='white', edgecolor='black', fontsize=10):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height, 
                         boxstyle="round,pad=0.02,rounding_size=0.15",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, wrap=True)

def draw_edge(ax, start, end):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5,
                               connectionstyle='arc3,rad=0'))

# Root node
draw_node(ax, 6, 6, 1.5, 0.7, 'Root', facecolor=root_color, edgecolor='#1976D2', fontsize=13)

# Shared prefix node
draw_node(ax, 6, 4.2, 3.8, 0.9, '"You are helpful. "', facecolor=shared_color, edgecolor='#388E3C', fontsize=13)

# Edge from root to shared prefix
draw_edge(ax, (6, 5.65), (6, 4.65))

# Unique suffix nodes
draw_node(ax, 2.5, 2, 3.4, 0.8, '"What is Python?"', facecolor=unique_color, edgecolor='#F9A825', fontsize=13)
draw_node(ax, 6, 2, 2.8, 0.8, '"Explain ML."', facecolor=unique_color, edgecolor='#F9A825', fontsize=13)
draw_node(ax, 9.5, 2, 2.8, 0.8, '"Write code."', facecolor=unique_color, edgecolor='#F9A825', fontsize=13)

# Edges from shared prefix to unique suffixes
draw_edge(ax, (4.7, 3.75), (2.5, 2.4))
draw_edge(ax, (6, 3.75), (6, 2.4))
draw_edge(ax, (7.3, 3.75), (9.5, 2.4))

# Request labels
ax.text(2.5, 1.2, 'Request 1', ha='center', va='center', fontsize=13, color='#666666', style='italic')
ax.text(6, 1.2, 'Request 2', ha='center', va='center', fontsize=13, color='#666666', style='italic')
ax.text(9.5, 1.2, 'Request 3', ha='center', va='center', fontsize=13, color='#666666', style='italic')

# Legend
legend_y = 0.4
ax.add_patch(FancyBboxPatch((1.5, legend_y - 0.2), 0.4, 0.4, boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor=shared_color, edgecolor='#388E3C', linewidth=1))
ax.text(2.1, legend_y, 'Shared prefix (computed once)', ha='left', va='center', fontsize=13)

ax.add_patch(FancyBboxPatch((6.5, legend_y - 0.2), 0.4, 0.4, boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor=unique_color, edgecolor='#F9A825', linewidth=1))
ax.text(7.1, legend_y, 'Unique suffix (computed per request)', ha='left', va='center', fontsize=13)

plt.tight_layout(pad=0.1)
save_figure(__file__)
