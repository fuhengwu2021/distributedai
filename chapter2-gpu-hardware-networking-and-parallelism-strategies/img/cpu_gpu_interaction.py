"""
CPU-GPU Interaction Diagram

This diagram illustrates how CPU and GPU interact during distributed training,
showing kernel launches, memory transfers, and PCIe connections.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__)) if __file__ else "."
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from math4ai import save_figure
from cpu import draw_cpu_shape
from gpu import draw_gpu_shape

# Create figure
# Calculate optimal figure size based on actual axis limits
# Axis limits: x from 0.8 to 4.8 (width 4.0), y from 1.0 to 3.0 (height 2.0)
data_width = 4.0
data_height = 2.0
data_aspect = data_width / data_height

# Set figure size to match data aspect ratio
fig_height = 2.5
fig_width = fig_height * data_aspect
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.axis('off')
# Don't use 'equal' aspect - it causes padding issues with mismatched figure/data ratios

# Configuration
ICON_SIZE = 1.2
SPACING = 2.5
CENTER_Y = 2.0

# Positions
cpu_x = 1.5
gpu_x = cpu_x + SPACING

# Draw CPU
draw_cpu_shape(ax, center_x=cpu_x, center_y=CENTER_Y, scale=ICON_SIZE, linewidth=2.0, show_text=True, text_fontsize=ICON_SIZE*20)

# Draw GPU
draw_gpu_shape(ax, center_x=gpu_x, center_y=CENTER_Y, scale=ICON_SIZE, linewidth=2.0, show_text=True, text_fontsize=ICON_SIZE*20)

# PCIe Connection (bidirectional arrow)
arrow_start_x = cpu_x + ICON_SIZE * 0.6 # Adjusted to be closer to the chip edge
arrow_end_x = gpu_x - ICON_SIZE * 0.6 # Adjusted to be closer to the chip edge

conn_pcie = ConnectionPatch(
    (arrow_start_x, CENTER_Y), (arrow_end_x, CENTER_Y),
    "data", "data", arrowstyle='<->', mutation_scale=20,
    linewidth=2.5, color='#E67E22', zorder=2
)
ax.add_patch(conn_pcie)

# PCIe Label
ax.text((cpu_x + gpu_x) / 2, CENTER_Y + 0.25, 'PCIe Gen4/5',
        fontsize=10, ha='center', va='center', fontweight='bold', color='#E67E22',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#E67E22', linewidth=1), zorder=5)

# Data Flow Annotations
# 1. CPU launches GPU kernels (CPU -> GPU) - straight line
kernel_y = CENTER_Y + 0.8
ax.annotate('', xy=(gpu_x - 0.5, kernel_y), xytext=(cpu_x + 0.4, kernel_y),
            arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2, ls='--'))
ax.text((cpu_x + gpu_x) / 2, kernel_y - 0.1, 'Control: CUDA Kernels',
        fontsize=9, ha='center', va='top', color='#27AE60', style='italic')

# 2. Memory transfers (CPU <-> GPU) - straight lines
memory_y = CENTER_Y - 0.8
ax.annotate('', xy=(gpu_x - 0.5, memory_y), xytext=(cpu_x + 0.4, memory_y),
            arrowprops=dict(arrowstyle='<->', color='#8E44AD', lw=2, ls='--'))
ax.text((cpu_x + gpu_x) / 2, memory_y + 0.1, 'Data: Memory Transfers',
        fontsize=9, ha='center', va='bottom', color='#8E44AD', style='italic')

# Final Layout - set limits to tightly match actual content
# Left: CPU icon at x=1.5, with some margin for labels
# Right: GPU icon at x=4.0, with some margin for labels
# Bottom: Memory transfers text at y=1.2 (memory_y), text extends to ~1.1, with small margin
# Top: CUDA kernels text at y=2.7 (kernel_y - 0.1), kernel_y at 2.8, with small margin
ax.set_xlim(0.8, 4.8)
ax.set_ylim(1.0, 3.0)
plt.tight_layout(pad=0)
save_figure(__file__)