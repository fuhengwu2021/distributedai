"""
GPU Memory Hierarchy Architecture Diagram

This diagram visualizes the GPU memory hierarchy, showing the tradeoffs between
speed, capacity, and latency across different memory levels.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

# Add shared directory to path for math4ai imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from math4ai import configure_math_fonts, save_figure

# Configure matplotlib for math expressions
configure_math_fonts()

# Data for a generic modern high-end GPU (e.g., NVIDIA H100/A100 class)
levels = ["Registers", "L1 / Shared Memory", "L2 Cache", "VRAM (HBM/GDDR)"]
capacities = ["~256 KB/SM", "~128 KB/SM", "40-96 MB", "16-80+ GB"]
bandwidths = [">100 TB/s", "~20 TB/s", "~5 TB/s", "1-3.5 TB/s"]
latencies = ["~1 cycle", "~30 cycles", "~200 cycles", "400-800 cycles"]

# Create figure
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(0, 12)
ax.set_ylim(-0.5, 8.5)
ax.axis('off')

# Colors - from top (fastest) to bottom (largest)
colors = ['#ff4d4d', '#ff944d', '#4db8ff', '#4dff88']

# Reverse data to draw bottom-up
rev_levels = levels[::-1]
rev_caps = capacities[::-1]
rev_bws = bandwidths[::-1]
rev_lats = latencies[::-1]
rev_colors = colors[::-1]

for i in range(len(rev_levels)):
    y_base = i * 2
    # Width narrowing toward the top to simulate a pyramid
    width = 8 - (i * 1.2)
    x_start = (12 - width) / 2
    
    # Draw box
    rect = patches.FancyBboxPatch((x_start, y_base), width, 1.5, 
                                 boxstyle="round,pad=0.1",
                                 linewidth=2, edgecolor='black', 
                                 facecolor=rev_colors[i], alpha=0.8)
    ax.add_patch(rect)
    
    # Text labels - increased font sizes for better readability
    ax.text(6, y_base + 1.0, rev_levels[i], ha='center', va='center', 
            fontweight='bold', fontsize=16)  # Increased to 16pt for main labels
    spec_text = f"{rev_caps[i]}  |  {rev_bws[i]}  |  {rev_lats[i]}"
    ax.text(6, y_base + 0.5, spec_text, ha='center', va='center', 
            fontsize=13)  # Increased to 13pt, removed "Capacity:", "BW:", "Latency:" labels

# Directional Indicators
ax.annotate('', xy=(1.0, 7.5), xytext=(1.0, 0.5),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2.5))
ax.text(0.7, 4, 'Latency ↑', rotation=90, 
        va='center', ha='center', fontweight='bold', fontsize=13)  # Increased to 13pt, simplified text

# No title - captions are provided in markdown
plt.tight_layout()
save_figure(__file__)
