"""
SGLang Architecture Diagram

Shows the flow of requests through SGLang's runtime:
- Clients (SGLang Program + HTTP Client)
- API Server (entry point)
- SGLang Runtime (SRT) with Tokenizer, Request Queue, Scheduler, GPU Workers, Detokenizer
- API Server (returns responses)
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os
import sys

# Add shared directory to path for math4ai imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from math4ai import configure_math_fonts

configure_math_fonts()

fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.set_aspect('equal')
ax.axis('off')

# Colors
client_color = '#E8F4FD'
server_color = '#FFF3E0'
runtime_color = '#E8F5E9'
component_color = '#FFFFFF'
gpu_color = '#FFECB3'
arrow_color = '#666666'

def draw_box(ax, x, y, width, height, label, facecolor='white', edgecolor='black', fontsize=11, bold=False):
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', fontsize=fontsize, weight=weight)

def draw_arrow(ax, start, end):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))

# Clients section (top)
ax.add_patch(FancyBboxPatch((1, 11.5), 8, 2, boxstyle="round,pad=0.02,rounding_size=0.2",
                            facecolor=client_color, edgecolor='#1976D2', linewidth=2))
ax.text(5, 13.2, 'Clients', ha='center', va='center', fontsize=12, weight='bold', color='#1976D2')

# Client boxes inside
draw_box(ax, 1.5, 11.8, 3, 1.2, 'SGLang Program\n+ Interpreter', facecolor=component_color, edgecolor='#1976D2', fontsize=10)
draw_box(ax, 5.5, 11.8, 3, 1.2, 'HTTP Client', facecolor=component_color, edgecolor='#1976D2', fontsize=10)

# Arrow from clients to API Server
draw_arrow(ax, (5, 11.5), (5, 10.7))

# API Server (entry)
draw_box(ax, 2.5, 9.8, 5, 0.8, 'API Server', facecolor=server_color, edgecolor='#F57C00', fontsize=11, bold=True)
ax.text(7.8, 10.2, 'Entry point', ha='left', va='center', fontsize=9, color='#666666', style='italic')

# Arrow to SRT
draw_arrow(ax, (5, 9.8), (5, 9.3))

# SGLang Runtime (SRT) - main box
ax.add_patch(FancyBboxPatch((1.5, 2.2), 7, 7, boxstyle="round,pad=0.02,rounding_size=0.2",
                            facecolor=runtime_color, edgecolor='#388E3C', linewidth=2))
ax.text(5, 9.0, 'SGLang Runtime (SRT)', ha='center', va='center', fontsize=12, weight='bold', color='#388E3C')

# Tokenizer
draw_box(ax, 3.5, 7.8, 3, 0.7, 'Tokenizer', facecolor=component_color, edgecolor='#388E3C', fontsize=10)
ax.text(6.8, 8.15, r'Text $\to$ Tokens', ha='left', va='center', fontsize=9, color='#666666', style='italic')

draw_arrow(ax, (5, 7.8), (5, 7.3))

# Request Queue
draw_box(ax, 3.5, 6.5, 3, 0.7, 'Request Queue', facecolor=component_color, edgecolor='#388E3C', fontsize=10)
ax.text(6.8, 6.85, 'Batches requests', ha='left', va='center', fontsize=9, color='#666666', style='italic')

draw_arrow(ax, (5, 6.5), (5, 6.0))

# Scheduler
draw_box(ax, 3.5, 5.0, 3, 0.9, 'Scheduler\nRadixAttention', facecolor=component_color, edgecolor='#388E3C', fontsize=10)
ax.text(6.8, 5.45, 'Intelligent batching', ha='left', va='center', fontsize=9, color='#666666', style='italic')

draw_arrow(ax, (5, 5.0), (5, 4.5))

# GPU Workers
draw_box(ax, 2.5, 3.4, 5, 1.0, r'GPU Workers' + '\n' + r'W0 $\rightarrow$ W1 $\rightarrow$ W2 $\rightarrow$ W3', facecolor=gpu_color, edgecolor='#FFA000', fontsize=10)
ax.text(7.8, 3.9, 'Model execution', ha='left', va='center', fontsize=9, color='#666666', style='italic')

draw_arrow(ax, (5, 3.4), (5, 2.9))

# Detokenizer
draw_box(ax, 3.5, 2.4, 3, 0.7, 'Detokenizer', facecolor=component_color, edgecolor='#388E3C', fontsize=10)
ax.text(6.8, 2.75, r'Tokens $\to$ Text', ha='left', va='center', fontsize=9, color='#666666', style='italic')

# Arrow from SRT to API Server (bottom)
draw_arrow(ax, (5, 2.2), (5, 1.7))

# API Server (exit)
draw_box(ax, 2.5, 0.8, 5, 0.8, 'API Server', facecolor=server_color, edgecolor='#F57C00', fontsize=11, bold=True)
ax.text(7.8, 1.2, 'Returns responses', ha='left', va='center', fontsize=9, color='#666666', style='italic')

# Save figure
plt.tight_layout(pad=0.1)
script_dir = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.splitext(os.path.basename(__file__))[0]
output_path = os.path.join(script_dir, f'{script_name}.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.02)
print(f"Saved figure to: {output_path}")

# Also save PDF
pdf_path = os.path.join(script_dir, f'{script_name}.pdf')
plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.02)
print(f"Saved figure to: {pdf_path}")
