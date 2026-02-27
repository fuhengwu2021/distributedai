#!/usr/bin/env python3
"""
llm-d Deployment Architecture Diagram
Chapter 9: Production LLM Serving Stack

Shows the architecture of llm-d deployment on Kubernetes:
- Envoy Proxy as the entry point
- Inference Gateway (IGW) for intelligent scheduling
- Prefill and Decode servers running vLLM
- KV Cache storage for disaggregated inference
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os
import sys

# Add shared directory to path for math4ai imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from math4ai import configure_math_fonts

# Configure matplotlib for math expressions
configure_math_fonts()

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.set_aspect('equal')
ax.axis('off')

# Colors
cluster_color = '#E3F2FD'    # Light blue
proxy_color = '#FFF3E0'      # Light orange
gateway_color = '#E8F5E9'    # Light green
server_color = '#F3E5F5'     # Light purple
storage_color = '#FFFDE7'    # Light yellow
border_color = '#929292'     # Dark gray

# Kubernetes Cluster box
cluster_box = FancyBboxPatch(
    (0.3, 0.3), 9.4, 7.4,
    boxstyle="round,pad=0.02,rounding_size=0.15",
    facecolor=cluster_color,
    edgecolor=border_color,
    linewidth=1
)
ax.add_patch(cluster_box)
ax.text(5, 7.5, 'Kubernetes Cluster', ha='center', va='center', 
        fontsize=12, fontweight='bold')

# Envoy Proxy box
proxy_box = FancyBboxPatch(
    (1.0, 5.5), 2.2, 1.2,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    facecolor=proxy_color,
    edgecolor=border_color,
    linewidth=1.5
)
ax.add_patch(proxy_box)
ax.text(2.1, 6.3, 'Envoy Proxy', ha='center', va='center', 
        fontsize=11, fontweight='bold')
ax.text(2.1, 5.9, '(Entry Point)', ha='center', va='center', 
        fontsize=12, style='italic', color='#666666')

# Inference Gateway box
gateway_box = FancyBboxPatch(
    (4.5, 5.5), 2.8, 1.2,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    facecolor=gateway_color,
    edgecolor=border_color,
    linewidth=1.5
)
ax.add_patch(gateway_box)
ax.text(5.9, 6.3, 'Inference Gateway', ha='center', va='center', 
        fontsize=11, fontweight='bold')
ax.text(5.9, 5.9, '(IGW Scheduler)', ha='center', va='center', 
        fontsize=12, style='italic', color='#666666')

# Arrow from Proxy to Gateway
ax.annotate('', xy=(4.5, 6.1), xytext=(3.2, 6.1),
            arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))

# Prefill Server box
prefill_box = FancyBboxPatch(
    (1.0, 3.0), 2.0, 1.5,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    facecolor=server_color,
    edgecolor='#7B1FA2',
    linewidth=1.5
)
ax.add_patch(prefill_box)
ax.text(2.0, 4.1, 'Prefill Server', ha='center', va='center', 
        fontsize=11, fontweight='bold', color='#7B1FA2')
ax.text(2.0, 3.7, '(vLLM)', ha='center', va='center', 
        fontsize=12, color='#666666')
ax.text(2.0, 3.35, 'Prompt Processing', ha='center', va='center', 
        fontsize=11, style='italic', color='#888888')

# Decode Server 1 box
decode1_box = FancyBboxPatch(
    (4.0, 3.0), 2.0, 1.5,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    facecolor=server_color,
    edgecolor='#7B1FA2',
    linewidth=1.5
)
ax.add_patch(decode1_box)
ax.text(5.0, 4.1, 'Decode Server 1', ha='center', va='center', 
        fontsize=11, fontweight='bold', color='#7B1FA2')
ax.text(5.0, 3.7, '(vLLM)', ha='center', va='center', 
        fontsize=12, color='#666666')
ax.text(5.0, 3.35, 'Token Generation', ha='center', va='center', 
        fontsize=11, style='italic', color='#888888')

# Decode Server 2 box
decode2_box = FancyBboxPatch(
    (7.0, 3.0), 2.0, 1.5,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    facecolor=server_color,
    edgecolor='#7B1FA2',
    linewidth=1.5
)
ax.add_patch(decode2_box)
ax.text(8.0, 4.1, 'Decode Server 2', ha='center', va='center', 
        fontsize=11, fontweight='bold', color='#7B1FA2')
ax.text(8.0, 3.7, '(vLLM)', ha='center', va='center', 
        fontsize=12, color='#666666')
ax.text(8.0, 3.35, 'Token Generation', ha='center', va='center', 
        fontsize=11, style='italic', color='#888888')

# Arrows from Gateway to servers (IGW routes to all servers)
# Gateway box is at (4.5, 5.5) with size (2.8, 1.2), so bottom center is (5.9, 5.5)
gateway_bottom = 5.5
gateway_center_x = 5.9

# Arrow to Prefill Server (box at x=1.0-3.0, top at y=4.5)
ax.annotate('', xy=(2.0, 4.5), xytext=(gateway_center_x, gateway_bottom),
            arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5,
                           connectionstyle='arc3,rad=0.003'))

# Arrow to Decode Server 1 (box at x=4.0-6.0, top at y=4.5)
ax.annotate('', xy=(5.0, 4.5), xytext=(gateway_center_x, gateway_bottom),
            arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))

# Arrow to Decode Server 2 (box at x=7.0-9.0, top at y=4.5)
ax.annotate('', xy=(8.0, 4.5), xytext=(gateway_center_x, gateway_bottom),
            arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5,
                           connectionstyle='arc3,rad=-0.003'))

# KV Cache Storage box
storage_box = FancyBboxPatch(
    (3.0, 0.8), 4.0, 1.2,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    facecolor=storage_color,
    edgecolor='#F57F17',
    linewidth=1.5
)
ax.add_patch(storage_box)
ax.text(5.0, 1.6, 'KV Cache Storage', ha='center', va='center', 
        fontsize=11, fontweight='bold', color='#F57F17')
ax.text(5.0, 1.15, '(NIXL / NVMe)', ha='center', va='center', 
        fontsize=12, style='italic', color='#666666')

# Arrows from servers to storage
ax.annotate('', xy=(3.5, 2.0), xytext=(2.0, 3.0),
            arrowprops=dict(arrowstyle='<->', color='#F57F17', lw=1.2,
                           connectionstyle='arc3,rad=0.2'))
ax.annotate('', xy=(5.0, 2.0), xytext=(5.0, 3.0),
            arrowprops=dict(arrowstyle='<->', color='#F57F17', lw=1.2))
ax.annotate('', xy=(6.5, 2.0), xytext=(8.0, 3.0),
            arrowprops=dict(arrowstyle='<->', color='#F57F17', lw=1.2,
                           connectionstyle='arc3,rad=-0.2'))

# Request flow annotation
ax.text(7.5, 6.1, 'Requests', ha='left', va='center', 
        fontsize=12, color='#666666')
ax.annotate('', xy=(1.0, 6.1), xytext=(0.5, 6.1),
            arrowprops=dict(arrowstyle='->', color='#1976D2', lw=2))

# Legend
legend_y = 7.0
legend_items = [
    (1.5, legend_y, proxy_color, 'Entry Point'),
    (3.5, legend_y, gateway_color, 'Scheduler'),
    (5.5, legend_y, server_color, 'Model Servers'),
    (7.8, legend_y, storage_color, 'Storage'),
]

for x, y, color, label in legend_items:
    box = FancyBboxPatch(
        (x - 0.2, y - 0.12), 0.25, 0.25,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor=color,
        edgecolor=border_color,
        linewidth=1
    )
    ax.add_patch(box)
    ax.text(x + 0.2, y, label, ha='left', va='center', fontsize=12)

# Tight layout and save
plt.tight_layout(pad=0.1)

# Save figure following book conventions
script_dir = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.splitext(os.path.basename(__file__))[0]
output_path = os.path.join(script_dir, f'{script_name}.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0)
print(f"Saved figure to: {output_path}")

# Also save PDF
pdf_path = os.path.join(script_dir, f'{script_name}.pdf')
plt.savefig(pdf_path, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0)
print(f"Saved figure to: {pdf_path}")

plt.close()
