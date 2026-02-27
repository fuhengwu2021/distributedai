#!/usr/bin/env python3
"""
Production LLM Serving Architecture Diagram
Chapter 9: Production LLM Serving Stack

Shows the high-level architecture of a production LLM serving system:
- Clients at the top
- API Gateway for routing, auth, rate limiting
- Tokenizer Service and Model Runners (vLLM/SGLang)
- Monitoring & Observability at the bottom
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os
import sys

# Add shared directory to path for math4ai imports
from math4ai import configure_math_fonts

# Configure matplotlib for math expressions
configure_math_fonts()

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(1, 10)
ax.set_ylim(2, 9)
ax.set_aspect('equal')
ax.axis('off')

# Colors
client_color = '#E3F2FD'      # Light blue
gateway_color = '#FFF3E0'     # Light orange
service_color = '#E8F5E9'     # Light green
model_color = '#F3E5F5'       # Light purple
monitoring_color = '#FFFDE7'  # Light yellow
border_color = '#424242'      # Dark gray

def draw_box(ax, x, y, width, height, label, sublabel=None, color='white'):
    """Draw a rounded box with label"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color,
        edgecolor=border_color,
        linewidth=1.5
    )
    ax.add_patch(box)
    
    if sublabel:
        ax.text(x, y + 0.15, label, ha='center', va='center', fontsize=13, fontweight='bold')
        ax.text(x, y - 0.2, sublabel, ha='center', va='center', fontsize=13, style='italic', color='#666666')
    else:
        ax.text(x, y, label, ha='center', va='center', fontsize=13, fontweight='bold')

def draw_arrow(ax, start, end, color='#666666'):
    """Draw an arrow between two points"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                               connectionstyle='arc3,rad=0'))

# Clients (top)
draw_box(ax, 5, 8.5, 2.5, 0.7, 'Clients', color=client_color)

# Arrow from clients to gateway
draw_arrow(ax, (5, 8.1), (5, 7.4))

# API Gateway
draw_box(ax, 5, 7, 3.5, 0.7, 'API Gateway', 'Routing, Auth, Rate Limiting', color=gateway_color)

# Arrows from gateway to services
draw_arrow(ax, (3.5, 6.6), (2.5, 5.9))
draw_arrow(ax, (5, 6.6), (5, 5.9))
draw_arrow(ax, (6.5, 6.6), (7.5, 5.9))

# Tokenizer Service
draw_box(ax, 2.5, 5.5, 2.2, 0.7, 'Tokenizer', 'Service', color=service_color)

# Model Runners
draw_box(ax, 5, 5.5, 2.2, 0.7, 'Model Runner', '(vLLM/SGLang)', color=model_color)
draw_box(ax, 7.5, 5.5, 2.2, 0.7, 'Model Runner', '(vLLM/SGLang)', color=model_color)

# GPU indicators under model runners
ax.text(5, 4.95, '[GPU]', ha='center', va='center', fontsize=13, color='#666666')
ax.text(7.5, 4.95, '[GPU]', ha='center', va='center', fontsize=13, color='#666666')

# Arrows to monitoring (dashed lines from all services)
for x in [2.5, 5, 7.5]:
    ax.plot([x, x], [5.1, 4.2], '--', color='#999999', lw=1)
    
# Horizontal line connecting to monitoring
ax.plot([2.5, 7.5], [4.2, 4.2], '--', color='#999999', lw=1)
ax.plot([5, 5], [4.2, 3.6], '--', color='#999999', lw=1)

# Monitoring & Observability
draw_box(ax, 5, 3.2, 3.5, 0.7, 'Monitoring & Observability', 'Prometheus, OpenTelemetry, Logging', color=monitoring_color)

# Add component descriptions on the side
descriptions = [
    (2.5, 7, "Request\nRouting"),
    (9.5, 5.5, "Inference\nEngines"),
    (2.5, 3.2, "Metrics &\nTracing"),
]

for x, y, text in descriptions:
    ax.text(x, y, text, ha='center', va='center', fontsize=13, color='#888888', style='italic')

# Add a legend box
legend_y = 2.25
legend_items = [
    (1.5, legend_y, client_color, 'Client Layer'),
    (3.5, legend_y, gateway_color, 'Gateway Layer'),
    (5.5, legend_y, service_color, 'Service Layer'),
    (7.5, legend_y, model_color, 'Inference Layer'),
]

for x, y, color, label in legend_items:
    box = FancyBboxPatch(
        (x - 0.3, y - 0.15), 0.3, 0.3,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        facecolor=color,
        edgecolor=border_color,
        linewidth=1
    )
    ax.add_patch(box)
    ax.text(x + 0.3, y, label, ha='left', va='center', fontsize=13)

# Data flow annotation
ax.annotate('', xy=(9, 7), xytext=(9, 5.5),
            arrowprops=dict(arrowstyle='<->', color='#666666', lw=1.5))
ax.text(9.3, 6.25, 'Request\nFlow', ha='left', va='center', fontsize=13, color='#666666')

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
