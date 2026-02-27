#!/usr/bin/env python3
"""
Multi-Model Routing Architecture Diagram
Chapter 9: Production LLM Serving Stack

Shows the architecture of multi-model routing with API Gateway:
- Client applications sending requests
- API Gateway parsing model field and routing
- Multiple vLLM services serving different models
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os
import sys

# Add shared directory to path for math4ai imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from math4ai import configure_math_fonts

# Configure matplotlib for math expressions
configure_math_fonts()

fig, ax = plt.subplots(figsize=(7, 6))
ax.set_xlim(0.7, 9.3)
ax.set_ylim(0, 7)
ax.set_aspect('equal')
ax.axis('off')

# Colors - more contrast
client_color = '#C5CAE9'     # Medium indigo
gateway_color = '#BBDEFB'    # Medium blue
service_color = '#A5D6A7'    # Medium green
border_color = '#212121'     # Darker gray

# Client Applications box
client_box = FancyBboxPatch(
    (1.5, 5.8), 7.0, 0.5,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    facecolor=client_color,
    edgecolor=border_color,
    linewidth=1.5
)
ax.add_patch(client_box)
ax.text(5, 6.05, 'Client Applications', ha='center', va='center',
        fontsize=12, fontweight='bold')

# Arrow from Client to Gateway
ax.annotate('', xy=(5, 5.0), xytext=(5, 5.8),
            arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))
ax.text(5.6, 5.4, "HTTP Request with 'model' field", ha='left', va='center',
        fontsize=12, style='italic', color='black')

# API Gateway box
gateway_box = FancyBboxPatch(
    (1.0, 3.2), 8.0, 1.8,
    boxstyle="round,pad=0.02,rounding_size=0.1",
    facecolor=gateway_color,
    edgecolor=border_color,
    linewidth=1.5
)
ax.add_patch(gateway_box)
ax.text(5, 4.7, 'API Gateway (Unified Entry Point)', ha='center', va='center',
        fontsize=12, fontweight='bold')

# Gateway features
gateway_features = [
    "Parses 'model' field from request body",
    "Routes to appropriate vLLM service",
    "Returns response to client"
]
for i, feature in enumerate(gateway_features):
    ax.text(5, 4.2 - i * 0.3, f"• {feature}", ha='center', va='center',
            fontsize=12, color='black')

# Arrows from Gateway to Services
ax.annotate('', xy=(3.0, 1.8), xytext=(3.5, 3.2),
            arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))
ax.annotate('', xy=(7.0, 1.8), xytext=(6.5, 3.2),
            arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))

# vLLM Service 1 box
service1_box = FancyBboxPatch(
    (1.0, 0.3), 3.5, 1.5,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    facecolor=service_color,
    edgecolor='#2E7D32',
    linewidth=1.5
)
ax.add_patch(service1_box)
ax.text(2.75, 1.55, 'vLLM Service 1', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#2E7D32')
ax.text(2.75, 1.2, '(Llama-3.2-1B)', ha='center', va='center',
        fontsize=12, color='black')
ax.text(2.75, 0.85, 'Pod: vllm-llama-32-1b', ha='center', va='center',
        fontsize=12, color='black')
ax.text(2.75, 0.55, 'Service: vllm-llama-32-1b:8000', ha='center', va='center',
        fontsize=12, color='black')

# vLLM Service 2 box
service2_box = FancyBboxPatch(
    (5.5, 0.3), 3.5, 1.5,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    facecolor=service_color,
    edgecolor='#2E7D32',
    linewidth=1.5
)
ax.add_patch(service2_box)
ax.text(7.25, 1.55, 'vLLM Service 2', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#2E7D32')
ax.text(7.25, 1.2, '(Qwen2.5-0.5B)', ha='center', va='center',
        fontsize=12, color='black')
ax.text(7.25, 0.85, 'Pod: vllm-qwen-0.5b', ha='center', va='center',
        fontsize=12, color='black')
ax.text(7.25, 0.55, 'Service: vllm-qwen-0.5b:8000', ha='center', va='center',
        fontsize=12, color='black')

# Legend
legend_y = 6.6
legend_items = [
    (2.0, legend_y, client_color, 'Client'),
    (4.0, legend_y, gateway_color, 'Gateway'),
    (6.5, legend_y, service_color, 'Model Server'),
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
