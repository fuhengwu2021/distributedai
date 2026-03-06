#!/usr/bin/env python3
"""
Multi-Engine Routing Architecture Diagram
Chapter 9: Production LLM Serving Stack

Shows the architecture of routing same model to different inference engines:
- Client applications sending requests with model and owned_by fields
- API Gateway parsing both fields and routing accordingly
- vLLM and SGLang services serving the same model
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os
import sys

# Add shared directory to path for math4ai imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from math4ai import configure_math_fonts, save_figure

# Configure matplotlib for math expressions
configure_math_fonts()

fig, ax = plt.subplots(figsize=(7, 6))
ax.set_xlim(0.7, 9.3)
ax.set_ylim(0, 8)
ax.set_aspect('equal')
ax.axis('off')

# Colors - more contrast
client_color = '#C5CAE9'     # Medium indigo
gateway_color = '#BBDEFB'    # Medium blue
vllm_color = '#A5D6A7'       # Medium green
sglang_color = '#FFCC80'     # Medium orange
border_color = '#212121'     # Darker gray

# Client Applications box
client_box = FancyBboxPatch(
    (1.5, 6.8), 7.0, 0.5,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    facecolor=client_color,
    edgecolor=border_color,
    linewidth=1.5
)
ax.add_patch(client_box)
ax.text(5, 7.05, 'Client Applications', ha='center', va='center',
        fontsize=12, fontweight='bold')

# Arrow from Client to Gateway
ax.annotate('', xy=(5, 5.9), xytext=(5, 6.8),
            arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))
ax.text(5.6, 6.35, "HTTP Request with 'model'", ha='left', va='center',
        fontsize=11, style='italic', color='black')
ax.text(5.6, 6.05, "and 'owned_by' fields", ha='left', va='center',
        fontsize=11, style='italic', color='black')

# API Gateway box
gateway_box = FancyBboxPatch(
    (1.0, 4.0), 8.0, 1.9,
    boxstyle="round,pad=0.02,rounding_size=0.1",
    facecolor=gateway_color,
    edgecolor=border_color,
    linewidth=1.5
)
ax.add_patch(gateway_box)
ax.text(5, 5.6, 'API Gateway (Unified Entry Point)', ha='center', va='center',
        fontsize=11, fontweight='bold')

# Gateway features
gateway_features = [
    "Parses 'model' field from request body",
    "Parses 'owned_by' field for engine selection",
    "Routes to appropriate service based on both fields",
    "Returns response to client"
]
for i, feature in enumerate(gateway_features):
    ax.text(5, 5.15 - i * 0.28, f"• {feature}", ha='center', va='center',
            fontsize=11, color='black')

# Arrows from Gateway to Services
ax.annotate('', xy=(3.0, 2.6), xytext=(3.5, 4.0),
            arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))
ax.annotate('', xy=(7.0, 2.6), xytext=(6.5, 4.0),
            arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))

# vLLM Service box
vllm_box = FancyBboxPatch(
    (0.8, 0.3), 3.9, 2.3,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    facecolor=vllm_color,
    edgecolor='#2E7D32',
    linewidth=1.5
)
ax.add_patch(vllm_box)
ax.text(2.75, 2.35, 'vLLM Service', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#2E7D32')
ax.text(2.75, 2.0, '(Llama-3.2-1B)', ha='center', va='center',
        fontsize=10, color='black')
ax.text(2.75, 1.65, 'Pod: vllm-llama-32-1b', ha='center', va='center',
        fontsize=10, color='black')
ax.text(2.75, 1.35, 'Service: vllm-llama-32-1b:8000', ha='center', va='center',
        fontsize=10, color='black')
ax.text(2.75, 1.0, 'inference_server: "vllm"', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#2E7D32')
ax.text(2.75, 0.65, 'Image: vllm/vllm-openai:v0.12.0', ha='center', va='center',
        fontsize=10, style='italic', color='black')

# SGLang Service box
sglang_box = FancyBboxPatch(
    (5.3, 0.3), 3.9, 2.3,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    facecolor=sglang_color,
    edgecolor='#E65100',
    linewidth=1.5
)
ax.add_patch(sglang_box)
ax.text(7.25, 2.35, 'SGLang Service', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#E65100')
ax.text(7.25, 2.0, '(Llama-3.2-1B)', ha='center', va='center',
        fontsize=10, color='black')
ax.text(7.25, 1.65, 'Pod: sglang-llama-32-1b', ha='center', va='center',
        fontsize=10, color='black')
ax.text(7.25, 1.35, 'Service: sglang-llama-32-1b:8000', ha='center', va='center',
        fontsize=10, color='black')
ax.text(7.25, 1.0, 'inference_server: "sglang"', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#E65100')
ax.text(7.25, 0.65, 'Image: lmsysorg/sglang:v0.5.6', ha='center', va='center',
        fontsize=10, style='italic', color='black')

# Legend
legend_y = 7.6
legend_items = [
    (1.8, legend_y, client_color, 'Client'),
    (3.5, legend_y, gateway_color, 'Gateway'),
    (5.2, legend_y, vllm_color, 'vLLM'),
    (6.7, legend_y, sglang_color, 'SGLang'),
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
    ax.text(x + 0.2, y, label, ha='left', va='center', fontsize=11)

plt.tight_layout(pad=0.1)
save_figure(__file__)
