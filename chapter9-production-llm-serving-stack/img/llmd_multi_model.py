#!/usr/bin/env python3
"""
llm-d Multi-Model Architecture Diagram
Chapter 9: Production LLM Serving Stack

Shows the llm-d architecture for multi-model serving:
- Client applications sending requests
- Inference Gateway with intelligent load balancing
- InferencePool routing layer
- Multiple ModelService instances with vLLM pods
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

fig, ax = plt.subplots(figsize=(6, 5))
ax.set_xlim(0.5, 9.5)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')

# Colors - consistent with other diagrams
client_color = '#C5CAE9'     # Medium indigo
gateway_color = '#BBDEFB'    # Medium blue
pool_color = '#B3E5FC'       # Light cyan
service_color = '#A5D6A7'    # Medium green
border_color = '#212121'     # Darker gray

# Client Applications box
client_box = FancyBboxPatch(
    (1.5, 8.0), 7.0, 0.5,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    facecolor=client_color,
    edgecolor=border_color,
    linewidth=1.5
)
ax.add_patch(client_box)
ax.text(5, 8.25, 'Client Applications', ha='center', va='center',
        fontsize=12, fontweight='bold')

# Arrow from Client to Gateway
ax.annotate('', xy=(5, 7.1), xytext=(5, 8.0),
            arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))
ax.text(5.6, 7.55, "HTTP Request with 'model' field", ha='left', va='center',
        fontsize=11, style='italic', color='#666666')

# Inference Gateway box
gateway_box = FancyBboxPatch(
    (1.0, 5.5), 8.0, 1.6,
    boxstyle="round,pad=0.02,rounding_size=0.1",
    facecolor=gateway_color,
    edgecolor=border_color,
    linewidth=1.5
)
ax.add_patch(gateway_box)
ax.text(5, 6.8, 'Inference Gateway (Kubernetes Gateway API)', ha='center', va='center',
        fontsize=11, fontweight='bold')

# Gateway features
gateway_features = [
    "Load balancing with prefix-cache awareness",
    "Intelligent request scheduling",
    "Traffic routing to InferencePool"
]
for i, feature in enumerate(gateway_features):
    ax.text(5, 6.35 - i * 0.28, f"• {feature}", ha='center', va='center',
            fontsize=10, color='#424242')

# Arrow from Gateway to Pool
ax.annotate('', xy=(5, 4.6), xytext=(5, 5.5),
            arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))

# InferencePool box
pool_box = FancyBboxPatch(
    (1.0, 3.0), 8.0, 1.6,
    boxstyle="round,pad=0.02,rounding_size=0.1",
    facecolor=pool_color,
    edgecolor='#0277BD',
    linewidth=1.5
)
ax.add_patch(pool_box)
ax.text(5, 4.3, 'InferencePool (Routing Layer)', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#0277BD')

# Pool features
pool_features = [
    "Routes requests to appropriate ModelService",
    "Model-aware request distribution",
    "Health checking and load balancing"
]
for i, feature in enumerate(pool_features):
    ax.text(5, 3.85 - i * 0.28, f"• {feature}", ha='center', va='center',
            fontsize=10, color='#424242')

# Arrows from Pool to Services
ax.annotate('', xy=(2.75, 2.1), xytext=(3.5, 3.0),
            arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))
ax.annotate('', xy=(7.25, 2.1), xytext=(6.5, 3.0),
            arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))

# ModelService 1 box
service1_box = FancyBboxPatch(
    (0.6, 0.2), 3.9, 1.9,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    facecolor=service_color,
    edgecolor='#2E7D32',
    linewidth=1.5
)
ax.add_patch(service1_box)
ax.text(2.55, 1.9, 'ModelService 1', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#2E7D32')
ax.text(2.55, 1.55, '(Llama-3.2-1B)', ha='center', va='center',
        fontsize=10, color='#666666')
ax.text(2.55, 1.2, 'vLLM Pods (2x)', ha='center', va='center',
        fontsize=10, color='#888888')
ax.text(2.55, 0.85, '• Intelligent load balancing', ha='center', va='center',
        fontsize=9, color='#888888')
ax.text(2.55, 0.55, '• Prefix cache aware routing', ha='center', va='center',
        fontsize=9, color='#888888')

# ModelService 2 box
service2_box = FancyBboxPatch(
    (5.5, 0.2), 3.9, 1.9,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    facecolor=service_color,
    edgecolor='#2E7D32',
    linewidth=1.5
)
ax.add_patch(service2_box)
ax.text(7.45, 1.9, 'ModelService 2', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#2E7D32')
ax.text(7.45, 1.55, '(Qwen2.5-0.5B)', ha='center', va='center',
        fontsize=10, color='#666666')
ax.text(7.45, 1.2, 'vLLM Pods (2x)', ha='center', va='center',
        fontsize=10, color='#888888')
ax.text(7.45, 0.85, '• Intelligent load balancing', ha='center', va='center',
        fontsize=9, color='#888888')
ax.text(7.45, 0.55, '• Prefix cache aware routing', ha='center', va='center',
        fontsize=9, color='#888888')

# Legend
legend_y = 8.7
legend_items = [
    (1.5, legend_y, client_color, 'Client'),
    (3.0, legend_y, gateway_color, 'Gateway'),
    (4.8, legend_y, pool_color, 'Pool'),
    (6.3, legend_y, service_color, 'Service'),
]

for x, y, color, label in legend_items:
    box = FancyBboxPatch(
        (x - 0.2, y - 0.012), 0.25, 0.25,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor=color,
        edgecolor=border_color,
        linewidth=1
    )
    ax.add_patch(box)
    ax.text(x + 0.12, y+0.1, label, ha='left', va='center', fontsize=11)

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
