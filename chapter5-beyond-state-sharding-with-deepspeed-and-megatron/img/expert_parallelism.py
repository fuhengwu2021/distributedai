"""
Expert Parallelism in MoE Models

This diagram illustrates how experts are distributed across GPUs in a Mixture of Experts
(MoE) model with expert parallelism. It shows:
1. Input tokens being processed by a shared router
2. Router decisions routing tokens to different experts
3. Experts distributed across 4 GPUs (2 experts per GPU)
4. All-to-all communication pattern for token routing
5. Combined output after expert processing
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from math4ai import configure_math_fonts, save_figure

configure_math_fonts()

fig, ax = plt.subplots(figsize=(14, 10))

# Colors
gpu_colors = ['#E3F2FD', '#E8F5E9', '#FFF3E0', '#FCE4EC']  # Light blue, green, orange, pink
expert_colors = ['#1976D2', '#388E3C', '#F57C00', '#C2185B']  # Darker versions
router_color = '#9C27B0'  # Purple
token_colors = ['#FF5722', '#2196F3', '#4CAF50', '#FFC107', '#9C27B0', '#00BCD4', '#E91E63', '#8BC34A']
arrow_color = '#546E7A'

# Layout parameters
gpu_width = 2.8
gpu_height = 3.5
gpu_spacing = 3.2
gpu_y = 3.5
gpu_start_x = 1.0

# Draw input tokens at top
token_y = 9.5
token_radius = 0.25
token_labels = [r'$t_1$', r'$t_2$', r'$t_3$', r'$t_4$', r'$t_5$', r'$t_6$', r'$t_7$', r'$t_8$']
token_x_positions = np.linspace(2.5, 11.5, 8)

ax.text(7, 10.3, 'Input Tokens', fontsize=14, ha='center', fontweight='bold')
for i, (x, label) in enumerate(zip(token_x_positions, token_labels)):
    circle = plt.Circle((x, token_y), token_radius, color=token_colors[i], ec='black', linewidth=1.5, zorder=5)
    ax.add_patch(circle)
    ax.text(x, token_y, label, fontsize=22, ha='center', va='center', color='white', fontweight='bold', zorder=10)

# Draw router
router_width = 4
router_height = 0.8
router_x = 5
router_y = 8.0
router = FancyBboxPatch((router_x, router_y), router_width, router_height,
                         boxstyle="round,pad=0.05,rounding_size=0.2",
                         facecolor=router_color, edgecolor='black', linewidth=2, zorder=4)
ax.add_patch(router)
ax.text(router_x + router_width/2, router_y + router_height/2, 'Router (Gating Network)',
        fontsize=18, ha='center', va='center', color='white', fontweight='bold', zorder=10)

# Arrows from tokens to router
for x in token_x_positions:
    ax.annotate('', xy=(router_x + router_width/2, router_y + router_height),
                xytext=(x, token_y - token_radius),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.2, 
                               connectionstyle='arc3,rad=0'))

# Draw GPUs with experts
expert_labels = [
    ['Expert 1', 'Expert 2'],
    ['Expert 3', 'Expert 4'],
    ['Expert 5', 'Expert 6'],
    ['Expert 7', 'Expert 8']
]

gpu_centers = []
expert_positions = []

for gpu_idx in range(4):
    gpu_x = gpu_start_x + gpu_idx * gpu_spacing
    gpu_centers.append(gpu_x + gpu_width/2)
    
    # GPU background
    gpu_rect = FancyBboxPatch((gpu_x, gpu_y), gpu_width, gpu_height,
                               boxstyle="round,pad=0.02,rounding_size=0.15",
                               facecolor=gpu_colors[gpu_idx], edgecolor='black', 
                               linewidth=2, zorder=2)
    ax.add_patch(gpu_rect)
    
    # GPU label
    ax.text(gpu_x + gpu_width/2, gpu_y + gpu_height + 0.2, f'GPU {gpu_idx}',
            fontsize=18, ha='center', va='bottom', fontweight='bold')
    
    # Draw 2 experts per GPU
    expert_width = 2.2
    expert_height = 1.2
    expert_x = gpu_x + (gpu_width - expert_width) / 2
    
    for exp_idx in range(2):
        expert_y = gpu_y + 0.4 + exp_idx * 1.6
        expert_rect = FancyBboxPatch((expert_x, expert_y), expert_width, expert_height,
                                      boxstyle="round,pad=0.02,rounding_size=0.1",
                                      facecolor='white', edgecolor=expert_colors[gpu_idx],
                                      linewidth=2, zorder=3)
        ax.add_patch(expert_rect)
        ax.text(expert_x + expert_width/2, expert_y + expert_height/2,
                expert_labels[gpu_idx][exp_idx], fontsize=14, ha='center', va='center',
                fontweight='bold', color=expert_colors[gpu_idx])
        expert_positions.append((expert_x + expert_width/2, expert_y + expert_height))

# Draw routing arrows from router to GPUs (all-to-all pattern)
# Token routing assignments (example: top-2 routing)
# t1 -> E1, E3; t2 -> E2, E5; t3 -> E4, E7; t4 -> E1, E6; etc.
routing_map = {
    0: [0, 2],   # t1 -> E1 (GPU0), E3 (GPU1)
    1: [1, 4],   # t2 -> E2 (GPU0), E5 (GPU2)
    2: [3, 6],   # t3 -> E4 (GPU1), E7 (GPU3)
    3: [0, 5],   # t4 -> E1 (GPU0), E6 (GPU2)
    4: [2, 7],   # t5 -> E3 (GPU1), E8 (GPU3)
    5: [1, 6],   # t6 -> E2 (GPU0), E7 (GPU3)
    6: [4, 3],   # t7 -> E5 (GPU2), E4 (GPU1)
    7: [5, 7],   # t8 -> E6 (GPU2), E8 (GPU3)
}

# Draw some representative routing arrows (not all to avoid clutter)
router_bottom = router_y
for token_idx in [0, 2, 4, 6]:  # Show routing for tokens 1, 3, 5, 7
    for expert_idx in routing_map[token_idx]:
        gpu_idx = expert_idx // 2
        target_x = gpu_centers[gpu_idx]
        target_y = gpu_y + gpu_height
        
        ax.annotate('', xy=(target_x, target_y),
                    xytext=(router_x + router_width/2, router_bottom),
                    arrowprops=dict(arrowstyle='->', color=token_colors[token_idx], 
                                   lw=1.5, alpha=0.7,
                                   connectionstyle=f'arc3,rad={0.1 * (gpu_idx - 1.5)}'))

# Draw subtle all-to-all communication lines between GPUs (dashed, not intrusive)
for i in range(3):
    x1 = gpu_centers[i]
    x2 = gpu_centers[i + 1]
    y_line = gpu_y + 0.15
    ax.plot([x1, x2], [y_line, y_line], '--', color='#78909C', lw=1.5, alpha=0.6)

# Add small label for all-to-all
ax.text(3, gpu_y - 0.5, 'All-to-All Communication', fontsize=16, ha='center', va='top', 
        color='#546E7A', style='italic')

# Draw output at bottom
output_y = 1.5
output_width = 10
output_height = 0.8
output_x = 2
output_rect = FancyBboxPatch((output_x, output_y), output_width, output_height,
                              boxstyle="round,pad=0.05,rounding_size=0.2",
                              facecolor='#37474F', edgecolor='black', linewidth=2, zorder=4)
ax.add_patch(output_rect)
ax.text(output_x + output_width/2, output_y + output_height/2, 'Combined Expert Outputs',
        fontsize=18, ha='center', va='center', color='white', fontweight='bold', zorder=10)

# Arrows from GPUs to output
for gpu_x_center in gpu_centers:
    ax.annotate('', xy=(output_x + output_width/2, output_y + output_height),
                xytext=(gpu_x_center, gpu_y),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5,
                               connectionstyle='arc3,rad=0'))

# Add legend/annotation box
legend_x = 0.3
legend_y = 0.3
ax.text(legend_x, legend_y, 
        'Top-2 Routing: Each token is processed by 2 experts\n'
        'Colored arrows show token routing decisions',
        fontsize=14, ha='left', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#ECEFF1', edgecolor='gray', alpha=0.9))

# Set axis properties
ax.set_xlim(-0.5, 14.5)
ax.set_ylim(0, 11)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout(pad=0.1)
save_figure(__file__)
