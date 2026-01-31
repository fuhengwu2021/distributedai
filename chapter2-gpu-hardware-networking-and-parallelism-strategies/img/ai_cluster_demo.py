"""
AI Cluster Architecture Diagram

This diagram visualizes a multi-node AI cluster showing how nodes with multiple GPUs
are connected via high-speed networks to enable distributed training and inference.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import os
import sys

# Import GPU and CPU drawing functions
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from gpu import draw_gpu_shape
from cpu import draw_cpu_shape

# Save figure (standard pattern: same name as script)
script_name = os.path.splitext(os.path.basename(__file__))[0]
output_path = os.path.join(script_dir, f'{script_name}.png')

# Create figure
# Calculate optimal figure size based on content
# Content spans roughly: x from 0.3 to 5.7 (width ~5.4), y from 1.0 to 2.9 (height ~1.9)
# Add small padding for labels
content_width = 5.4
content_height = 1.9
content_aspect = content_width / content_height

# Set figure size to match content aspect ratio (reduced size)
fig_height = 3.2
fig_width = fig_height * content_aspect
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.axis('off')
# Don't use 'equal' aspect - it causes padding issues with mismatched figure/data ratios

# Configuration
NODE_WIDTH, NODE_HEIGHT = 2.0, 1.2
NODE_SPACING = 3.0
CENTER_Y = 2.0
GPU_Y_OFFSET = -0.15  # Move GPUs downward to bring them closer to labels
GPU_SPACING = 0.6
NODE_LABEL_Y_OFFSET = 0.8
GPU_LABEL_Y_OFFSET = -0.35  # Moved upward (less negative)

def draw_gpu_icon(center_x, center_y, label, ax):
    """Draws a GPU icon using the function imported from gpu.py."""
    # Scale factor for the GPU icon in this diagram
    scale = 0.525
    
    # Draw GPU shape using imported function
    draw_gpu_shape(ax, center_x=center_x, center_y=center_y, 
                   scale=scale, linewidth=2.5, show_text=True)
    
    # GPU label (below the icon)
    ax.text(center_x, center_y + GPU_LABEL_Y_OFFSET, label,
            fontsize=14, ha='center', va='top', color='#34495E', zorder=4)

def draw_node(center_x, center_y, node_label, ax):
    """Draw a node box with label and CPU icon."""
    node_box = FancyBboxPatch(
        (center_x - NODE_WIDTH/2, center_y - NODE_HEIGHT/2),
        NODE_WIDTH, NODE_HEIGHT,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        linewidth=2, edgecolor='#2C3E50', facecolor='#ECF0F1', zorder=1
    )
    ax.add_patch(node_box)
    
    # Add CPU icon in the node (positioned above center, below the label)
    cpu_y = center_y + 0.45
    cpu_scale = 0.42
    cpu_spacing = 0.35  # Spacing for ellipsis on left and right of CPU
    
    # Add ellipsis on left side of CPU
    ax.text(center_x - cpu_spacing, cpu_y, '...', fontsize=16, ha='center', va='center',
            fontweight='bold', color='#7F8C8D', zorder=3)
    
    # Draw CPU icon
    draw_cpu_shape(ax, center_x=center_x, center_y=cpu_y, 
                   scale=cpu_scale, linewidth=2.0, show_text=True)
    
    # Add ellipsis on right side of CPU
    ax.text(center_x + cpu_spacing, cpu_y, '...', fontsize=16, ha='center', va='center',
            fontweight='bold', color='#7F8C8D', zorder=3)
    
    ax.text(center_x, center_y + NODE_LABEL_Y_OFFSET, node_label,
            fontsize=16, ha='center', va='center', fontweight='bold', color='#2C3E50')
    return center_x, center_y

# Draw Nodes
n1_x, n1_y = draw_node(1.5, CENTER_Y, 'Node 1', ax)
n2_x, n2_y = draw_node(1.5 + NODE_SPACING, CENTER_Y, 'Node 2', ax)

# GPU y position (moved downward to bring closer to labels)
gpu_y = CENTER_Y + GPU_Y_OFFSET

# Draw GPUs for Node 1
draw_gpu_icon(n1_x - GPU_SPACING, gpu_y, 'GPU0', ax)
ax.text(n1_x, gpu_y, '...', fontsize=20, ha='center', va='center', 
        fontweight='bold', color='#7F8C8D', zorder=3)
draw_gpu_icon(n1_x + GPU_SPACING, gpu_y, 'GPU7', ax)

# NVSwitch connection within Node 1 (simplified - showing connection between visible GPUs)
nvlink_node1 = ConnectionPatch(
    (n1_x - GPU_SPACING + 0.15, gpu_y), (n1_x + GPU_SPACING - 0.15, gpu_y),
    "data", "data", arrowstyle='<->', mutation_scale=12,
    linewidth=1.5, color='#3498DB', linestyle='--', zorder=2, alpha=0.7
)
ax.add_patch(nvlink_node1)

# Draw GPUs for Node 2
draw_gpu_icon(n2_x - GPU_SPACING, gpu_y, 'GPU0', ax)
ax.text(n2_x, gpu_y, '...', fontsize=20, ha='center', va='center', 
        fontweight='bold', color='#7F8C8D', zorder=3)
draw_gpu_icon(n2_x + GPU_SPACING, gpu_y, 'GPU7', ax)

# NVSwitch connection within Node 2 (simplified - showing connection between visible GPUs)
nvlink_node2 = ConnectionPatch(
    (n2_x - GPU_SPACING + 0.15, gpu_y), (n2_x + GPU_SPACING - 0.15, gpu_y),
    "data", "data", arrowstyle='<->', mutation_scale=12,
    linewidth=1.5, color='#3498DB', linestyle='--', zorder=2, alpha=0.7
)
ax.add_patch(nvlink_node2)

# InfiniBand Connection
conn = ConnectionPatch(
    (n1_x + NODE_WIDTH/2, gpu_y), (n2_x - NODE_WIDTH/2, gpu_y),
    "data", "data", arrowstyle='<->', mutation_scale=20,
    linewidth=3, color='#E74C3C', zorder=2
)
ax.add_patch(conn)

# Network Labels
# InfiniBand label (between nodes)
ax.text((n1_x + n2_x)/2, gpu_y + 0.15, 'InfiniBand',
        fontsize=14, ha='center', va='center', fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E74C3C', edgecolor='none'), zorder=5)

# NVSwitch labels (within nodes - shown for both nodes)
ax.text(n1_x, gpu_y + 0.15, 'NVSwitch', fontsize=14, ha='center', va='center',
        fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#3498DB', edgecolor='none'), zorder=5)
ax.text(n2_x, gpu_y + 0.15, 'NVSwitch', fontsize=14, ha='center', va='center',
        fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#3498DB', edgecolor='none'), zorder=5)

# Final Layout - set limits to tightly match actual content
# Left: GPU0 at x=0.9 (1.5 - 0.6), with some margin for labels
# Right: GPU7 at x=5.1 (4.5 + 0.6), with some margin for labels  
# Bottom: GPU labels at y=1.55 (2.0 - 0.45), with some margin
# Top: InfiniBand label at y=2.4 (2.0 + 0.4), node labels at y=2.8 (2.0 + 0.8)
ax.set_xlim(0.2, 5.8)
ax.set_ylim(1.2, 3.0)
plt.tight_layout(pad=0)
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0,
            facecolor='white', edgecolor='none')
print(f"Saved figure to: {output_path}")
plt.close()

