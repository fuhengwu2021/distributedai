"""
AI Cluster Architecture Diagram

This diagram visualizes a multi-node AI cluster showing how nodes with multiple GPUs
are connected via high-speed networks to enable distributed training and inference.
"""

import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure
from gpu import draw_gpu_shape
from cpu import draw_cpu_shape

LABELS = {
    "en": {
        "node_1": "Node 1",
        "node_2": "Node 2",
        "infiniband": "InfiniBand",
        "nvswitch": "NVSwitch",
        "gpu0": "GPU0",
        "gpu7": "GPU7",
        "cpu": "CPU",
        "gpu": "GPU",
    },
    "zh": {
        "node_1": "节点 1",
        "node_2": "节点 2",
        "infiniband": "InfiniBand",
        "nvswitch": "NVSwitch",
        "gpu0": "GPU0",
        "gpu7": "GPU7",
        "cpu": "CPU",
        "gpu": "GPU",
    }
}

# Configuration
NODE_WIDTH, NODE_HEIGHT = 2.0, 1.2
NODE_SPACING = 3.0
CENTER_Y = 2.0
GPU_Y_OFFSET = -0.15  # Move GPUs downward to bring them closer to labels
GPU_SPACING = 0.6
NODE_LABEL_Y_OFFSET = 0.8
GPU_LABEL_Y_OFFSET = -0.35  # Moved upward (less negative)


def draw_gpu_icon(center_x, center_y, label, ax, gpu_text="GPU"):
    """Draws a GPU icon using the function imported from gpu.py."""
    scale = 0.525
    draw_gpu_shape(ax, center_x=center_x, center_y=center_y, 
                   scale=scale, linewidth=2.5, show_text=True, text_label=gpu_text)
    
    # GPU label (below the icon)
    ax.text(center_x, center_y + GPU_LABEL_Y_OFFSET, label,
            fontsize=14, ha='center', va='top', color='#34495E', zorder=4)


def draw_node(center_x, center_y, node_label, ax, cpu_text="CPU"):
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
                   scale=cpu_scale, linewidth=2.0, show_text=True, text_label=cpu_text)
    
    # Add ellipsis on right side of CPU
    ax.text(center_x + cpu_spacing, cpu_y, '...', fontsize=16, ha='center', va='center',
            fontweight='bold', color='#7F8C8D', zorder=3)
    
    ax.text(center_x, center_y + NODE_LABEL_Y_OFFSET, node_label,
            fontsize=16, ha='center', va='center', fontweight='bold', color='#2C3E50')
    return center_x, center_y


def draw(text: dict) -> plt.Figure:
    content_width = 5.4
    content_height = 1.9
    content_aspect = content_width / content_height

    fig_height = 3.2
    fig_width = fig_height * content_aspect
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    # Draw Nodes
    n1_x, n1_y = draw_node(1.5, CENTER_Y, text['node_1'], ax, cpu_text=text['cpu'])
    n2_x, n2_y = draw_node(1.5 + NODE_SPACING, CENTER_Y, text['node_2'], ax, cpu_text=text['cpu'])

    # GPU y position
    gpu_y = CENTER_Y + GPU_Y_OFFSET

    # Draw GPUs for Node 1
    draw_gpu_icon(n1_x - GPU_SPACING, gpu_y, text['gpu0'], ax, gpu_text=text['gpu'])
    ax.text(n1_x, gpu_y, '...', fontsize=20, ha='center', va='center', 
            fontweight='bold', color='#7F8C8D', zorder=3)
    draw_gpu_icon(n1_x + GPU_SPACING, gpu_y, text['gpu7'], ax, gpu_text=text['gpu'])

    # NVSwitch connection within Node 1
    nvlink_node1 = ConnectionPatch(
        (n1_x - GPU_SPACING + 0.15, gpu_y), (n1_x + GPU_SPACING - 0.15, gpu_y),
        "data", "data", arrowstyle='<->', mutation_scale=12,
        linewidth=1.5, color='#3498DB', linestyle='--', zorder=2, alpha=0.7
    )
    ax.add_patch(nvlink_node1)

    # Draw GPUs for Node 2
    draw_gpu_icon(n2_x - GPU_SPACING, gpu_y, text['gpu0'], ax, gpu_text=text['gpu'])
    ax.text(n2_x, gpu_y, '...', fontsize=20, ha='center', va='center', 
            fontweight='bold', color='#7F8C8D', zorder=3)
    draw_gpu_icon(n2_x + GPU_SPACING, gpu_y, text['gpu7'], ax, gpu_text=text['gpu'])

    # NVSwitch connection within Node 2
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
    ax.text((n1_x + n2_x)/2, gpu_y + 0.15, text['infiniband'],
            fontsize=14, ha='center', va='center', fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E74C3C', edgecolor='none'), zorder=5)

    ax.text(n1_x, gpu_y + 0.15, text['nvswitch'], fontsize=14, ha='center', va='center',
            fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#3498DB', edgecolor='none'), zorder=5)
    ax.text(n2_x, gpu_y + 0.15, text['nvswitch'], fontsize=14, ha='center', va='center',
            fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#3498DB', edgecolor='none'), zorder=5)

    ax.set_xlim(0.2, 5.8)
    ax.set_ylim(1.2, 3.0)
    plt.tight_layout(pad=0)
    return fig


if __name__ == '__main__':
    localized_figure(draw, "ai_cluster_demo", LABELS, __file__, pad_inches=0.0)
