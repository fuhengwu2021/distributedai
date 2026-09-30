#!/usr/bin/env python3
"""
Network Topology and Communication Patterns Diagram
Shows distributed communication patterns and bottleneck identification.

Follows ~/mmb's localized_figure standard:
- Single implementation, multiple outputs (<stem>.png for English, <stem>_zh.png for Chinese)
- High-resolution (300 DPI) PNG exports
"""

import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

# Import localized_figure and styling from shared/figstyle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "title_ring": "Ring AllReduce Communication",
        "ring_desc": "Each GPU sends gradients to next GPU in ring\nN-1 steps to complete AllReduce",
        "gpu_fmt": "GPU{i}",
        "title_bottleneck": "Identifying Communication Bottlenecks",
        "node0": "Node 0",
        "node1": "Node 1",
        "gpu_node_fmt": "GPU {idx}",
        "nvlink": "NVLink\n(600 GB/s)",
        "bottleneck": "⚠ Network\nBottleneck",
        "infiniband": "InfiniBand\n(200 GB/s)",
        "bw_title": "Bandwidth Hierarchy:",
        "bw_desc": "NVLink (600 GB/s) >> InfiniBand (200 GB/s) >> Ethernet (100 Gbps = 12.5 GB/s)",
    },
    "zh": {
        "title_ring": "环形 AllReduce 通信拓扑 (Ring AllReduce)",
        "ring_desc": "每张 GPU 向环中下一张 GPU 传递梯度分块\n经历 N-1 步完成 AllReduce 归约通信",
        "gpu_fmt": "GPU{i}",
        "title_bottleneck": "跨机通信瓶颈定位与带宽层级对比",
        "node0": "计算节点 0 (Node 0)",
        "node1": "计算节点 1 (Node 1)",
        "gpu_node_fmt": "GPU {idx}",
        "nvlink": "片间 NVLink\n(600 GB/s)",
        "bottleneck": "⚠ 跨机网络\n通信瓶颈",
        "infiniband": "跨机 InfiniBand\n(200 GB/s)",
        "bw_title": "硬件互联带宽层级（Bandwidth Hierarchy）：",
        "bw_desc": "机内 NVLink (600 GB/s) >> 跨机 InfiniBand (200 GB/s) >> 以太网 (100 Gbps = 12.5 GB/s)",
    }
}


def draw(text: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left plot: Ring AllReduce topology
    ax1 = axes[0]
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title(text["title_ring"], fontsize=14, fontweight='bold', pad=20)

    # GPU positions in a ring
    n_gpus = 8
    angles = np.linspace(0, 2*np.pi, n_gpus, endpoint=False) - np.pi/2
    radius = 1.3
    gpu_positions = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]

    # Draw GPUs
    gpu_colors = plt.cm.Set3(np.linspace(0, 1, n_gpus))
    for i, (pos, color) in enumerate(zip(gpu_positions, gpu_colors)):
        circle = Circle(pos, 0.25, facecolor=color, edgecolor='#37474F', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(pos[0], pos[1], text["gpu_fmt"].format(i=i), ha='center', va='center', fontsize=9, fontweight='bold')

    # Draw ring connections with arrows
    for i in range(n_gpus):
        start = gpu_positions[i]
        end = gpu_positions[(i + 1) % n_gpus]

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)

        shrink = 0.3
        start_adj = (start[0] + shrink * dx/length, start[1] + shrink * dy/length)
        end_adj = (end[0] - shrink * dx/length, end[1] - shrink * dy/length)

        ax1.annotate('', xy=end_adj, xytext=start_adj,
                    arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2, 
                                   connectionstyle='arc3,rad=0.1'))

    # Add legend/explanation
    ax1.text(0, -1.8, text["ring_desc"], 
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='#E3F2FD', edgecolor='#1565C0', alpha=0.9))

    # Right plot: Communication bottleneck visualization
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.axis('off')
    ax2.set_title(text["title_bottleneck"], fontsize=14, fontweight='bold', pad=10)

    # Draw two nodes
    node_colors = ['#E3F2FD', '#E8F5E9']
    node_labels = [text["node0"], text["node1"]]

    for i, (color, label) in enumerate(zip(node_colors, node_labels)):
        x_base = 1 + i * 5

        # Node box
        node_box = FancyBboxPatch((x_base, 2), 3, 5,
                                   boxstyle="round,pad=0.05,rounding_size=0.2",
                                   facecolor=color, edgecolor='#37474F', linewidth=2)
        ax2.add_patch(node_box)
        ax2.text(x_base + 1.5, 6.7, label, ha='center', va='center', 
                fontsize=12, fontweight='bold')

        # GPUs inside node
        for j in range(4):
            gpu_y = 5.5 - j * 1.1
            gpu_box = FancyBboxPatch((x_base + 0.3, gpu_y - 0.35), 2.4, 0.7,
                                      boxstyle="round,pad=0.02,rounding_size=0.1",
                                      facecolor='#BBDEFB' if i == 0 else '#C8E6C9',
                                      edgecolor='#546E7A', linewidth=1)
            ax2.add_patch(gpu_box)
            ax2.text(x_base + 1.5, gpu_y, text["gpu_node_fmt"].format(idx=i*4 + j), ha='center', va='center', fontsize=9)

    # Draw NVLink connections (fast, within node)
    for i in range(2):
        x_base = 1 + i * 5
        for j in range(3):
            y1 = 5.5 - j * 1.1 - 0.35
            y2 = 5.5 - (j+1) * 1.1 + 0.35
            ax2.annotate('', xy=(x_base + 1.5, y2), xytext=(x_base + 1.5, y1),
                        arrowprops=dict(arrowstyle='<->', color='#4CAF50', lw=2))

    # NVLink label
    ax2.text(2.5, 2.5, text["nvlink"], ha='center', va='center', fontsize=8,
             color='#2E7D32', fontweight='bold')
    ax2.text(7.5, 2.5, text["nvlink"], ha='center', va='center', fontsize=8,
             color='#2E7D32', fontweight='bold')

    # Draw network connection (slow, between nodes) - BOTTLENECK
    ax2.annotate('', xy=(6, 4.5), xytext=(4, 4.5),
                arrowprops=dict(arrowstyle='<->', color='#F44336', lw=3,
                               connectionstyle='arc3,rad=0'))

    # Bottleneck indicator
    ax2.text(5, 5.3, text["bottleneck"], ha='center', va='center', fontsize=10,
             color='#C62828', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#FFEBEE', edgecolor='#C62828'))
    ax2.text(5, 3.7, text["infiniband"], ha='center', va='center', fontsize=8,
             color='#C62828')

    # Bandwidth comparison box
    bw_box = FancyBboxPatch((0.5, 0.3), 9, 1.4,
                             boxstyle="round,pad=0.05,rounding_size=0.1",
                             facecolor='#FFF8E1', edgecolor='#FF8F00', linewidth=2)
    ax2.add_patch(bw_box)
    ax2.text(5, 1.3, text["bw_title"], ha='center', va='center', 
             fontsize=11, fontweight='bold', color='#E65100')
    ax2.text(5, 0.7, text["bw_desc"],
             ha='center', va='center', fontsize=9, color='#37474F')

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    localized_figure(draw, "network_topology", LABELS, __file__)
