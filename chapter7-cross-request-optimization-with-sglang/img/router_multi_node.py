"""
Router-Based Multi-Node Deployment

Shows the router distributing requests across multiple nodes,
each running a complete model instance (data parallelism via routing).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "router": "Router",
        "cache_aware": "cache-aware",
        "workers": ["Worker 1", "Worker 2", "Worker 3"],
        "nodes": ["Node 1", "Node 2", "Node 3"],
        "full_model": "Full Model",
        "request_dist": "request distribution",
    },
    "zh": {
        "router": "Router 网关",
        "cache_aware": "（Cache 感知）",
        "workers": ["Worker 1", "Worker 2", "Worker 3"],
        "nodes": ["节点 1", "节点 2", "节点 3"],
        "full_model": "完整模型",
        "request_dist": "请求分发",
    }
}


def draw_box(ax, x, y, width, height, label, color, edge_color, fontsize=11, sublabel=None):
    """Draw a rounded box with label."""
    box = patches.FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=2, edgecolor=edge_color, facecolor=color
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + 0.18, label, ha='center', va='center', 
                fontsize=fontsize, fontweight='bold')
        ax.text(x, y - 0.15, sublabel, ha='center', va='center', 
                fontsize=fontsize-1, color='#555', style='italic')
    else:
        ax.text(x, y, label, ha='center', va='center', 
                fontsize=fontsize, fontweight='bold')


def draw_arrow(ax, start, end, color='#546e7a', lw=2):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_xlim(0.3, 8.7)
    ax.set_ylim(1, 5.)
    ax.axis('off')
    
    # Colors
    router_color = '#fce4ec'
    worker_color = '#e8f5e9'
    
    # Router at top
    draw_box(ax, 4.5, 4.5, 2.4, 0.8, text['router'], router_color, '#c62828', fontsize=12,
             sublabel=text['cache_aware'])
    
    # Workers (3 nodes)
    worker_positions = [1.5, 4.5, 7.5]
    worker_labels = text['workers']
    node_labels = text['nodes']
    
    for i, (wx, wlabel, nlabel) in enumerate(zip(worker_positions, worker_labels, node_labels)):
        # Worker box
        draw_box(ax, wx, 2.0, 2.0, 1.4, wlabel, worker_color, '#388e3c', fontsize=11)
        # Node label below
        ax.text(wx, 1.1, nlabel, ha='center', va='center', fontsize=10, color='#666')
        # "Full Model" inside
        ax.text(wx, 1.75, text['full_model'], ha='center', va='center', fontsize=9, 
                color='#555', style='italic')
        
        # Arrow from router to worker
        draw_arrow(ax, (4.5, 4.05), (wx, 2.75), color='#546e7a', lw=1.5)
    
    # Annotations
    ax.text(2., 3.6, text['request_dist'], ha='center', va='center', 
            fontsize=10, color='#666', style='italic')
    
    plt.tight_layout(pad=0.1)
    return fig


if __name__ == '__main__':
    localized_figure(draw, "router_multi_node", LABELS, __file__, pad_inches=0.02, use_math_fonts=True)
