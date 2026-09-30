#!/usr/bin/env python3
"""
Visualize the Modern AI Model Lifecycle

This script creates a circular diagram showing the continuous lifecycle:
Data Engineering → Model Training → Model Inference → Model Benchmarking → 
Model Deployment → Data Engineering (repeat)
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Ensure shared directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "stages": [
            "Data Engineering",
            "Model Training",
            "Model Inference",
            "Model Benchmarking",
            "Model Deployment",
        ]
    },
    "zh": {
        "stages": [
            "数据工程",
            "模型训练",
            "模型推理",
            "基准测试",
            "模型部署",
        ]
    }
}

STAGE_CONFIG = [
    {
        "angle": np.pi / 2,  # Top
        "color": "#E3F2FD",
        "edge_color": "#1976D2",
    },
    {
        "angle": np.pi / 2 + 2 * np.pi / 5,  # Top-right
        "color": "#E8F5E9",
        "edge_color": "#2E7D32",
    },
    {
        "angle": np.pi / 2 + 4 * np.pi / 5,  # Bottom-right
        "color": "#FFF3E0",
        "edge_color": "#E65100",
    },
    {
        "angle": np.pi / 2 + 6 * np.pi / 5,  # Bottom-left
        "color": "#F3E5F5",
        "edge_color": "#7B1FA2",
    },
    {
        "angle": np.pi / 2 + 8 * np.pi / 5,  # Top-left
        "color": "#FFEBEE",
        "edge_color": "#C62828",
    },
]


def draw(text: dict) -> plt.Figure:
    """Create a circular lifecycle diagram"""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    
    stage_names = text["stages"]
    radius = 0.9
    box_width = 0.8
    box_height = 0.4
    
    stage_positions = []
    for i, cfg in enumerate(STAGE_CONFIG):
        angle = cfg["angle"]
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        stage_positions.append((x, y))
        
        box = FancyBboxPatch(
            (x - box_width/2, y - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.02",
            facecolor=cfg["color"],
            edgecolor=cfg["edge_color"],
            linewidth=2.0,
            zorder=3
        )
        ax.add_patch(box)
        
        ax.text(
            x, y,
            stage_names[i],
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            color=cfg["edge_color"],
            zorder=4
        )
    
    arrow_style = dict(
        arrowstyle='->',
        lw=2.0,
        color='#333333',
        zorder=2,
        mutation_scale=20,
        shrinkA=5,
        shrinkB=5
    )
    
    def get_box_intersection(cx, cy, target_x, target_y, box_w, box_h):
        dx = target_x - cx
        dy = target_y - cy
        
        if abs(dx) < 1e-10:
            if dy > 0:
                return (cx, cy + box_h/2)
            else:
                return (cx, cy - box_h/2)
        if abs(dy) < 1e-10:
            if dx > 0:
                return (cx + box_w/2, cy)
            else:
                return (cx - box_w/2, cy)
        
        t_left = (cx - box_w/2 - cx) / dx
        y_left = cy + t_left * dy
        if t_left > 0 and cy - box_h/2 <= y_left <= cy + box_h/2:
            return (cx - box_w/2, y_left)
        
        t_right = (cx + box_w/2 - cx) / dx
        y_right = cy + t_right * dy
        if t_right > 0 and cy - box_h/2 <= y_right <= cy + box_h/2:
            return (cx + box_w/2, y_right)
        
        t_top = (cy + box_h/2 - cy) / dy
        x_top = cx + t_top * dx
        if t_top > 0 and cx - box_w/2 <= x_top <= cx + box_w/2:
            return (x_top, cy + box_h/2)
        
        t_bottom = (cy - box_h/2 - cy) / dy
        x_bottom = cx + t_bottom * dx
        if t_bottom > 0 and cx - box_w/2 <= x_bottom <= cx + box_w/2:
            return (x_bottom, cy - box_h/2)
        
        return (cx, cy)
    
    for i in range(len(STAGE_CONFIG)):
        start_idx = i
        end_idx = (i + 1) % len(STAGE_CONFIG)
        
        start_x, start_y = stage_positions[start_idx]
        end_x, end_y = stage_positions[end_idx]
        
        start_point_x, start_point_y = get_box_intersection(
            start_x, start_y, end_x, end_y, box_width, box_height
        )
        end_point_x, end_point_y = get_box_intersection(
            end_x, end_y, start_x, start_y, box_width, box_height
        )
        
        arrow = FancyArrowPatch(
            (start_point_x, start_point_y),
            (end_point_x, end_point_y),
            **arrow_style
        )
        ax.add_patch(arrow)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    localized_figure(draw, "model_lifecycle", LABELS, __file__)
