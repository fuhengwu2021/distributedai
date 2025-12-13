#!/usr/bin/env python3
"""
Visualize the Modern AI Model Lifecycle

This script creates a circular diagram showing the continuous lifecycle:
Data Engineering → Model Training → Model Inference → Model Benchmarking → 
Model Deployment → Data Engineering (repeat)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def create_lifecycle_diagram(output_path: str = "code/chapter1/model_lifecycle.png"):
    """Create a circular lifecycle diagram"""
    
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Define stages with their positions (circular layout)
    stages = [
        {
            "name": "Data Engineering",
            "angle": np.pi / 2,  # Top
            "color": "#E3F2FD",
            "edge_color": "#1976D2",
            "description": "Data collection, curation,\ntransformation, validation,\nexploration"
        },
        {
            "name": "Model Training",
            "angle": np.pi / 2 + 2 * np.pi / 5,  # Top-right
            "color": "#E8F5E9",
            "edge_color": "#2E7D32",
            "description": "Forward, backprop, gradient\ndescent, hyperparameter tuning,\nPEFT, fine-tuning, RLHF"
        },
        {
            "name": "Model Inference",
            "angle": np.pi / 2 + 4 * np.pi / 5,  # Bottom-right
            "color": "#FFF3E0",
            "edge_color": "#E65100",
            "description": "Quantization, caching,\nONNX conversion, operator\nfusion, CUDA kernel optimization"
        },
        {
            "name": "Model Benchmarking",
            "angle": np.pi / 2 + 6 * np.pi / 5,  # Bottom-left
            "color": "#F3E5F5",
            "edge_color": "#7B1FA2",
            "description": "Precision/recall, engineering\nperformance, profiling,\nstress testing, scenario testing"
        },
        {
            "name": "Model Deployment",
            "angle": np.pi / 2 + 8 * np.pi / 5,  # Top-left
            "color": "#FFEBEE",
            "edge_color": "#C62828",
            "description": "Autoscaling, scheduling,\nload balancing, observability"
        },
    ]
    
    # Radius for stage boxes
    radius = 0.9
    box_width = 0.8
    box_height = 0.4
    
    # Draw stages
    stage_positions = []
    for i, stage in enumerate(stages):
        angle = stage["angle"]
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        stage_positions.append((x, y))
        
        # Create rounded rectangle for stage
        box = FancyBboxPatch(
            (x - box_width/2, y - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.02",
            facecolor=stage["color"],
            edgecolor=stage["edge_color"],
            linewidth=2.0,
            zorder=3
        )
        ax.add_patch(box)
        
        # Add stage name
        ax.text(
            x, y + 0.08,
            stage["name"],
            ha='center', va='center',
            fontsize=9, fontweight='bold',
            color=stage["edge_color"],
            zorder=4
        )
        
        # Add description
        ax.text(
            x, y - 0.05,
            stage["description"],
            ha='center', va='center',
            fontsize=7,
            color='#333333',
            zorder=4
        )
    
    # Draw arrows between stages (circular flow)
    arrow_style = dict(
        arrowstyle='->',
        lw=2.0,
        color='#333333',
        zorder=2,
        mutation_scale=20,  # Make arrowhead larger
        shrinkA=5,
        shrinkB=5
    )
    
    def get_box_intersection(cx, cy, target_x, target_y, box_w, box_h):
        """Find where line from box center to target intersects box boundary"""
        dx = target_x - cx
        dy = target_y - cy
        
        if abs(dx) < 1e-10:  # Vertical line
            if dy > 0:
                return (cx, cy + box_h/2)
            else:
                return (cx, cy - box_h/2)
        if abs(dy) < 1e-10:  # Horizontal line
            if dx > 0:
                return (cx + box_w/2, cy)
            else:
                return (cx - box_w/2, cy)
        
        # Calculate intersections with all four edges
        # Left edge: x = cx - box_w/2
        t_left = (cx - box_w/2 - cx) / dx
        y_left = cy + t_left * dy
        if t_left > 0 and cy - box_h/2 <= y_left <= cy + box_h/2:
            return (cx - box_w/2, y_left)
        
        # Right edge: x = cx + box_w/2
        t_right = (cx + box_w/2 - cx) / dx
        y_right = cy + t_right * dy
        if t_right > 0 and cy - box_h/2 <= y_right <= cy + box_h/2:
            return (cx + box_w/2, y_right)
        
        # Top edge: y = cy + box_h/2
        t_top = (cy + box_h/2 - cy) / dy
        x_top = cx + t_top * dx
        if t_top > 0 and cx - box_w/2 <= x_top <= cx + box_w/2:
            return (x_top, cy + box_h/2)
        
        # Bottom edge: y = cy - box_h/2
        t_bottom = (cy - box_h/2 - cy) / dy
        x_bottom = cx + t_bottom * dx
        if t_bottom > 0 and cx - box_w/2 <= x_bottom <= cx + box_w/2:
            return (x_bottom, cy - box_h/2)
        
        # Fallback (shouldn't happen)
        return (cx, cy)
    
    for i in range(len(stages)):
        start_idx = i
        end_idx = (i + 1) % len(stages)
        
        start_x, start_y = stage_positions[start_idx]
        end_x, end_y = stage_positions[end_idx]
        
        # Get intersection points on box boundaries
        start_point_x, start_point_y = get_box_intersection(
            start_x, start_y, end_x, end_y, box_width, box_height
        )
        end_point_x, end_point_y = get_box_intersection(
            end_x, end_y, start_x, start_y, box_width, box_height
        )
        
        # Create straight arrow with visible arrowhead
        arrow = FancyArrowPatch(
            (start_point_x, start_point_y),
            (end_point_x, end_point_y),
            **arrow_style
        )
        ax.add_patch(arrow)
    
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Lifecycle diagram saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    create_lifecycle_diagram()
