#!/usr/bin/env python3
"""
Diagram of an MPI Alltoall operation.
Visualizes a total exchange where every rank sends a distinct block 
to every other rank. Effectively a matrix transposition.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import os
import sys

# Ensure shared directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "rank": "rank {i}",
        "in": "in{i}",
        "out": "out{i}",
        "formula": r'$\mathrm{outX}[\mathrm{Y} \cdot \mathrm{count} + i] = \mathrm{inY}[\mathrm{X} \cdot \mathrm{count} + i]$',
    },
    "zh": {
        "rank": "Rank {i}",
        "in": "输入{i}",
        "out": "输出{i}",
        "formula": r'$\mathrm{outX}[\mathrm{Y} \cdot \mathrm{count} + i] = \mathrm{inY}[\mathrm{X} \cdot \mathrm{count} + i]$',
    }
}


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by mixing it with white; amount 0 is pure color, 1 is white.
    Used to distinguish source ranks via shading.
    """
    try:
        c = mcolors.to_rgb(color)
    except ValueError:
        return color
    c = np.array(c)
    white = np.array([1.0, 1.0, 1.0])
    new_c = c * (1 - amount) + white * amount
    return new_c


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # --- Configuration ---
    lane_width = 1.0
    box_width = 0.7
    block_height = 0.6
    
    stack_bottom_y = 0.8
    rank_label_y = 3.6
    base_colors = ['#0077BB', '#CC3333', '#66AA55', '#Eebb00']
    
    start_x_left = 0.0
    start_x_right = 5.5

    def get_block_y(index):
        return stack_bottom_y + (3 - index) * block_height

    def draw_lane(offset_x, rank_idx):
        x_center = offset_x + (rank_idx * lane_width) + (lane_width / 2)
        line_x = offset_x + (rank_idx * lane_width)
        ax.plot([line_x, line_x], [0, 4.2], color='black', linestyle='--', linewidth=1, zorder=0)
        ax.text(x_center, rank_label_y, text["rank"].format(i=rank_idx), ha='center', va='bottom', fontsize=14, color='black')
        return x_center

    # --- LEFT SIDE (Input) ---
    for rank_idx in range(4):
        cx = draw_lane(start_x_left, rank_idx)
        
        for block_idx in range(4):
            dest_rank = block_idx
            source_rank = rank_idx
            shade_factor = source_rank * 0.2
            fill_color = lighten_color(base_colors[dest_rank], shade_factor)
            y_pos = get_block_y(block_idx)
            
            rect = patches.Rectangle(
                (cx - box_width/2, y_pos), 
                box_width, block_height, 
                linewidth=1.0, edgecolor='black', facecolor=fill_color, zorder=2
            )
            ax.add_patch(rect)
            
        ax.text(cx, stack_bottom_y + 2*block_height, text["in"].format(i=rank_idx), 
                ha='center', va='center', fontsize=26, color='black', alpha=0.7)

    ax.plot([4, 4], [0, 4.2], color='black', linestyle='--', linewidth=1)

    # --- ARROW ---
    arrow = patches.FancyArrowPatch(
        (4.2, 2.0), (5.2, 2.0),
        mutation_scale=30, color='gray', linewidth=0
    )
    ax.add_patch(arrow)

    # --- RIGHT SIDE (Output) ---
    for rank_idx in range(4):
        cx = draw_lane(start_x_right, rank_idx)
        
        for block_idx in range(4):
            source_rank = block_idx
            dest_rank = rank_idx
            shade_factor = source_rank * 0.2
            fill_color = lighten_color(base_colors[dest_rank], shade_factor)
            y_pos = get_block_y(block_idx)
            
            rect = patches.Rectangle(
                (cx - box_width/2, y_pos), 
                box_width, block_height, 
                linewidth=1.0, edgecolor='black', facecolor=fill_color, zorder=2
            )
            ax.add_patch(rect)

        ax.text(cx, stack_bottom_y + 2*block_height, text["out"].format(i=rank_idx), 
                ha='center', va='center', fontsize=26, color='black', alpha=0.7)

    ax.plot([start_x_right + 4, start_x_right + 4], [0, 4.2], color='black', linestyle='--', linewidth=1)

    # --- Mathematical Annotation ---
    label_x = start_x_right + 2.0
    ax.text(label_x, 0.2, text["formula"], 
            ha='center', va='center', fontsize=16)

    ax.set_xlim(-0.2, start_x_right + 4.2)
    ax.set_ylim(0, 4.2)
    ax.axis('off')
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    localized_figure(draw, "all2all", LABELS, __file__)
