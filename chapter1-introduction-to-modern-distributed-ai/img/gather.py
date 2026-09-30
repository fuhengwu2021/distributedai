#!/usr/bin/env python3
"""
Diagram of an MPI Gather operation.
Visualizes data collection from all ranks (0-3) into a single ordered stack 
on the root rank (2).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

# Ensure shared directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "rank": "rank {i}",
        "root": "(root)",
        "in": "in{i}",
        "out": "out",
        "formula": r'$\mathrm{out}[Y \cdot \mathrm{count} + i] = \mathrm{inY}[i]$',
    },
    "zh": {
        "rank": "Rank {i}",
        "root": "(根节点)",
        "in": "输入{i}",
        "out": "输出",
        "formula": r'$\mathrm{out}[Y \cdot \mathrm{count} + i] = \mathrm{inY}[i]$',
    }
}


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # --- Configuration ---
    lane_width = 1.0     # Width of one process column
    box_width = 0.7      # Width of the data block
    block_height = 0.6   # Height of a single data chunk
    rank_label_y = 3.5   # Vertical position for "rank X"
    
    start_x_left = 0.0
    start_x_right = 5.5
    
    colors = ['#0077BB', '#CC3333', '#66AA55', '#Eebb00']
    stack_bottom_y = 0.8
    
    def get_block_y(index):
        return stack_bottom_y + (3 - index) * block_height

    def draw_lane(offset_x, rank_idx, is_root=False):
        x_center = offset_x + (rank_idx * lane_width) + (lane_width / 2)
        line_x = offset_x + (rank_idx * lane_width)
        ax.plot([line_x, line_x], [0, 4.2], 
                color='black', linestyle='--', linewidth=1, zorder=0)
        
        ax.text(x_center, rank_label_y, text["rank"].format(i=rank_idx), 
                ha='center', va='bottom', fontsize=14, color='black')
        
        if is_root:
            ax.text(x_center, rank_label_y - .025, text["root"], 
                    ha='center', va='top', fontsize=12, color='black')
        
        return x_center

    # --- LEFT GROUP (Input) ---
    for i in range(4):
        cx = draw_lane(start_x_left, i)
        y_pos = get_block_y(i)
        
        rect = patches.Rectangle(
            (cx - box_width/2, y_pos), 
            box_width, block_height, 
            linewidth=1.2, edgecolor='black', facecolor=colors[i], zorder=2
        )
        ax.add_patch(rect)
        
        text_col = 'white' if i != 3 else 'black'
        ax.text(cx, y_pos + block_height/2, text["in"].format(i=i), 
                ha='center', va='center', fontsize=14, color=text_col)

    ax.plot([4, 4], [0, 4.2], color='black', linestyle='--', linewidth=1)

    # --- ARROW ---
    arrow = patches.FancyArrowPatch(
        (4.2, 2.0), (5.2, 2.0),
        mutation_scale=30, color='gray', linewidth=0
    )
    ax.add_patch(arrow)

    # --- RIGHT GROUP (Output) ---
    for i in range(4):
        is_root = (i == 2)
        cx = draw_lane(start_x_right, i, is_root)
        
        if is_root:
            for j in range(4):
                y_pos = get_block_y(j)
                rect = patches.Rectangle(
                    (cx - box_width/2, y_pos), 
                    box_width, block_height, 
                    linewidth=1.2, edgecolor='black', facecolor=colors[j], linestyle='-', zorder=2
                )
                ax.add_patch(rect)
            
            stack_center_y = stack_bottom_y + 2 * block_height
            ax.text(cx, stack_center_y, text["out"], 
                    ha='center', va='center', fontsize=26, color='black', fontweight='bold')

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
    localized_figure(draw, "gather", LABELS, __file__)
