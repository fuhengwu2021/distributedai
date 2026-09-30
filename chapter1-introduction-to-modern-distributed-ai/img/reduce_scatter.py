#!/usr/bin/env python3
"""
Diagram of an MPI Reduce-Scatter operation.
Visualizes a reduction where specific blocks from all ranks are summed 
and the results are scattered to corresponding ranks.
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
        "in": "in{i}",
        "out": "out{i}",
        "formula": r'$\mathrm{outY}[i] = \sum(\mathrm{inX}[\mathrm{Y} \cdot \mathrm{count} + i])$',
    },
    "zh": {
        "rank": "Rank {i}",
        "in": "输入{i}",
        "out": "输出{i}",
        "formula": r'$\mathrm{outY}[i] = \sum(\mathrm{inX}[\mathrm{Y} \cdot \mathrm{count} + i])$',
    }
}


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # --- Configuration ---
    lane_width = 1.0
    box_width = 0.7
    block_height = 0.6
    total_in_height = block_height * 4
    
    stack_bottom_y = 0.8
    rank_label_y = 3.6
    colors = ['#0077BB', '#CC3333', '#66AA55', '#Eebb00']
    
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
    for i in range(4):
        cx = draw_lane(start_x_left, i)
        
        rect = patches.Rectangle(
            (cx - box_width/2, stack_bottom_y), 
            box_width, total_in_height, 
            linewidth=1.2, edgecolor='black', facecolor=colors[i], zorder=2
        )
        ax.add_patch(rect)
        
        for b in range(1, 4):
            y_line = stack_bottom_y + b * block_height
            ax.plot([cx - box_width/2, cx + box_width/2], [y_line, y_line], 
                    color='black', linestyle='--', linewidth=0.8, zorder=3)
            
        text_col = 'white' if i != 3 else 'black'
        ax.text(cx, stack_bottom_y + total_in_height/2, text["in"].format(i=i), 
                ha='center', va='center', fontsize=16, color=text_col)

    ax.plot([4, 4], [0, 4.2], color='black', linestyle='--', linewidth=1)

    # --- ARROW ---
    arrow = patches.FancyArrowPatch(
        (4.2, 2.0), (5.2, 2.0),
        mutation_scale=30, color='gray', linewidth=0
    )
    ax.add_patch(arrow)

    # --- RIGHT SIDE (Output) ---
    for i in range(4):
        cx = draw_lane(start_x_right, i)
        y_pos = get_block_y(i)
        
        rect = patches.Rectangle(
            (cx - box_width/2, y_pos), 
            box_width, block_height, 
            linewidth=1.2, edgecolor='black', facecolor='white', zorder=2
        )
        ax.add_patch(rect)
        
        ax.text(cx, y_pos + block_height/2, text["out"].format(i=i), 
                ha='center', va='center', fontsize=14, color='black')

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
    localized_figure(draw, "reduce_scatter", LABELS, __file__)
