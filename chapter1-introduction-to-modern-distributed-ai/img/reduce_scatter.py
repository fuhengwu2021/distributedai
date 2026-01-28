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

# Add shared directory to path for math4ai imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from math4ai import configure_math_fonts

# Configure matplotlib for math expressions
configure_math_fonts()

def draw_reduce_scatter_diagram():
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # --- Configuration ---
    lane_width = 1.0
    box_width = 0.7
    # The input is a tall bar containing 4 logical blocks
    block_height = 0.6
    total_in_height = block_height * 4
    
    # Vertical positioning
    # We want the input bars to be centered vertically roughly
    # Let's say the bottom of the lowest block is at y=0.8
    stack_bottom_y = 0.8
    
    rank_label_y = 3.6
    colors = ['#0077BB', '#CC3333', '#66AA55', '#Eebb00']
    
    start_x_left = 0.0
    start_x_right = 5.5

    # Helper to calculate y position of a specific block index (0=top, 3=bottom)
    def get_block_y(index):
        return stack_bottom_y + (3 - index) * block_height

    # --- Helper: Draw Lane ---
    def draw_lane(offset_x, rank_idx):
        x_center = offset_x + (rank_idx * lane_width) + (lane_width / 2)
        line_x = offset_x + (rank_idx * lane_width)
        ax.plot([line_x, line_x], [0, 4.2], color='black', linestyle='--', linewidth=1, zorder=0)
        ax.text(x_center, rank_label_y, f"rank {rank_idx}", ha='center', va='bottom', fontsize=14, color='black')
        return x_center

    # --- LEFT SIDE (Input) ---
    for i in range(4):
        cx = draw_lane(start_x_left, i)
        
        # Draw the main colored bar
        rect = patches.Rectangle(
            (cx - box_width/2, stack_bottom_y), 
            box_width, total_in_height, 
            linewidth=1.2, edgecolor='black', facecolor=colors[i], zorder=2
        )
        ax.add_patch(rect)
        
        # Draw dashed segment separators inside the bar
        for b in range(1, 4): # lines at 1, 2, 3 heights
            y_line = stack_bottom_y + b * block_height
            ax.plot([cx - box_width/2, cx + box_width/2], [y_line, y_line], 
                    color='black', linestyle='--', linewidth=0.8, zorder=3)
            
        # Label "inX" centered in the whole bar
        # Text color logic: Yellow (index 3) is light, so use black text. Others white.
        text_col = 'white' if i != 3 else 'black'
        ax.text(cx, stack_bottom_y + total_in_height/2, f"in{i}", 
                ha='center', va='center', fontsize=16, color=text_col)

    # Closing line left
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
        
        # Each rank gets ONE block (the result of the sum for that block index)
        # Position corresponds to the block index (staggered)
        y_pos = get_block_y(i)
        
        # White box for reduced data
        rect = patches.Rectangle(
            (cx - box_width/2, y_pos), 
            box_width, block_height, 
            linewidth=1.2, edgecolor='black', facecolor='white', zorder=2
        )
        ax.add_patch(rect)
        
        # Label "outX"
        ax.text(cx, y_pos + block_height/2, f"out{i}", 
                ha='center', va='center', fontsize=14, color='black')

    # Closing line right
    ax.plot([start_x_right + 4, start_x_right + 4], [0, 4.2], color='black', linestyle='--', linewidth=1)

    # --- Mathematical Annotation ---
    label_x = start_x_right + 2.0
    # LaTeX formula for Reduce-Scatter
    ax.text(label_x, 0.2, r'$\mathrm{outY}[i] = \sum(\mathrm{inX}[\mathrm{Y} \cdot \mathrm{count} + i])$', 
            ha='center', va='center', fontsize=16)

    # --- Final Layout ---
    ax.set_xlim(-0.2, start_x_right + 4.2)
    ax.set_ylim(0, 4.2)
    ax.axis('off')
    plt.tight_layout()

    # --- Save Figure ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_path = os.path.join(script_dir, f'{script_name}.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved figure to: {output_path}")

if __name__ == "__main__":
    draw_reduce_scatter_diagram()

