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

# Add shared directory to path for math4ai imports
# (Assumes this script is located in chapterX-topic/img/)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from math4ai import configure_math_fonts

# Configure matplotlib for math expressions
configure_math_fonts()

def draw_gather_diagram():
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # --- Configuration ---
    # Geometry
    lane_width = 1.0     # Width of one process column
    box_width = 0.7      # Width of the data block
    block_height = 0.6   # Height of a single data chunk
    rank_label_y = 3.5   # Vertical position for "rank X"
    
    # Offsets
    start_x_left = 0.0
    start_x_right = 5.5
    
    # Colors (Blue, Red, Green, Yellow)
    colors = ['#0077BB', '#CC3333', '#66AA55', '#Eebb00']
    
    # Vertical positions for the stack (top to bottom)
    # We define 4 slots. Slot 0 is top, Slot 3 is bottom.
    # To place them, we need y-coordinates.
    # Let's say bottom of stack is at y=0.8.
    stack_bottom_y = 0.8
    # y positions for blocks 3, 2, 1, 0 (from bottom up)
    # y[i] is the bottom coordinate of block i
    # We want block 0 at top.
    # block 3 bottom = stack_bottom_y
    # block 2 bottom = stack_bottom_y + block_height
    # ...
    # block i bottom = stack_bottom_y + (3-i)*block_height
    def get_block_y(index):
        return stack_bottom_y + (3 - index) * block_height

    # --- Helper: Draw a single process lane ---
    def draw_lane(offset_x, rank_idx, is_root=False):
        # Center x for this rank
        x_center = offset_x + (rank_idx * lane_width) + (lane_width / 2)
        
        # Vertical separator line (left side of lane)
        line_x = offset_x + (rank_idx * lane_width)
        ax.plot([line_x, line_x], [0, 4.2], 
                color='black', linestyle='--', linewidth=1, zorder=0)
        
        # Rank Label
        ax.text(x_center, rank_label_y, f"rank {rank_idx}", 
                ha='center', va='bottom', fontsize=14, color='black')
        
        # Root Label
        if is_root:
            ax.text(x_center, rank_label_y - .025, "(root)", 
                    ha='center', va='top', fontsize=12, color='black')
        
        return x_center

    # --- LEFT GROUP (Input) ---
    for i in range(4):
        cx = draw_lane(start_x_left, i)
        
        # Draw staggered input boxes
        # Each rank i has a box in the position corresponding to its final slot
        y_pos = get_block_y(i)
        
        rect = patches.Rectangle(
            (cx - box_width/2, y_pos), 
            box_width, block_height, 
            linewidth=1.2, edgecolor='black', facecolor=colors[i], zorder=2
        )
        ax.add_patch(rect)
        
        # Text "inX"
        text_col = 'white' if i != 3 else 'black' # Yellow is light, needs black text
        ax.text(cx, y_pos + block_height/2, f"in{i}", 
                ha='center', va='center', fontsize=14, color=text_col)

    # Closing line for left group
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
            # Draw the full stack
            for j in range(4): # 0 to 3
                y_pos = get_block_y(j)
                
                # Dashed lines between blocks for visual separation
                # (Bottom edge of current block, unless it's the very bottom)
                ls = '-'
                
                rect = patches.Rectangle(
                    (cx - box_width/2, y_pos), 
                    box_width, block_height, 
                    linewidth=1.2, edgecolor='black', facecolor=colors[j], linestyle=ls, zorder=2
                )
                ax.add_patch(rect)
            
            # Label "out" centered over the stack
            # Center of the stack is between block 1 and 2
            stack_center_y = stack_bottom_y + 2 * block_height
            ax.text(cx, stack_center_y, "out", 
                    ha='center', va='center', fontsize=26, color='black', fontweight='bold')

    # Closing line for right group
    ax.plot([start_x_right + 4, start_x_right + 4], [0, 4.2], color='black', linestyle='--', linewidth=1)

    # --- Mathematical Annotation ---
    label_x = start_x_right + 2.0
    # LaTeX formula for Gather
    ax.text(label_x, 0.2, r'$\mathrm{out}[Y \cdot \mathrm{count} + i] = \mathrm{inY}[i]$', 
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
    draw_gather_diagram()

