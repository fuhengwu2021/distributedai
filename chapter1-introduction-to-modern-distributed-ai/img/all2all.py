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

# Add shared directory to path for math4ai imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from math4ai import configure_math_fonts, save_figure

# Configure matplotlib for math expressions
configure_math_fonts()

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by mixing it with white; amount 0 is pure color, 1 is white.
    Used to distinguish source ranks via shading.
    """
    try:
        c = mcolors.to_rgb(color)
    except ValueError:
        # Fallback if color format is tricky
        return color
    c = np.array(c)
    white = np.array([1.0, 1.0, 1.0])
    new_c = c * (1 - amount) + white * amount
    return new_c

def draw_alltoall_diagram():
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # --- Configuration ---
    lane_width = 1.0
    box_width = 0.7
    block_height = 0.6
    
    # Bottom of the stack
    stack_bottom_y = 0.8
    
    rank_label_y = 3.6
    
    # Base colors for DESTINATIONS: 0=Blue, 1=Red, 2=Green, 3=Yellow
    base_colors = ['#0077BB', '#CC3333', '#66AA55', '#Eebb00']
    
    start_x_left = 0.0
    start_x_right = 5.5

    # Helper for vertical position (0=top, 3=bottom)
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
    # Logic: On Rank 'i', the block at position 'j' is destined for Rank 'j'.
    # Color: Base color of Rank 'j', shaded by Source Rank 'i'.
    for rank_idx in range(4):
        cx = draw_lane(start_x_left, rank_idx)
        
        for block_idx in range(4): # 0=Top, 3=Bottom
            # In the diagram, the top block (0) is usually Blue (Dest 0), 
            # second (1) is Red (Dest 1), etc.
            dest_rank = block_idx
            source_rank = rank_idx
            
            # Shading logic: Rank 0 source is dark, Rank 3 source is light
            shade_factor = source_rank * 0.2 # 0.0, 0.2, 0.4, 0.6
            
            fill_color = lighten_color(base_colors[dest_rank], shade_factor)
            y_pos = get_block_y(block_idx)
            
            rect = patches.Rectangle(
                (cx - box_width/2, y_pos), 
                box_width, block_height, 
                linewidth=1.0, edgecolor='black', facecolor=fill_color, zorder=2
            )
            ax.add_patch(rect)
            
            # Optional: Add text label to middle rank to identify structure
            if rank_idx == 0 and block_idx == 2:
                # Just a label for the whole column if needed, but original doesn't have internal text
                ax.text(cx, stack_bottom_y + 2*block_height, f"in{rank_idx}", 
                        ha='center', va='center', fontsize=14, color='black', alpha=0.3)
            # Or replicate exact image text "inX" overlaying the blocks
            if block_idx == 2: # Middle of stack
                 ax.text(cx, stack_bottom_y + 2*block_height, f"in{rank_idx}", 
                        ha='center', va='center', fontsize=26, color='black', alpha=0.7)


    # Closing line left
    ax.plot([4, 4], [0, 4.2], color='black', linestyle='--', linewidth=1)

    # --- ARROW ---
    arrow = patches.FancyArrowPatch(
        (4.2, 2.0), (5.2, 2.0),
        mutation_scale=30, color='gray', linewidth=0
    )
    ax.add_patch(arrow)

    # --- RIGHT SIDE (Output) ---
    # Logic: On Rank 'i', we have collected all blocks destined for 'i'.
    # The stack is ordered by source rank (0 at top, 3 at bottom).
    for rank_idx in range(4):
        cx = draw_lane(start_x_right, rank_idx)
        
        # All blocks on this rank are "base_color[rank_idx]"
        # But they have different shades because they came from different sources
        for block_idx in range(4): # 0=Top, 3=Bottom
            # The top block came from Rank 0, the bottom from Rank 3
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

        # Label "outX" overlay
        ax.text(cx, stack_bottom_y + 2*block_height, f"out{rank_idx}", 
                ha='center', va='center', fontsize=26, color='black', alpha=0.7)

    # Closing line right
    ax.plot([start_x_right + 4, start_x_right + 4], [0, 4.2], color='black', linestyle='--', linewidth=1)

    # --- Mathematical Annotation ---
    label_x = start_x_right + 2.0
    # LaTeX formula for Alltoall
    # outX[Y*count + i] = inY[X*count + i]
    # Y is the loop over chunks/source ranks
    ax.text(label_x, 0.2, r'$\mathrm{outX}[\mathrm{Y} \cdot \mathrm{count} + i] = \mathrm{inY}[\mathrm{X} \cdot \mathrm{count} + i]$', 
            ha='center', va='center', fontsize=16)

    # --- Final Layout ---
    ax.set_xlim(-0.2, start_x_right + 4.2)
    ax.set_ylim(0, 4.2)
    ax.axis('off')
    plt.tight_layout()
    save_figure(__file__)

if __name__ == "__main__":
    draw_alltoall_diagram()


