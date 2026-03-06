"""
Diagram of an MPI Reduce operation.
Visualizes data aggregation from multiple ranks (0-3) into a single root rank (2)
using a summation operation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

# Add shared directory to path for math4ai imports
# (Assumes this script is located in chapterX-topic/img/)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from math4ai import configure_math_fonts, save_figure

# Configure matplotlib for math expressions
configure_math_fonts()

def draw_reduce_diagram():
    # Create figure
    # Using a wider aspect ratio to accommodate the left->right flow
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # --- Configuration ---
    box_width = 0.65
    box_height = 2.0
    box_y_bottom = 0.8  # Bottom y-coordinate of the boxes
    rank_label_y = 3.3  # Height for "rank X" text
    lane_width = 1.0    # Visual width of one rank's "lane"
    
    # Colors for the input boxes (Blue, Red, Green, Yellow) to match the source image
    colors = ['#0077BB', '#CC3333', '#66AA55', '#Eebb00']
    
    # --- Helper: Draw a single process lane ---
    def draw_lane(offset_x, rank_idx, box_text=None, box_color=None, is_root=False):
        # Center x-coordinate for this rank
        x_center = offset_x + (rank_idx * lane_width) + (lane_width / 2)
        
        # 1. Vertical separator line (dashed) - drawn on the left of the lane
        line_x = offset_x + (rank_idx * lane_width)
        ax.plot([line_x, line_x], [0, 4], 
                color='black', linestyle='--', linewidth=1, zorder=0)
        
        # 2. Rank Label
        ax.text(x_center, rank_label_y, f"rank {rank_idx}", 
                ha='center', va='bottom', fontsize=14, color='black')
        
        # 3. Root Label (if applicable)
        if is_root:
            ax.text(x_center, rank_label_y - 0.25, "(root)", 
                    ha='center', va='top', fontsize=12, color='black')
            
        # 4. Data Box (if text provided)
        if box_text:
            # If color is provided, fill it. If not (white), just black edge.
            face_c = box_color if box_color else 'white'
            edge_c = 'black'
            
            rect = patches.Rectangle(
                (x_center - box_width/2, box_y_bottom), 
                box_width, box_height, 
                linewidth=1.5, edgecolor=edge_c, facecolor=face_c, zorder=2
            )
            ax.add_patch(rect)
            
            # Text inside the box
            # We use distinct coloring logic: white text for dark boxes, black for light
            text_color = 'white' if box_color and rank_idx < 2 else 'black'
            ax.text(x_center, box_y_bottom + box_height/2, box_text, 
                    ha='center', va='center', fontsize=16, color=text_color)

        return x_center # Return center for potential use

    # --- LEFT GROUP (Input State) ---
    start_x_left = 0
    for i in range(4):
        draw_lane(start_x_left, i, f"in{i}", colors[i])

    # Closing dashed line for left group
    ax.plot([4, 4], [0, 4], color='black', linestyle='--', linewidth=1)

    # --- ARROW (Transformation) ---
    # Centered between the two groups (approx x=4 to x=5.5)
    arrow = patches.FancyArrowPatch(
        (4.2, 1.8), (5.2, 1.8),
        mutation_scale=30, 
        color='gray',
        linewidth=0
    )
    ax.add_patch(arrow)

    # --- RIGHT GROUP (Output State) ---
    start_x_right = 5.5
    for i in range(4):
        # Only rank 2 gets a box in the output (Reduce operation)
        if i == 2:
            draw_lane(start_x_right, i, "out", box_color=None, is_root=True)
        else:
            draw_lane(start_x_right, i, box_text=None)

    # Closing dashed line for right group
    ax.plot([start_x_right + 4, start_x_right + 4], [0, 4], color='black', linestyle='--', linewidth=1)

    # --- Mathematical Annotation ---
    # Positioned under the right group
    # We use LaTeX formatting as per guidelines
    label_x = start_x_right + 2.0 # Center of right group
    ax.text(label_x, 0.2, r'$\mathrm{out}[i] = \sum(\mathrm{inX}[i])$', 
            ha='center', va='center', fontsize=18)

    # --- Final Layout & Styling ---
    ax.set_xlim(-0.2, start_x_right + 4.2)
    ax.set_ylim(0, 4.0)
    ax.axis('off')  # Turn off axes/ticks for diagrammatic look
    
    plt.tight_layout()
    save_figure(__file__)

if __name__ == "__main__":
    draw_reduce_diagram()