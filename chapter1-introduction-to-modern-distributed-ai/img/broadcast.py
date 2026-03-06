import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from math4ai import save_figure

def draw_broadcast_diagram():
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Configuration
    box_width = 0.6
    box_height = 2.0
    box_y_bottom = 1.0  # Where the bottom of the box sits
    rank_labels_y = 3.6 # Height for "rank X" labels
    
    # --- Helper function to draw a single process column ---
    def draw_lane(x_center, label_text, box_text=None, is_root=False):
        # Draw vertical dashed separators (left of the lane)
        ax.plot([x_center - 0.5, x_center - 0.5], [0, 4], 
                color='black', linestyle='--', linewidth=1, zorder=0)
        
        # Draw the "rank X" label
        ax.text(x_center, rank_labels_y, label_text, 
                ha='center', va='bottom', fontsize=12, color='black')
        
        # If it's the root, add "(root)" label
        if is_root:
            ax.text(x_center, rank_labels_y - 0.3, "(root)", 
                    ha='center', va='top', fontsize=11, color='black')

        # Draw the data box if text is provided
        if box_text:
            rect = patches.Rectangle(
                (x_center - box_width/2, box_y_bottom), 
                box_width, box_height, 
                linewidth=1.5, edgecolor='black', facecolor='white', zorder=2
            )
            ax.add_patch(rect)
            ax.text(x_center, box_y_bottom + box_height/2, box_text, 
                    ha='center', va='center', fontsize=14)

    # --- LEFT SIDE (Input State) ---
    # Ranks 0 to 3
    for i in range(4):
        is_root = (i == 2)
        box_text = "in" if is_root else None
        draw_lane(i, f"rank {i}", box_text, is_root)
    
    # Closing dashed line for the left group
    ax.plot([3.5, 3.5], [0, 4], color='black', linestyle='--', linewidth=1)

    # --- CENTRAL ARROW ---
    # Draw a custom polygon arrow or use FancyArrow
    arrow_x_start = 3.8
    arrow_x_end = 4.8
    arrow_y = 2.0
    
    arrow = patches.FancyArrowPatch(
        (arrow_x_start, arrow_y), (arrow_x_end, arrow_y),
        mutation_scale=40, 
        color='gray',
        linewidth=0 # No outline
    )
    ax.add_patch(arrow)

    # --- RIGHT SIDE (Output State) ---
    offset = 5.5  # Shift the right group over
    
    for i in range(4):
        draw_lane(offset + i, f"rank {i}", "out")
        
    # Closing dashed line for the right group
    ax.plot([offset + 3.5, offset + 3.5], [0, 4], color='black', linestyle='--', linewidth=1)

    # --- Mathematical Label ---
    # "out[i] = in[i]" placed below the right group
    # We position it roughly centered under the right group
    center_right_group = offset + 1.5
    ax.text(center_right_group, 0.2, "out[i] = in[i]", 
            ha='center', va='center', fontsize=16, fontfamily='sans-serif')

    # --- Final Layout Adjustments ---
    ax.set_xlim(-0.6, offset + 4.0)
    ax.set_ylim(0, 4.2)
    ax.axis('off')  # Turn off the actual plot axes (ticks, spines)
    
    plt.tight_layout()
    save_figure(__file__)

if __name__ == "__main__":
    draw_broadcast_diagram()
    