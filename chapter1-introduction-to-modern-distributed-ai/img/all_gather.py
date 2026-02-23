import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


from math4ai import configure_math_fonts

# Configure matplotlib for math expressions
configure_math_fonts()


def draw_allgather_diagram_v2():
    # Setup Figure
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-1, 5)
    ax.axis('off')

    # Constants
    ranks = 4
    colors = ['#0088cc', '#cc0000', '#669933', '#ffcc00'] # Blue, Red, Green, Yellow
    block_width = 0.8
    block_height = 0.8
    
    # Offsets for the two groups (Input vs Output)
    left_start_x = 0
    right_start_x = 6
    
    # Function to draw a single block
    def draw_block(x, y, color, label=None, label_color='black'):
        rect = patches.Rectangle(
            (x, y), block_width, block_height,
            linewidth=1, edgecolor='black', facecolor=color
        )
        ax.add_patch(rect)
        if label:
            ax.text(x + block_width/2, y + block_height/2, label,
                    ha='center', va='center', fontsize=12, color=label_color)

    # --- Draw Input State (Left) ---
    for r in range(ranks):
        # Rank Labels
        ax.text(left_start_x + r + 0.4, 4.2, f"rank {r}", 
                ha='center', fontsize=14, fontweight='bold')
        
        # Vertical Separators (Dashed)
        line_x = left_start_x + r
        ax.plot([line_x, line_x], [-0.5, 4.5], color='black', linestyle='--', linewidth=0.8)
        
        # Draw the staggered blocks
        # Rank 0 has block at top (index 0), Rank 1 at index 1, etc.
        # We invert Y so index 0 is at the top (y=3)
        y_pos = 3 - r 
        x_pos = left_start_x + r + (1 - block_width)/2
        
        draw_block(x_pos, y_pos, colors[r], f"in{r}")

    # Closing line for left section
    ax.plot([left_start_x + ranks, left_start_x + ranks], [-0.5, 4.5], color='black', linestyle='--', linewidth=0.8)

    # --- Draw Arrow ---
    ax.annotate(
        "", xy=(right_start_x - 0.5, 1.5), xytext=(left_start_x + ranks + 0.5, 1.5),
        arrowprops=dict(facecolor='grey', edgecolor='grey', shrink=0.05, width=10)
    )

    # --- Draw Output State (Right) ---
    for r in range(ranks):
        # Rank Labels
        ax.text(right_start_x + r + 0.4, 4.2, f"rank {r}", 
                ha='center', fontsize=14, fontweight='bold')
        
        # Vertical Separators
        line_x = right_start_x + r
        ax.plot([line_x, line_x], [-0.5, 4.5], color='black', linestyle='--', linewidth=0.8)
        
        # Draw full stacks for each rank
        x_pos = right_start_x + r + (1 - block_width)/2
        
        for i in range(ranks):
            y_pos = 3 - i
            # No labels inside the blocks now
            draw_block(x_pos, y_pos, colors[i], None)
            
        # Place "out" label in the center of the stack
        # The stack is from y=0 to y=3 + block_height. The center is around y=1.6
        ax.text(right_start_x + r + 0.5, 1.86, "out",
                ha='center', va='center', fontsize=24, color='black', fontweight='bold')

    # Closing line for right section
    ax.plot([right_start_x + ranks, right_start_x + ranks], [-0.5, 4.5], color='black', linestyle='--', linewidth=0.8)

    # --- Formula Text ---
    ax.text(8, -0.8, r"out[Y*count+i] = inY[i]", fontsize=14, ha='center', fontfamily='monospace')

    plt.tight_layout()
    
    # Save figure (standard pattern: same name as script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_path = os.path.join(script_dir, f'{script_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.03)
    print(f"Saved figure to: {output_path}")
    plt.close()  # Close to free memory

if __name__ == '__main__':
    draw_allgather_diagram_v2()