import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_gpu_shape(ax, center_x=0, center_y=0, scale=1.0, linewidth=None, show_text=True, text_fontsize=None):
    """
    Draw a GPU icon shape on an existing axis.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to draw on
    center_x, center_y : float
        Center position of the GPU icon (default: 0, 0)
    scale : float
        Scale factor for the icon (default: 1.0)
    linewidth : float or None
        Line width for the chip border (default: None, uses 10*scale)
    show_text : bool
        Whether to show "GPU" text in the center (default: True)
    text_fontsize : float or None
        Font size for the text (default: None, uses 30*scale for smaller text)
    """
    # Define colors
    icon_color = '#1E64AC'  # Professional blue
    
    # 1. Draw Main Chip Body (Rounded Rectangle)
    chip_size = 0.6 * scale
    if linewidth is None:
        linewidth = 10 * scale
    
    chip = patches.FancyBboxPatch(
        (center_x - chip_size/2, center_y - chip_size/2), 
        chip_size, chip_size,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=linewidth, edgecolor=icon_color, facecolor='none'
    )
    ax.add_patch(chip)

    # 2. Draw Pins (rectangles on four sides)
    pin_w, pin_l = 0.025 * scale, 0.12 * scale
    num_pins = 5
    spacing = chip_size / (num_pins + 1)

    for i in range(num_pins):
        pos = -chip_size/2 + (i + 1) * spacing
        # Top and Bottom
        ax.add_patch(patches.Rectangle(
            (center_x + pos - pin_w/2, center_y + chip_size/2), 
            pin_w, pin_l, color=icon_color
        ))
        ax.add_patch(patches.Rectangle(
            (center_x + pos - pin_w/2, center_y - chip_size/2 - pin_l), 
            pin_w, pin_l, color=icon_color
        ))
        # Left and Right
        ax.add_patch(patches.Rectangle(
            (center_x - chip_size/2 - pin_l, center_y + pos - pin_w/2), 
            pin_l, pin_w, color=icon_color
        ))
        ax.add_patch(patches.Rectangle(
            (center_x + chip_size/2, center_y + pos - pin_w/2), 
            pin_l, pin_w, color=icon_color
        ))

    # 3. Add "GPU" Text (if requested)
    if show_text:
        if text_fontsize is None:
            fontsize = 30 * scale  # Smaller font size (was 60)
        else:
            fontsize = text_fontsize
        ax.text(center_x, center_y, 'GPU', fontsize=fontsize, fontweight='bold', 
                color=icon_color, ha='center', va='center', fontfamily='sans-serif')

def draw_gpu_icon(filename):
    """Draw GPU icon to a file (for standalone generation)."""
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')
    ax.axis('off')

    # Define colors
    bg_color = '#F8F9FA'
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Draw GPU shape
    draw_gpu_shape(ax, center_x=0, center_y=0, scale=1.0, show_text=True)

    # Set plot limits and save
    limit = 0.55
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.03)
    plt.close()

# Save figure (standard pattern: same name as script)
script_dir = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.splitext(os.path.basename(__file__))[0]
output_path = os.path.join(script_dir, f'{script_name}.png')
draw_gpu_icon(output_path)
print(f"Saved figure to: {output_path}")
