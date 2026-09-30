import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "cpu": "CPU",
    },
    "zh": {
        "cpu": "CPU",
    }
}


def draw_cpu_shape(ax, center_x=0, center_y=0, scale=1.0, linewidth=None, show_text=True, text_fontsize=None, text_label="CPU"):
    """
    Draw a CPU icon shape on an existing axis.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to draw on
    center_x, center_y : float
        Center position of the CPU icon (default: 0, 0)
    scale : float
        Scale factor for the icon (default: 1.0)
    linewidth : float or None
        Line width for the chip border (default: None, uses 10*scale)
    show_text : bool
        Whether to show "CPU" text in the center (default: True)
    text_fontsize : float or None
        Font size for the text (default: None, uses 30*scale for smaller text)
    text_label : str
        Text label in the center (default: "CPU")
    """
    # Define colors - green for CPU
    icon_color = '#2E7D32'  # Professional green
    
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

    # 3. Add "CPU" Text (if requested)
    if show_text:
        if text_fontsize is None:
            fontsize = 30 * scale
        else:
            fontsize = text_fontsize
        ax.text(center_x, center_y, text_label, fontsize=fontsize, fontweight='bold', 
                color=icon_color, ha='center', va='center', fontfamily='sans-serif')


def draw(text: dict) -> plt.Figure:
    """Draw CPU icon."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')
    ax.axis('off')

    # Define colors
    bg_color = '#F8F9FA'
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Draw CPU shape
    draw_cpu_shape(ax, center_x=0, center_y=0, scale=1.0, show_text=True, text_label=text["cpu"])

    # Set plot limits
    limit = 0.55
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    return fig


if __name__ == '__main__':
    localized_figure(draw, "cpu", LABELS, __file__)
