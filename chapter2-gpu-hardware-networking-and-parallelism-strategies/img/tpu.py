import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "tpu": "TPU",
    },
    "zh": {
        "tpu": "TPU",
    }
}


def draw(text: dict) -> plt.Figure:
    """Draw TPU icon."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')
    ax.axis('off')

    # Define colors - orange/red for TPU
    bg_color = '#F8F9FA'
    icon_color = '#E65100'  # Professional orange
    
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # 1. Draw Main Chip Body (Rounded Rectangle)
    chip_size = 0.6
    chip = patches.FancyBboxPatch(
        (-chip_size/2, -chip_size/2), chip_size, chip_size,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=10, edgecolor=icon_color, facecolor='none'
    )
    ax.add_patch(chip)

    # 2. Draw Pins (rectangles on four sides)
    pin_w, pin_l = 0.025, 0.12
    num_pins = 5
    spacing = chip_size / (num_pins + 1)

    for i in range(num_pins):
        pos = -chip_size/2 + (i + 1) * spacing
        # Top and Bottom
        ax.add_patch(patches.Rectangle((pos - pin_w/2, chip_size/2), pin_w, pin_l, color=icon_color))
        ax.add_patch(patches.Rectangle((pos - pin_w/2, -chip_size/2 - pin_l), pin_w, pin_l, color=icon_color))
        # Left and Right
        ax.add_patch(patches.Rectangle((-chip_size/2 - pin_l, pos - pin_w/2), pin_l, pin_w, color=icon_color))
        ax.add_patch(patches.Rectangle((chip_size/2, pos - pin_w/2), pin_l, pin_w, color=icon_color))

    # 3. Add "TPU" Text
    ax.text(0, 0, text["tpu"], fontsize=60, fontweight='bold', color=icon_color,
            ha='center', va='center', fontfamily='sans-serif')

    # Set plot limits
    limit = 0.55
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    return fig


if __name__ == '__main__':
    localized_figure(draw, "tpu", LABELS, __file__)
