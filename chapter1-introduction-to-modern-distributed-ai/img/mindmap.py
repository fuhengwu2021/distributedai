import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

from mathicon import render_mindmap


def draw_allreduce_icon(ax, center_x, center_y, size=0.32):
    """Draws an AllReduce ring: values flow around ranks and reduce to one result."""
    cx, cy = center_x, center_y
    r_ring = size * 0.62
    r_node = size * 0.16
    node_color = "#3a5a40"
    arrow_color = "#8a9a8f"

    positions = [
        (cx, cy + r_ring),
        (cx + r_ring, cy),
        (cx, cy - r_ring),
        (cx - r_ring, cy),
    ]
    for i in range(4):
        start = positions[i]
        end = positions[(i + 1) % 4]
        arrow = FancyArrowPatch(
            start, end,
            connectionstyle="arc3,rad=0.35",
            arrowstyle="-|>", mutation_scale=8,
            color=arrow_color, lw=1.1, zorder=4,
        )
        ax.add_patch(arrow)
    for (x, y) in positions:
        ax.add_patch(patches.Circle(
            (x, y), r_node, facecolor=node_color, edgecolor="black",
            linewidth=0.8, zorder=6,
        ))
    ax.text(cx, cy, r"$\Sigma$", ha="center", va="center",
            fontsize=9, color=node_color, zorder=7)


if __name__ == "__main__":
    render_mindmap(__file__)
