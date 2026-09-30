#!/usr/bin/env python3
"""
Generate Tokens Per Second (TPS) timeline diagram.
Shows how TPS is calculated across multiple concurrent requests.

Follows ~/mmb's localized_figure standard:
- Single implementation, multiple outputs (<stem>.png for English, <stem>_zh.png for Chinese)
- High-resolution (300 DPI) PNG exports
"""

import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Import localized_figure and styling from shared/figstyle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "title": "Tokens Per Second (TPS) Timeline",
        "t_start": "T_start",
        "tx": "Tx",
        "ty": "Ty",
        "t_end": "T_end",
        "l1": "L1",
        "l2": "L2",
        "ln_1": "Ln-1",
        "ln": "Ln",
    },
    "zh": {
        "title": "每秒生成 Token 吞吐量（TPS）时间线分布",
        "t_start": "T_start",
        "tx": "Tx",
        "ty": "Ty",
        "t_end": "T_end",
        "l1": "L1",
        "l2": "L2",
        "ln_1": "Ln-1",
        "ln": "Ln",
    }
}


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')

    line_color = '#6B7280'
    circle_color = '#9CA3AF'
    text_color = '#374151'
    dashed_color = '#9CA3AF'

    y_main = 4.0
    y_spacing = 0.6

    # Main timeline
    ax.plot([0.5, 11.5], [y_main, y_main], color=line_color, linewidth=1.5, zorder=1)

    # Timeline markers
    markers = [
        (0.5, text["t_start"]),
        (1.5, text["tx"]),
        (9.5, text["ty"]),
        (11.5, text["t_end"])
    ]

    for x, label in markers:
        circle = Circle((x, y_main), 0.08, facecolor='white', edgecolor=circle_color, linewidth=1.5, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y_main + 0.3, label, ha='center', va='bottom', fontsize=10, color=text_color)

    # Dashed vertical lines at Tx and Ty
    ax.plot([1.5, 1.5], [y_main, 0.5], color=dashed_color, linewidth=1, linestyle='--', zorder=0)
    ax.plot([9.5, 9.5], [y_main, 0.5], color=dashed_color, linewidth=1, linestyle='--', zorder=0)

    # Request lines
    requests = [
        (1.5, 3.0, text["l1"], y_main - y_spacing),
        (2.0, 4.0, text["l2"], y_main - 2*y_spacing),
        (7.5, 9.0, text["ln_1"], y_main - 3*y_spacing),
        (8.5, 10.5, text["ln"], y_main - 4*y_spacing),
    ]

    for x_start, x_end, label, y in requests:
        ax.plot([x_start, x_start], [y_main, y], color=line_color, linewidth=1, zorder=1)
        ax.plot([x_start, x_end], [y, y], color=line_color, linewidth=1, zorder=1)
        circle_start = Circle((x_start, y_main), 0.08, facecolor='white', edgecolor=circle_color, linewidth=1.5, zorder=2)
        ax.add_patch(circle_start)
        circle_end = Circle((x_end, y), 0.08, facecolor='white', edgecolor=circle_color, linewidth=1.5, zorder=2)
        ax.add_patch(circle_end)
        ax.text(x_end + 0.2, y + 0.15, label, ha='left', va='bottom', fontsize=10, color=text_color)

    # Ellipsis in the middle
    ax.text(5.5, y_main - 2.5*y_spacing, '...', ha='center', va='center', fontsize=14, color=text_color)

    # Title
    ax.text(6.0, 4.8, text["title"], 
            ha='center', va='center', fontsize=12, color=text_color, fontweight='bold')

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    localized_figure(draw, "tps_timeline", LABELS, __file__)
