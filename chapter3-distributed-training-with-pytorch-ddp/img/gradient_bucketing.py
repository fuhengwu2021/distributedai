"""
Gradient bucketing in DDP: parameter segments grouped into buckets by size.
Bucket boundaries shown as vertical dashed lines (e.g. every 25 MB).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "param": "P{i}",
        "xlabel": "Parameter size (MB)",
    },
    "zh": {
        "param": "P{i}",
        "xlabel": "参数体积 (MB)",
    },
}


def draw(text: dict) -> plt.Figure:
    # Parameter block sizes (MB) – sum > 25 so bucket boundaries appear
    param_sizes = np.array([5, 10, 8, 6, 12, 4, 9])
    bucket_cap_mb = 25

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])

    colors = ["#0096d6", "#d62728", "#6aa84f", "#ffd11a", "#9467bd", "#8c564b", "#e377c2"]
    start = 0
    for i, size in enumerate(param_sizes):
        ax.barh(0, size, left=start, height=0.5, color=colors[i % len(colors)], edgecolor="black", linewidth=0.8)
        ax.text(start + size / 2, 0, text["param"].format(i=i), ha="center", va="center", fontsize=11)
        start += size

    # Bucket boundaries (every bucket_cap_mb)
    total = sum(param_sizes)
    for boundary in range(bucket_cap_mb, int(total), bucket_cap_mb):
        ax.axvline(boundary, linestyle="--", color="gray", linewidth=1.5)

    ax.set_xlabel(text["xlabel"], fontsize=12)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    localized_figure(draw, "gradient_bucketing", LABELS, __file__)
