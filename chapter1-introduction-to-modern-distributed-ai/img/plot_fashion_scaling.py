#!/usr/bin/env python3
"""
Plot training time vs number of GPUs to show scaling efficiency.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Ensure shared directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "x_label": "Number of GPUs",
        "y_label": "Training Time (seconds)",
    },
    "zh": {
        "x_label": "GPU 数量",
        "y_label": "训练耗时（秒）",
    }
}

# Data from experiments
gpus = np.array([1, 2, 4, 6, 8])
times = np.array([8.78, 5.91, 3.69, 2.92, 2.44])


def draw(text: dict) -> plt.Figure:
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    ax1.plot(gpus, times, 'o-', linewidth=2, markersize=6, color='#2E86AB')
    ax1.set_xlabel(text["x_label"], fontsize=11)
    ax1.set_ylabel(text["y_label"], fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(gpus)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    localized_figure(draw, "plot_fashion_scaling", LABELS, __file__)
