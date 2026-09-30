"""
GPU Memory Hierarchy Architecture Diagram

This diagram visualizes the GPU memory hierarchy, showing the tradeoffs between
speed, capacity, and latency across different memory levels.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "levels": ["Registers", "L1 / Shared Memory", "L2 Cache", "VRAM (HBM/GDDR)"],
        "capacities": ["~256 KB/SM", "~128 KB/SM", "40-96 MB", "16-80+ GB"],
        "bandwidths": [">100 TB/s", "~20 TB/s", "~5 TB/s", "1-3.5 TB/s"],
        "latencies": ["~1 cycle", "~30 cycles", "~200 cycles", "400-800 cycles"],
        "latency_indicator": "Latency ↑",
    },
    "zh": {
        "levels": ["寄存器 (Registers)", "L1 缓存 / 共享内存", "L2 缓存 (L2 Cache)", "全局显存 (HBM / GDDR)"],
        "capacities": ["~256 KB/SM", "~128 KB/SM", "40-96 MB", "16-80+ GB"],
        "bandwidths": [">100 TB/s", "~20 TB/s", "~5 TB/s", "1-3.5 TB/s"],
        "latencies": ["~1 周期", "~30 周期", "~200 周期", "400-800 周期"],
        "latency_indicator": "访问延迟 Latency ↑",
    }
}


def draw(text: dict) -> plt.Figure:
    levels = text["levels"]
    capacities = text["capacities"]
    bandwidths = text["bandwidths"]
    latencies = text["latencies"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, 8.5)
    ax.axis('off')

    colors = ['#ff4d4d', '#ff944d', '#4db8ff', '#4dff88']

    rev_levels = levels[::-1]
    rev_caps = capacities[::-1]
    rev_bws = bandwidths[::-1]
    rev_lats = latencies[::-1]
    rev_colors = colors[::-1]

    for i in range(len(rev_levels)):
        y_base = i * 2
        width = 8 - (i * 1.2)
        x_start = (12 - width) / 2
        
        rect = patches.FancyBboxPatch((x_start, y_base), width, 1.5, 
                                     boxstyle="round,pad=0.1",
                                     linewidth=2, edgecolor='black', 
                                     facecolor=rev_colors[i], alpha=0.8)
        ax.add_patch(rect)
        
        ax.text(6, y_base + 1.0, rev_levels[i], ha='center', va='center', 
                fontweight='bold', fontsize=16)
        spec_text = f"{rev_caps[i]}  |  {rev_bws[i]}  |  {rev_lats[i]}"
        ax.text(6, y_base + 0.5, spec_text, ha='center', va='center', 
                fontsize=13)

    ax.annotate('', xy=(1.0, 7.5), xytext=(1.0, 0.5),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2.5))
    ax.text(0.7, 4, text["latency_indicator"], rotation=90, 
            va='center', ha='center', fontweight='bold', fontsize=13)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    localized_figure(draw, "gpu_mem", LABELS, __file__, use_math_fonts=True)
