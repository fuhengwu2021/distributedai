"""
CPU-GPU Interaction Diagram

This diagram illustrates how CPU and GPU interact during distributed training,
showing kernel launches, memory transfers, and PCIe connections.
"""

import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure
from cpu import draw_cpu_shape
from gpu import draw_gpu_shape

LABELS = {
    "en": {
        "pcie": "PCIe Gen4/5",
        "control": "Control: CUDA Kernels",
        "data": "Data: Memory Transfers",
        "cpu": "CPU",
        "gpu": "GPU",
    },
    "zh": {
        "pcie": "PCIe Gen4/5",
        "control": "控制流：下发 CUDA Kernel",
        "data": "数据流：主机/设备内存传输",
        "cpu": "CPU",
        "gpu": "GPU",
    }
}

# Configuration
ICON_SIZE = 1.2
SPACING = 2.5
CENTER_Y = 2.0


def draw(text: dict) -> plt.Figure:
    data_width = 4.0
    data_height = 2.0
    data_aspect = data_width / data_height

    fig_height = 2.5
    fig_width = fig_height * data_aspect
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    # Positions
    cpu_x = 1.5
    gpu_x = cpu_x + SPACING

    # Draw CPU
    draw_cpu_shape(ax, center_x=cpu_x, center_y=CENTER_Y, scale=ICON_SIZE, linewidth=2.0,
                   show_text=True, text_fontsize=ICON_SIZE * 20, text_label=text["cpu"])

    # Draw GPU
    draw_gpu_shape(ax, center_x=gpu_x, center_y=CENTER_Y, scale=ICON_SIZE, linewidth=2.0,
                   show_text=True, text_fontsize=ICON_SIZE * 20, text_label=text["gpu"])

    # PCIe Connection (bidirectional arrow)
    arrow_start_x = cpu_x + ICON_SIZE * 0.6
    arrow_end_x = gpu_x - ICON_SIZE * 0.6

    conn_pcie = ConnectionPatch(
        (arrow_start_x, CENTER_Y), (arrow_end_x, CENTER_Y),
        "data", "data", arrowstyle='<->', mutation_scale=20,
        linewidth=2.5, color='#E67E22', zorder=2
    )
    ax.add_patch(conn_pcie)

    # PCIe Label
    ax.text((cpu_x + gpu_x) / 2, CENTER_Y + 0.25, text["pcie"],
            fontsize=10, ha='center', va='center', fontweight='bold', color='#E67E22',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#E67E22', linewidth=1), zorder=5)

    # Data Flow Annotations
    # 1. CPU launches GPU kernels (CPU -> GPU)
    kernel_y = CENTER_Y + 0.8
    ax.annotate('', xy=(gpu_x - 0.5, kernel_y), xytext=(cpu_x + 0.4, kernel_y),
                arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2, ls='--'))
    ax.text((cpu_x + gpu_x) / 2, kernel_y - 0.1, text["control"],
            fontsize=9, ha='center', va='top', color='#27AE60', style='italic')

    # 2. Memory transfers (CPU <-> GPU)
    memory_y = CENTER_Y - 0.8
    ax.annotate('', xy=(gpu_x - 0.5, memory_y), xytext=(cpu_x + 0.4, memory_y),
                arrowprops=dict(arrowstyle='<->', color='#8E44AD', lw=2, ls='--'))
    ax.text((cpu_x + gpu_x) / 2, memory_y + 0.1, text["data"],
            fontsize=9, ha='center', va='bottom', color='#8E44AD', style='italic')

    ax.set_xlim(0.8, 4.8)
    ax.set_ylim(1.0, 3.0)
    plt.tight_layout(pad=0)
    return fig


if __name__ == '__main__':
    localized_figure(draw, "cpu_gpu_interaction", LABELS, __file__, pad_inches=0.0)