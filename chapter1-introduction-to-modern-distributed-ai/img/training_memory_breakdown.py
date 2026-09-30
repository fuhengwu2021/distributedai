#!/usr/bin/env python3
"""
Visualize training memory requirements breakdown for a 7B parameter model.
Left subplot: FP32 (Float32) precision - single stacked bar showing 120-128GB range
Right subplot: BF16 (Bfloat16) precision - single stacked bar showing 64-72GB range
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
        "components": ["Weights", "Gradients", "Optimizer", "Activations"],
        "x_label_fp32": "FP32 Precision",
        "x_label_bf16": "BF16 Precision",
        "y_label": "Memory (GB)",
        "title_fp32": "Training Memory Breakdown for 7B Model\n(FP32 + Adam Optimizer)",
        "title_bf16": "Training Memory Breakdown for 7B Model\n(BF16 + Adam Optimizer)",
        "gpu_limit": "A100 GPU (80GB)",
    },
    "zh": {
        "components": ["模型权重", "梯度", "优化器状态", "前向激活值"],
        "x_label_fp32": "FP32 精度",
        "x_label_bf16": "BF16 精度",
        "y_label": "显存占用 (GB)",
        "title_fp32": "7B 模型训练显存构成分解\n（FP32 + Adam 优化器）",
        "title_bf16": "7B 模型训练显存构成分解\n（BF16 + Adam 优化器）",
        "gpu_limit": "A100 GPU 显存上限 (80GB)",
    }
}


def draw(text: dict) -> plt.Figure:
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    components = text["components"]

    # Left plot: FP32
    fp32_memory = [28, 28, 56, 12]
    fp32_total_min = 28 + 28 + 56 + 8   # 120 GB
    fp32_total_max = 28 + 28 + 56 + 16  # 128 GB

    x_pos = 0
    width = 0.6
    bottom = 0
    for i, (comp, mem) in enumerate(zip(components, fp32_memory)):
        ax1.bar(x_pos, mem, width, bottom=bottom, 
                label=comp, color=colors[i], edgecolor='white', linewidth=1.5)
        if mem > 3:
            ax1.text(x_pos, bottom + mem/2, f'{int(mem)} GB', 
                     ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        bottom += mem

    ax1.text(x_pos, fp32_total_max, f'{fp32_total_min}-{fp32_total_max} GB', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax1.set_xlabel(text["x_label_fp32"], fontsize=12, fontweight='bold')
    ax1.set_ylabel(text["y_label"], fontsize=12, fontweight='bold')
    ax1.set_xticks([x_pos])
    ax1.set_xticklabels(['FP32'], fontsize=11)
    ax1.set_ylim(0, 140)
    ax1.set_title(text["title_fp32"], fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=80, color='red', linestyle='--', linewidth=2, alpha=0.7, label=text["gpu_limit"])
    ax1.legend(loc='center left', fontsize=9, framealpha=0.9, bbox_to_anchor=(1.02, 0.5))

    # Right plot: BF16
    bf16_memory = [14, 14, 28, 12]
    bf16_total_min = 14 + 14 + 28 + 8   # 64 GB
    bf16_total_max = 14 + 14 + 28 + 16  # 72 GB

    x_pos2 = 0
    width2 = 0.6
    bottom2 = 0
    for i, (comp, mem) in enumerate(zip(components, bf16_memory)):
        ax2.bar(x_pos2, mem, width2, bottom=bottom2, 
                label=comp, color=colors[i], edgecolor='white', linewidth=1.5)
        if mem > 3:
            ax2.text(x_pos2, bottom2 + mem/2, f'{int(mem)} GB', 
                     ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        bottom2 += mem

    ax2.text(x_pos2, bf16_total_max, f'{bf16_total_min}-{bf16_total_max} GB', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax2.set_xlabel(text["x_label_bf16"], fontsize=12, fontweight='bold')
    ax2.set_ylabel(text["y_label"], fontsize=12, fontweight='bold')
    ax2.set_xticks([x_pos2])
    ax2.set_xticklabels(['BF16'], fontsize=11)
    ax2.set_ylim(0, 80)
    ax2.set_title(text["title_bf16"], fontsize=13, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=80, color='red', linestyle='--', linewidth=2, alpha=0.7, label=text["gpu_limit"])
    ax2.legend(loc='center left', fontsize=9, framealpha=0.9, bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    return fig


if __name__ == '__main__':
    localized_figure(draw, "training_memory_breakdown", LABELS, __file__)
