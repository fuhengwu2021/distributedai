#!/usr/bin/env python3
"""
Visualize memory usage timeline during training.
Shows how weights, activations, gradients, and optimizer states occupy VRAM at different stages.
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
        "stages": ["Forward\nPass", "Backward\nPass", "Optimizer\nStep"],
        "comp_weights": "Weights",
        "comp_optimizer": "Optimizer States",
        "comp_activations": "Activations",
        "comp_gradients": "Gradients",
        "y_label": "Memory (GB)",
        "gpu_limit": "A100 GPU (80GB)",
        "peak_memory": "Peak Memory",
    },
    "zh": {
        "stages": ["前向传播", "反向传播", "优化器更新"],
        "comp_weights": "模型权重",
        "comp_optimizer": "优化器状态",
        "comp_activations": "前向激活值",
        "comp_gradients": "梯度",
        "y_label": "显存占用 (GB)",
        "gpu_limit": "A100 GPU 显存上限 (80GB)",
        "peak_memory": "显存峰值",
    }
}


def draw(text: dict) -> plt.Figure:
    weights = 14
    optimizer_states = 28
    activations_avg = 12
    gradients = 14

    stages = text["stages"]
    x_pos = np.arange(len(stages))

    colors = {
        'weights': '#2E86AB',
        'optimizer': '#A23B72',
        'activations': '#F18F01',
        'gradients': '#C73E1D'
    }

    fig, ax = plt.subplots(figsize=(6, 5))
    width = 0.4
    seen_labels = set()

    # Forward pass
    bars_forward = []
    labels_forward = [text["comp_weights"], text["comp_optimizer"], text["comp_activations"]]
    values_forward = [weights, optimizer_states, activations_avg]
    colors_forward = [colors['weights'], colors['optimizer'], colors['activations']]

    bottom = 0
    for i, (label, value, color) in enumerate(zip(labels_forward, values_forward, colors_forward)):
        label_to_use = label if label not in seen_labels else ''
        if label not in seen_labels:
            seen_labels.add(label)
        bar = ax.bar(x_pos[0], value, width, bottom=bottom, 
                     label=label_to_use, color=color, 
                     edgecolor='white', linewidth=1.5)
        bars_forward.append(bar)
        if value > 2:
            ax.text(x_pos[0], bottom + value/2, f'{int(value)} GB',
                    ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        bottom += value

    # Backward pass
    bars_backward = []
    labels_backward = [text["comp_weights"], text["comp_optimizer"], text["comp_activations"], text["comp_gradients"]]
    values_backward = [weights, optimizer_states, activations_avg, gradients]
    colors_backward = [colors['weights'], colors['optimizer'], colors['activations'], colors['gradients']]

    bottom = 0
    for i, (label, value, color) in enumerate(zip(labels_backward, values_backward, colors_backward)):
        label_to_use = label if label not in seen_labels else ''
        if label not in seen_labels:
            seen_labels.add(label)
        bar = ax.bar(x_pos[1], value, width, bottom=bottom,
                     label=label_to_use, color=color,
                     edgecolor='white', linewidth=1.5)
        bars_backward.append(bar)
        if value > 2:
            ax.text(x_pos[1], bottom + value/2, f'{int(value)} GB',
                    ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        bottom += value

    # Optimizer step
    bars_optimizer = []
    labels_optimizer = [text["comp_weights"], text["comp_optimizer"], text["comp_gradients"]]
    values_optimizer = [weights, optimizer_states, gradients]
    colors_optimizer = [colors['weights'], colors['optimizer'], colors['gradients']]

    bottom = 0
    for i, (label, value, color) in enumerate(zip(labels_optimizer, values_optimizer, colors_optimizer)):
        label_to_use = label if label not in seen_labels else ''
        if label not in seen_labels:
            seen_labels.add(label)
        bar = ax.bar(x_pos[2], value, width, bottom=bottom,
                     label=label_to_use, color=color,
                     edgecolor='white', linewidth=1.5)
        bars_optimizer.append(bar)
        if value > 2:
            ax.text(x_pos[2], bottom + value/2, f'{int(value)} GB',
                    ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        bottom += value

    total_forward = sum(values_forward)
    total_backward = sum(values_backward)
    total_optimizer = sum(values_optimizer)

    ax.text(x_pos[0], total_forward, f'{int(total_forward)} GB',
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.text(x_pos[1], total_backward, f'{int(total_backward)} GB',
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.text(x_pos[2], total_optimizer, f'{int(total_optimizer)} GB',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.axhline(y=80, color='red', linestyle='--', linewidth=2, alpha=0.7, label=text["gpu_limit"])

    ax.set_ylabel(text["y_label"], fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stages, fontsize=12)
    ax.set_ylim(0, 85)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

    ax.annotate(text["peak_memory"], xy=(x_pos[1], total_backward), xytext=(x_pos[1] + .8, total_backward + 5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, fontweight='bold', color='red', ha='center')

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    localized_figure(draw, "training_memory_timeline", LABELS, __file__)
