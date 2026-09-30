#!/usr/bin/env python3
"""
Scaling Efficiency Visualization
Shows ideal linear scaling vs actual scaling with efficiency percentages.

Follows ~/mmb's localized_figure standard:
- Single implementation, multiple outputs (<stem>.png for English, <stem>_zh.png for Chinese)
- High-resolution (300 DPI) PNG exports
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Import localized_figure and styling from shared/figstyle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "legend_ideal": "Ideal (Linear)",
        "legend_actual": "Actual",
        "legend_overhead": "Communication Overhead",
        "xlabel": "Number of GPUs",
        "ylabel_throughput": "Throughput (samples/sec)",
        "title_throughput": "Throughput Scaling",
        "ylabel_efficiency": "Scaling Efficiency (%)",
        "title_efficiency": "Scaling Efficiency by GPU Count",
        "legend_excellent": "Excellent (>90%)",
        "legend_good": "Good (>70%)",
        "legend_mod": "Moderate (>50%)",
        "amdahl": "Amdahl's Law: Speedup = 1 / (S + P/N)",
    },
    "zh": {
        "legend_ideal": "理想线性加速 (Ideal Linear)",
        "legend_actual": "实际测试吞吐 (Actual)",
        "legend_overhead": "通信与同步开销损失",
        "xlabel": "GPU 数量（卡数）",
        "ylabel_throughput": "吞吐量 (samples/sec)",
        "title_throughput": "多卡训练吞吐扩展曲线",
        "ylabel_efficiency": "并行扩展效率 (%)",
        "title_efficiency": "不同卡数下的并行扩展效率",
        "legend_excellent": "优秀 (>90%)",
        "legend_good": "良好 (>70%)",
        "legend_mod": "一般 (>50%)",
        "amdahl": "阿姆达尔定律：Speedup = 1 / (S + P/N)",
    }
}


def draw(text: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Data
    gpus = np.array([1, 2, 4, 8, 16, 32])

    # Throughput scaling
    baseline_throughput = 100  # samples/sec with 1 GPU
    ideal_throughput = baseline_throughput * gpus

    # Realistic scaling with diminishing returns
    efficiencies = np.array([100, 95, 88, 81, 72, 62])  # %
    actual_throughput = ideal_throughput * efficiencies / 100

    # Left plot: Throughput vs GPUs
    ax1 = axes[0]
    ax1.plot(gpus, ideal_throughput, 'b--', linewidth=2, marker='o', 
             markersize=8, label=text["legend_ideal"], alpha=0.7)
    ax1.plot(gpus, actual_throughput, 'g-', linewidth=2.5, marker='s',
             markersize=8, label=text["legend_actual"], color='#2E7D32')

    # Fill the gap
    ax1.fill_between(gpus, actual_throughput, ideal_throughput, 
                     alpha=0.2, color='red', label=text["legend_overhead"])

    ax1.set_xlabel(text["xlabel"], fontsize=12)
    ax1.set_ylabel(text["ylabel_throughput"], fontsize=12)
    ax1.set_title(text["title_throughput"], fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(gpus)
    ax1.set_xticklabels(gpus)

    # Annotate efficiency at each point
    for i, (g, t, e) in enumerate(zip(gpus, actual_throughput, efficiencies)):
        if i > 0:  # Skip 1 GPU
            ax1.annotate(f'{e}%', (g, t), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=9, color='#1565C0')

    # Right plot: Scaling Efficiency
    ax2 = axes[1]

    # Bar colors based on efficiency
    colors = []
    for e in efficiencies:
        if e >= 90:
            colors.append('#4CAF50')  # Green - Excellent
        elif e >= 70:
            colors.append('#FFC107')  # Yellow - Good
        elif e >= 50:
            colors.append('#FF9800')  # Orange - Moderate
        else:
            colors.append('#F44336')  # Red - Poor

    bars = ax2.bar(range(len(gpus)), efficiencies, color=colors, edgecolor='#37474F', linewidth=1.5)

    # Add value labels on bars
    for bar, e in zip(bars, efficiencies):
        height = bar.get_height()
        ax2.annotate(f'{e}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_xlabel(text["xlabel"], fontsize=12)
    ax2.set_ylabel(text["ylabel_efficiency"], fontsize=12)
    ax2.set_title(text["title_efficiency"], fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(gpus)))
    ax2.set_xticklabels(gpus)
    ax2.set_ylim(0, 110)
    ax2.axhline(y=90, color='#4CAF50', linestyle='--', alpha=0.5, label=text["legend_excellent"])
    ax2.axhline(y=70, color='#FFC107', linestyle='--', alpha=0.5, label=text["legend_good"])
    ax2.axhline(y=50, color='#FF9800', linestyle='--', alpha=0.5, label=text["legend_mod"])
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add Amdahl's Law annotation
    ax2.text(4.5, 105, text["amdahl"], 
             fontsize=10, style='italic', ha='center',
             bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    localized_figure(draw, "scaling_efficiency", LABELS, __file__)
