#!/usr/bin/env python3
"""
Training Iteration Breakdown Diagram
Shows time spent in each phase of distributed training.

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
        "phases": ["Forward", "Backward", "Communication", "Optimizer", "Data Loading"],
        "xlabel1": "Configuration",
        "ylabel1": "Time per Iteration (ms)",
        "title1": "Training Iteration Time Breakdown",
        "legend_compute": "Compute",
        "legend_comm": "Commun",
        "legend_other": "Other",
        "xlabel2": "Percentage of Iteration Time (%)",
        "ylabel2": "Configuration",
        "title2": "Time Distribution by Category",
    },
    "zh": {
        "phases": ["前向传播 (Forward)", "反向求导 (Backward)", "多卡通信 (Communication)", "优化器更新 (Optimizer)", "数据加载 (Data Loading)"],
        "xlabel1": "GPU 并行拓扑配置",
        "ylabel1": "单步迭代耗时 (ms)",
        "title1": "单步训练迭代耗时结构分解",
        "legend_compute": "算力计算 (Compute)",
        "legend_comm": "集合通信 (Communication)",
        "legend_other": "其他开销 (Other)",
        "xlabel2": "迭代耗时占比 (%)",
        "ylabel2": "GPU 并行拓扑配置",
        "title2": "计算与通信耗时占比随卡数演进",
    }
}


def draw(text: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Data for different GPU configurations
    configs = ['1 GPU', '4 GPUs', '8 GPUs', '16 GPUs']

    # Time breakdown (ms) for each phase
    data = {
        '1 GPU':   {'Forward': 45, 'Backward': 85, 'Communication': 0,  'Optimizer': 15, 'Data Loading': 10},
        '4 GPUs':  {'Forward': 12, 'Backward': 22, 'Communication': 8,  'Optimizer': 4,  'Data Loading': 8},
        '8 GPUs':  {'Forward': 6,  'Backward': 12, 'Communication': 12, 'Optimizer': 2,  'Data Loading': 6},
        '16 GPUs': {'Forward': 3,  'Backward': 6,  'Communication': 18, 'Optimizer': 1,  'Data Loading': 5},
    }

    raw_phases = ['Forward', 'Backward', 'Communication', 'Optimizer', 'Data Loading']
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#607D8B']

    # Left plot: Stacked bar chart
    ax1 = axes[0]

    x = np.arange(len(configs))
    width = 0.6
    bottom = np.zeros(len(configs))

    for raw_p, p_label, color in zip(raw_phases, text["phases"], colors):
        values = [data[config][raw_p] for config in configs]
        ax1.bar(x, values, width, label=p_label, bottom=bottom, color=color, edgecolor='white', linewidth=1)
        bottom += values

    ax1.set_xlabel(text["xlabel1"], fontsize=12)
    ax1.set_ylabel(text["ylabel1"], fontsize=12)
    ax1.set_title(text["title1"], fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add total time labels
    totals = [sum(data[config].values()) for config in configs]
    for i, total in enumerate(totals):
        ax1.annotate(f'{total}ms', (i, total), textcoords="offset points",
                    xytext=(0, 5), ha='center', fontsize=10, fontweight='bold')

    # Right plot: Communication overhead percentage
    ax2 = axes[1]

    comm_pct = []
    compute_pct = []
    other_pct = []

    for config in configs:
        total = sum(data[config].values())
        comm = data[config]['Communication']
        compute = data[config]['Forward'] + data[config]['Backward']
        other = data[config]['Optimizer'] + data[config]['Data Loading']

        comm_pct.append(comm / total * 100)
        compute_pct.append(compute / total * 100)
        other_pct.append(other / total * 100)

    # Stacked horizontal bar
    y = np.arange(len(configs))
    height = 0.5

    ax2.barh(y, compute_pct, height, label=text["legend_compute"], color='#4CAF50')
    ax2.barh(y, comm_pct, height, left=compute_pct, label=text["legend_comm"], color='#FF9800')
    ax2.barh(y, other_pct, height, left=np.array(compute_pct) + np.array(comm_pct), 
             label=text["legend_other"], color='#607D8B')

    ax2.set_xlabel(text["xlabel2"], fontsize=12)
    ax2.set_ylabel(text["ylabel2"], fontsize=12)
    ax2.set_title(text["title2"], fontsize=14, fontweight='bold')
    ax2.set_yticks(y)
    ax2.set_yticklabels(configs)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.set_xlim(0, 100)
    ax2.grid(True, alpha=0.3, axis='x')

    # Add percentage labels
    for i, (comp, comm) in enumerate(zip(compute_pct, comm_pct)):
        ax2.text(comp/2, i, f'{comp:.0f}%', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
        if comm > 5:
            ax2.text(comp + comm/2, i, f'{comm:.0f}%', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    localized_figure(draw, "training_breakdown", LABELS, __file__)
