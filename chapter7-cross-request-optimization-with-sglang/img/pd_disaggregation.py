"""
Prefill/Decode Disaggregation Resource Utilization

Shows why PD disaggregation makes sense:
- Prefill is compute-bound (high compute utilization, moderate memory)
- Decode is memory-bound (low compute utilization, high memory bandwidth)

Bar chart comparing resource utilization for unified vs disaggregated deployment.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "categories": ['Compute\nUtilization', 'Memory\nBandwidth'],
        "ylabel": 'Utilization (%)',
        "prefill_title": 'Prefill Phase',
        "prefill_sub": '(Compute-Bound)',
        "prefill_annot": 'Parallel token\nprocessing',
        "decode_title": 'Decode Phase',
        "decode_sub": '(Memory-Bound)',
        "decode_annot": 'KV cache\nloading',
    },
    "zh": {
        "categories": ['算力利用率', '显存带宽'],
        "ylabel": '利用率 (%)',
        "prefill_title": 'Prefill 阶段',
        "prefill_sub": '（算力密集）',
        "prefill_annot": '并行 Token\n批量计算',
        "decode_title": 'Decode 阶段',
        "decode_sub": '（访存密集）',
        "decode_annot": 'KV Cache\n逐字加载',
    }
}


def draw(text: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.))
    
    # Colors
    compute_color = '#4A90D9'  # Blue
    memory_color = '#F5A623'   # Orange
    
    # Data
    categories = text["categories"]
    
    # Left plot: Prefill characteristics
    ax1 = axes[0]
    prefill_values = [85, 40]  # High compute, moderate memory
    bars1 = ax1.bar(categories, prefill_values, color=[compute_color, memory_color],
                    edgecolor='white', linewidth=2, width=0.6)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel(text["ylabel"], fontsize=14)
    ax1.text(0.5, 1.05, text["prefill_title"], ha='center', va='bottom', 
             transform=ax1.transAxes, fontsize=14, fontweight='bold')
    ax1.text(0.5, 0.98, text["prefill_sub"], ha='center', va='bottom',
             transform=ax1.transAxes, fontsize=14, color='#666', style='italic')
    
    # Add value labels
    for bar, val in zip(bars1, prefill_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add annotation for prefill
    ax1.annotate(text["prefill_annot"], xy=(0, 75), xytext=(0.55, 55),
                fontsize=14, ha='center', color='#333',
                arrowprops=dict(arrowstyle='->', color='#666', lw=1))
    
    ax1.tick_params(labelsize=11)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    # Right plot: Decode characteristics
    ax2 = axes[1]
    decode_values = [25, 90]  # Low compute, high memory
    bars2 = ax2.bar(categories, decode_values, color=[compute_color, memory_color],
                    edgecolor='white', linewidth=2, width=0.6)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel(text["ylabel"], fontsize=14)
    ax2.text(0.5, 1.05, text["decode_title"], ha='center', va='bottom',
             transform=ax2.transAxes, fontsize=14, fontweight='bold')
    ax2.text(0.5, 0.98, text["decode_sub"], ha='center', va='bottom',
             transform=ax2.transAxes, fontsize=14, color='#666', style='italic')
    
    # Add value labels
    for bar, val in zip(bars2, decode_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add annotation for decode
    ax2.annotate(text["decode_annot"], xy=(1, 80), xytext=(0.5, 60),
                fontsize=14, ha='center', color='#333',
                arrowprops=dict(arrowstyle='->', color='#666', lw=1))
    
    ax2.tick_params(labelsize=11)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout(pad=0.05)
    return fig


if __name__ == '__main__':
    localized_figure(draw, "pd_disaggregation", LABELS, __file__, pad_inches=0.05, use_math_fonts=True)
