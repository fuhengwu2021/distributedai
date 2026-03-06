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

from math4ai import configure_math_fonts, save_figure
configure_math_fonts()


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.))
    
    # Colors
    compute_color = '#4A90D9'  # Blue
    memory_color = '#F5A623'   # Orange
    
    # Data
    categories = ['Compute\nUtilization', 'Memory\nBandwidth']
    
    # Left plot: Prefill characteristics
    ax1 = axes[0]
    prefill_values = [85, 40]  # High compute, moderate memory
    bars1 = ax1.bar(categories, prefill_values, color=[compute_color, memory_color],
                    edgecolor='white', linewidth=2, width=0.6)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Utilization (%)', fontsize=14)
    ax1.text(0.5, 1.05, 'Prefill Phase', ha='center', va='bottom', 
             transform=ax1.transAxes, fontsize=14, fontweight='bold')
    ax1.text(0.5, 0.98, '(Compute-Bound)', ha='center', va='bottom',
             transform=ax1.transAxes, fontsize=14, color='#666', style='italic')
    
    # Add value labels
    for bar, val in zip(bars1, prefill_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add annotation for prefill
    ax1.annotate('Parallel token\nprocessing', xy=(0, 75), xytext=(0.55, 55),
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
    ax2.set_ylabel('Utilization (%)', fontsize=14)
    ax2.text(0.5, 1.05, 'Decode Phase', ha='center', va='bottom',
             transform=ax2.transAxes, fontsize=14, fontweight='bold')
    ax2.text(0.5, 0.98, '(Memory-Bound)', ha='center', va='bottom',
             transform=ax2.transAxes, fontsize=14, color='#666', style='italic')
    
    # Add value labels
    for bar, val in zip(bars2, decode_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add annotation for decode
    ax2.annotate('KV cache\nloading', xy=(1, 80), xytext=(0.5, 60),
                fontsize=14, ha='center', color='#333',
                arrowprops=dict(arrowstyle='->', color='#666', lw=1))
    
    ax2.tick_params(labelsize=11)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    # Add implication text at bottom
    '''fig.text(0.5, -0.02, 
             'Different resource profiles $\\rightarrow$ Separate scaling for optimal utilization',
             ha='center', va='top', fontsize=14, style='italic', color='#444')'''
    
    plt.tight_layout(pad=0.05)
    save_figure(__file__)


if __name__ == '__main__':
    main()
