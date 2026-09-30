#!/usr/bin/env python3
"""
Latency Percentiles Distribution Diagram
Shows the concept of P50, P95, P99 latency metrics.

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
        "p50_label": "P50 = {p50:.1f}ms",
        "p95_label": "P95 = {p95:.1f}ms",
        "p99_label": "P99 = {p99:.1f}ms",
        "mean_label": "Mean = {mean_lat:.1f}ms",
        "xlabel": "Latency (ms)",
        "ylabel_density": "Density",
        "title_hist": "Latency Distribution with Percentiles",
        "ylabel_pct": "Percentile (%)",
        "title_cdf": "Cumulative Distribution Function (CDF)",
        "p_annot": "P{p}: {val:.1f}ms",
        "tail_latency": "Tail Latency\n(P99+)",
    },
    "zh": {
        "p50_label": "P50 = {p50:.1f}ms",
        "p95_label": "P95 = {p95:.1f}ms",
        "p99_label": "P99 = {p99:.1f}ms",
        "mean_label": "均值 Mean = {mean_lat:.1f}ms",
        "xlabel": "时延 (ms)",
        "ylabel_density": "概率密度 (Density)",
        "title_hist": "时延分布直方图与核心分位数",
        "ylabel_pct": "累积百分比 (%)",
        "title_cdf": "累积分布函数曲线 (CDF)",
        "p_annot": "P{p}: {val:.1f}ms",
        "tail_latency": "长尾时延\n(P99+ Tail)",
    }
}


def draw(text: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Generate realistic latency distribution (log-normal)
    np.random.seed(42)
    n_samples = 10000

    # Log-normal distribution for latency (typical for inference)
    mu, sigma = 2.5, 0.5  # Parameters for log-normal
    latencies = np.random.lognormal(mu, sigma, n_samples)

    # Add some tail latency spikes
    n_spikes = int(n_samples * 0.02)  # 2% spikes
    spike_latencies = np.random.uniform(50, 100, n_spikes)
    latencies = np.concatenate([latencies, spike_latencies])

    # Calculate percentiles
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    mean_lat = np.mean(latencies)

    # Left plot: Histogram with percentile lines
    ax1 = axes[0]

    # Histogram
    n, bins, patches = ax1.hist(latencies, bins=100, density=True, alpha=0.7,
                                 color='#64B5F6', edgecolor='white', linewidth=0.5)

    # Color bars based on percentile regions
    for patch, b in zip(patches, bins[:-1]):
        if b <= p50:
            patch.set_facecolor('#4CAF50')  # Green - below P50
        elif b <= p95:
            patch.set_facecolor('#FFC107')  # Yellow - P50-P95
        elif b <= p99:
            patch.set_facecolor('#FF9800')  # Orange - P95-P99
        else:
            patch.set_facecolor('#F44336')  # Red - above P99

    # Percentile lines
    ax1.axvline(p50, color='#2E7D32', linestyle='-', linewidth=2.5, label=text["p50_label"].format(p50=p50))
    ax1.axvline(p95, color='#F57C00', linestyle='-', linewidth=2.5, label=text["p95_label"].format(p95=p95))
    ax1.axvline(p99, color='#C62828', linestyle='-', linewidth=2.5, label=text["p99_label"].format(p99=p99))
    ax1.axvline(mean_lat, color='#1565C0', linestyle='--', linewidth=2, label=text["mean_label"].format(mean_lat=mean_lat))

    ax1.set_xlabel(text["xlabel"], fontsize=12)
    ax1.set_ylabel(text["ylabel_density"], fontsize=12)
    ax1.set_title(text["title_hist"], fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim(0, 80)
    ax1.grid(True, alpha=0.3)

    # Right plot: CDF with percentile markers
    ax2 = axes[1]

    sorted_latencies = np.sort(latencies)
    cdf = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies) * 100

    ax2.plot(sorted_latencies, cdf, color='#1565C0', linewidth=2.5)

    # Mark percentiles
    percentiles = [50, 95, 99]
    colors = ['#2E7D32', '#F57C00', '#C62828']
    markers = ['o', 's', '^']

    for p, c, m in zip(percentiles, colors, markers):
        val = np.percentile(latencies, p)
        ax2.scatter([val], [p], color=c, s=150, marker=m, zorder=5, edgecolor='white', linewidth=2)
        ax2.hlines(p, 0, val, colors=c, linestyles='--', alpha=0.5)
        ax2.vlines(val, 0, p, colors=c, linestyles='--', alpha=0.5)
        ax2.annotate(text["p_annot"].format(p=p, val=val), (val, p), textcoords="offset points",
                    xytext=(10, 5), fontsize=11, fontweight='bold', color=c)

    ax2.set_xlabel(text["xlabel"], fontsize=12)
    ax2.set_ylabel(text["ylabel_pct"], fontsize=12)
    ax2.set_title(text["title_cdf"], fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 80)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    # Add annotation about tail latency
    ax2.text(60, 30, text["tail_latency"], fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor='#FFEBEE', edgecolor='#C62828', alpha=0.9))
    ax2.annotate('', xy=(75, 99), xytext=(60, 40),
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5))

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    localized_figure(draw, "latency_distribution", LABELS, __file__)
