"""
Latency Percentiles Distribution Diagram
Shows the concept of P50, P95, P99 latency metrics.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from math4ai import save_figure

def create_latency_distribution_diagram():
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
    for i, (patch, b) in enumerate(zip(patches, bins[:-1])):
        if b <= p50:
            patch.set_facecolor('#4CAF50')  # Green - below P50
        elif b <= p95:
            patch.set_facecolor('#FFC107')  # Yellow - P50-P95
        elif b <= p99:
            patch.set_facecolor('#FF9800')  # Orange - P95-P99
        else:
            patch.set_facecolor('#F44336')  # Red - above P99
    
    # Percentile lines
    ax1.axvline(p50, color='#2E7D32', linestyle='-', linewidth=2.5, label=f'P50 = {p50:.1f}ms')
    ax1.axvline(p95, color='#F57C00', linestyle='-', linewidth=2.5, label=f'P95 = {p95:.1f}ms')
    ax1.axvline(p99, color='#C62828', linestyle='-', linewidth=2.5, label=f'P99 = {p99:.1f}ms')
    ax1.axvline(mean_lat, color='#1565C0', linestyle='--', linewidth=2, label=f'Mean = {mean_lat:.1f}ms')
    
    ax1.set_xlabel('Latency (ms)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Latency Distribution with Percentiles', fontsize=14, fontweight='bold')
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
        ax2.annotate(f'P{p}: {val:.1f}ms', (val, p), textcoords="offset points",
                    xytext=(10, 5), fontsize=11, fontweight='bold', color=c)
    
    ax2.set_xlabel('Latency (ms)', fontsize=12)
    ax2.set_ylabel('Percentile (%)', fontsize=12)
    ax2.set_title('Cumulative Distribution Function (CDF)', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 80)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # Add annotation about tail latency
    ax2.text(60, 30, 'Tail Latency\n(P99+)', fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor='#FFEBEE', edgecolor='#C62828', alpha=0.9))
    ax2.annotate('', xy=(75, 99), xytext=(60, 40),
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5))
    
    plt.tight_layout()
    save_figure(__file__)

if __name__ == '__main__':
    create_latency_distribution_diagram()
