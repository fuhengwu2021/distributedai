"""
Scaling Efficiency Visualization
Shows ideal linear scaling vs actual scaling with efficiency percentages.
"""
import matplotlib.pyplot as plt
import numpy as np

def create_scaling_efficiency_diagram():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Data
    gpus = np.array([1, 2, 4, 8, 16, 32])
    
    # Throughput scaling
    baseline_throughput = 100  # samples/sec with 1 GPU
    ideal_throughput = baseline_throughput * gpus
    
    # Realistic scaling with diminishing returns
    # Efficiency decreases as GPUs increase due to communication overhead
    efficiencies = np.array([100, 95, 88, 81, 72, 62])  # %
    actual_throughput = ideal_throughput * efficiencies / 100
    
    # Left plot: Throughput vs GPUs
    ax1 = axes[0]
    ax1.plot(gpus, ideal_throughput, 'b--', linewidth=2, marker='o', 
             markersize=8, label='Ideal (Linear)', alpha=0.7)
    ax1.plot(gpus, actual_throughput, 'g-', linewidth=2.5, marker='s',
             markersize=8, label='Actual', color='#2E7D32')
    
    # Fill the gap
    ax1.fill_between(gpus, actual_throughput, ideal_throughput, 
                     alpha=0.2, color='red', label='Communication Overhead')
    
    ax1.set_xlabel('Number of GPUs', fontsize=12)
    ax1.set_ylabel('Throughput (samples/sec)', fontsize=12)
    ax1.set_title('Throughput Scaling', fontsize=14, fontweight='bold')
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
    
    ax2.set_xlabel('Number of GPUs', fontsize=12)
    ax2.set_ylabel('Scaling Efficiency (%)', fontsize=12)
    ax2.set_title('Scaling Efficiency by GPU Count', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(gpus)))
    ax2.set_xticklabels(gpus)
    ax2.set_ylim(0, 110)
    ax2.axhline(y=90, color='#4CAF50', linestyle='--', alpha=0.5, label='Excellent (>90%)')
    ax2.axhline(y=70, color='#FFC107', linestyle='--', alpha=0.5, label='Good (>70%)')
    ax2.axhline(y=50, color='#FF9800', linestyle='--', alpha=0.5, label='Moderate (>50%)')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add Amdahl's Law annotation
    ax2.text(4.5, 105, "Amdahl's Law: Speedup = 1 / (S + P/N)", 
             fontsize=10, style='italic', ha='center',
             bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('scaling_efficiency.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('scaling_efficiency.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

if __name__ == '__main__':
    create_scaling_efficiency_diagram()
    print("Generated: scaling_efficiency.png")
