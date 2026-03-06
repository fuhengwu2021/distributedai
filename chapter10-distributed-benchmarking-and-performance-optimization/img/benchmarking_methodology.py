"""
Benchmarking Methodology Flow Diagram
Shows the proper workflow for benchmarking distributed AI systems.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

from math4ai import save_figure

def create_benchmarking_methodology_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Benchmarking Methodology Flow', fontsize=18, fontweight='bold',
            ha='center', va='center')
    
    # Colors
    colors = {
        'setup': '#E3F2FD',
        'warmup': '#FFF3E0',
        'measure': '#E8F5E9',
        'analyze': '#F3E5F5',
        'arrow': '#546E7A'
    }
    
    # Phase boxes - vertical flow
    phases = [
        {'name': 'Setup Phase', 'y': 8.2, 'color': colors['setup'],
         'items': ['Fix random seeds', 'Document hardware config', 'Version control scripts', 'Record system state']},
        {'name': 'Warmup Phase', 'y': 6.2, 'color': colors['warmup'],
         'items': ['Run 10-20 warmup iterations', 'JIT compilation completes', 'Memory allocation stabilizes', 'Caches warm up']},
        {'name': 'Measurement Phase', 'y': 4.2, 'color': colors['measure'],
         'items': ['torch.cuda.synchronize()', 'Multiple iterations (100+)', 'Record individual timings', 'Multiple independent runs']},
        {'name': 'Analysis Phase', 'y': 2.2, 'color': colors['analyze'],
         'items': ['Calculate mean, std, min, max', 'Compute P50, P95, P99', 'Statistical significance tests', 'Generate reports']},
    ]
    
    for i, phase in enumerate(phases):
        # Main box
        box = FancyBboxPatch((1, phase['y'] - 0.8), 12, 1.6,
                             boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor=phase['color'], edgecolor='#37474F', linewidth=2)
        ax.add_patch(box)
        
        # Phase name
        ax.text(2, phase['y'] + 0.4, phase['name'], fontsize=14, fontweight='bold',
                ha='left', va='center', color='#1A237E')
        
        # Items
        for j, item in enumerate(phase['items']):
            x_pos = 2.5 + (j % 2) * 5.5
            y_pos = phase['y'] - 0.1 - (j // 2) * 0.4
            ax.text(x_pos, y_pos, f'• {item}', fontsize=10, ha='left', va='center', color='#37474F')
        
        # Arrow to next phase
        if i < len(phases) - 1:
            ax.annotate('', xy=(7, phase['y'] - 1.0), xytext=(7, phase['y'] - 0.8),
                       arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    
    # Side annotations - Common Pitfalls
    pitfall_box = FancyBboxPatch((0.2, 0.3), 4.5, 1.5,
                                  boxstyle="round,pad=0.05,rounding_size=0.1",
                                  facecolor='#FFEBEE', edgecolor='#C62828', linewidth=2)
    ax.add_patch(pitfall_box)
    ax.text(2.45, 1.5, '⚠ Common Pitfalls', fontsize=11, fontweight='bold',
            ha='center', va='center', color='#C62828')
    ax.text(0.5, 1.1, '• No warmup (cold start bias)', fontsize=9, ha='left', color='#37474F')
    ax.text(0.5, 0.8, '• Single measurement only', fontsize=9, ha='left', color='#37474F')
    ax.text(0.5, 0.5, '• Including data loading time', fontsize=9, ha='left', color='#37474F')
    
    # Best Practices box
    best_box = FancyBboxPatch((5.0, 0.3), 4.5, 1.5,
                               boxstyle="round,pad=0.05,rounding_size=0.1",
                               facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(best_box)
    ax.text(7.25, 1.5, '✓ Best Practices', fontsize=11, fontweight='bold',
            ha='center', va='center', color='#2E7D32')
    ax.text(5.3, 1.1, '• 3-5 independent runs', fontsize=9, ha='left', color='#37474F')
    ax.text(5.3, 0.8, '• Report confidence intervals', fontsize=9, ha='left', color='#37474F')
    ax.text(5.3, 0.5, '• Document everything', fontsize=9, ha='left', color='#37474F')
    
    # Key Metrics box
    metrics_box = FancyBboxPatch((9.8, 0.3), 4.0, 1.5,
                                  boxstyle="round,pad=0.05,rounding_size=0.1",
                                  facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
    ax.add_patch(metrics_box)
    ax.text(11.8, 1.5, 'Key Metrics', fontsize=11, fontweight='bold',
            ha='center', va='center', color='#1565C0')
    ax.text(10.1, 1.1, '• Throughput (samples/s)', fontsize=9, ha='left', color='#37474F')
    ax.text(10.1, 0.8, '• Latency (P50/P95/P99)', fontsize=9, ha='left', color='#37474F')
    ax.text(10.1, 0.5, '• Scaling efficiency (%)', fontsize=9, ha='left', color='#37474F')
    
    plt.tight_layout()
    save_figure(__file__)

if __name__ == '__main__':
    create_benchmarking_methodology_diagram()
