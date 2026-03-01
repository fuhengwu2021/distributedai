"""
Training Iteration Breakdown Diagram
Shows time spent in each phase of distributed training.
"""
import matplotlib.pyplot as plt
import numpy as np

from math4ai import save_figure

def create_training_breakdown_diagram():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Data for different GPU configurations
    configs = ['1 GPU', '4 GPUs', '8 GPUs', '16 GPUs']
    
    # Time breakdown (ms) for each phase
    # As GPUs increase, communication overhead grows
    data = {
        '1 GPU':   {'Forward': 45, 'Backward': 85, 'Communication': 0,  'Optimizer': 15, 'Data Loading': 10},
        '4 GPUs':  {'Forward': 12, 'Backward': 22, 'Communication': 8,  'Optimizer': 4,  'Data Loading': 8},
        '8 GPUs':  {'Forward': 6,  'Backward': 12, 'Communication': 12, 'Optimizer': 2,  'Data Loading': 6},
        '16 GPUs': {'Forward': 3,  'Backward': 6,  'Communication': 18, 'Optimizer': 1,  'Data Loading': 5},
    }
    
    phases = ['Forward', 'Backward', 'Communication', 'Optimizer', 'Data Loading']
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#607D8B']
    
    # Left plot: Stacked bar chart
    ax1 = axes[0]
    
    x = np.arange(len(configs))
    width = 0.6
    bottom = np.zeros(len(configs))
    
    for phase, color in zip(phases, colors):
        values = [data[config][phase] for config in configs]
        ax1.bar(x, values, width, label=phase, bottom=bottom, color=color, edgecolor='white', linewidth=1)
        bottom += values
    
    ax1.set_xlabel('Configuration', fontsize=12)
    ax1.set_ylabel('Time per Iteration (ms)', fontsize=12)
    ax1.set_title('Training Iteration Time Breakdown', fontsize=14, fontweight='bold')
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
    
    # Calculate percentages
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
    
    ax2.barh(y, compute_pct, height, label='Compute', color='#4CAF50')
    ax2.barh(y, comm_pct, height, left=compute_pct, label='Commun', color='#FF9800')
    ax2.barh(y, other_pct, height, left=np.array(compute_pct) + np.array(comm_pct), 
             label='Other', color='#607D8B')
    
    ax2.set_xlabel('Percentage of Iteration Time (%)', fontsize=12)
    ax2.set_ylabel('Configuration', fontsize=12)
    ax2.set_title('Time Distribution by Category', fontsize=14, fontweight='bold')
    ax2.set_yticks(y)
    ax2.set_yticklabels(configs)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.set_xlim(0, 100)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add percentage labels
    for i, (comp, comm) in enumerate(zip(compute_pct, comm_pct)):
        ax2.text(comp/2, i, f'{comp:.0f}%', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
        if comm > 5:  # Only show if visible
            ax2.text(comp + comm/2, i, f'{comm:.0f}%', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
    
    
    plt.tight_layout()
    save_figure(__file__)

if __name__ == '__main__':
    create_training_breakdown_diagram()
