"""
Network Topology and Communication Patterns Diagram
Shows distributed communication patterns and bottleneck identification.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
import numpy as np

def create_network_topology_diagram():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Left plot: Ring AllReduce topology
    ax1 = axes[0]
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Ring AllReduce Communication', fontsize=14, fontweight='bold', pad=20)
    
    # GPU positions in a ring
    n_gpus = 8
    angles = np.linspace(0, 2*np.pi, n_gpus, endpoint=False) - np.pi/2
    radius = 1.3
    gpu_positions = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
    
    # Draw GPUs
    gpu_colors = plt.cm.Set3(np.linspace(0, 1, n_gpus))
    for i, (pos, color) in enumerate(zip(gpu_positions, gpu_colors)):
        circle = Circle(pos, 0.25, facecolor=color, edgecolor='#37474F', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(pos[0], pos[1], f'GPU{i}', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Draw ring connections with arrows
    for i in range(n_gpus):
        start = gpu_positions[i]
        end = gpu_positions[(i + 1) % n_gpus]
        
        # Calculate arrow position (slightly offset from center)
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        # Direction vector
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        
        # Shorten arrow to not overlap with circles
        shrink = 0.3
        start_adj = (start[0] + shrink * dx/length, start[1] + shrink * dy/length)
        end_adj = (end[0] - shrink * dx/length, end[1] - shrink * dy/length)
        
        ax1.annotate('', xy=end_adj, xytext=start_adj,
                    arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2, 
                                   connectionstyle='arc3,rad=0.1'))
    
    # Add legend/explanation
    ax1.text(0, -1.8, 'Each GPU sends gradients to next GPU in ring\n'
                      'N-1 steps to complete AllReduce', 
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='#E3F2FD', edgecolor='#1565C0', alpha=0.9))
    
    # Right plot: Communication bottleneck visualization
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.axis('off')
    ax2.set_title('Identifying Communication Bottlenecks', fontsize=14, fontweight='bold', pad=10)
    
    # Draw two nodes
    node_colors = ['#E3F2FD', '#E8F5E9']
    node_labels = ['Node 0', 'Node 1']
    
    for i, (color, label) in enumerate(zip(node_colors, node_labels)):
        x_base = 1 + i * 5
        
        # Node box
        node_box = FancyBboxPatch((x_base, 2), 3, 5,
                                   boxstyle="round,pad=0.05,rounding_size=0.2",
                                   facecolor=color, edgecolor='#37474F', linewidth=2)
        ax2.add_patch(node_box)
        ax2.text(x_base + 1.5, 6.7, label, ha='center', va='center', 
                fontsize=12, fontweight='bold')
        
        # GPUs inside node
        for j in range(4):
            gpu_y = 5.5 - j * 1.1
            gpu_box = FancyBboxPatch((x_base + 0.3, gpu_y - 0.35), 2.4, 0.7,
                                      boxstyle="round,pad=0.02,rounding_size=0.1",
                                      facecolor='#BBDEFB' if i == 0 else '#C8E6C9',
                                      edgecolor='#546E7A', linewidth=1)
            ax2.add_patch(gpu_box)
            ax2.text(x_base + 1.5, gpu_y, f'GPU {i*4 + j}', ha='center', va='center', fontsize=9)
    
    # Draw NVLink connections (fast, within node)
    for i in range(2):
        x_base = 1 + i * 5
        for j in range(3):
            y1 = 5.5 - j * 1.1 - 0.35
            y2 = 5.5 - (j+1) * 1.1 + 0.35
            ax2.annotate('', xy=(x_base + 1.5, y2), xytext=(x_base + 1.5, y1),
                        arrowprops=dict(arrowstyle='<->', color='#4CAF50', lw=2))
    
    # NVLink label
    ax2.text(2.5, 2.5, 'NVLink\n(600 GB/s)', ha='center', va='center', fontsize=8,
             color='#2E7D32', fontweight='bold')
    ax2.text(7.5, 2.5, 'NVLink\n(600 GB/s)', ha='center', va='center', fontsize=8,
             color='#2E7D32', fontweight='bold')
    
    # Draw network connection (slow, between nodes) - BOTTLENECK
    ax2.annotate('', xy=(6, 4.5), xytext=(4, 4.5),
                arrowprops=dict(arrowstyle='<->', color='#F44336', lw=3,
                               connectionstyle='arc3,rad=0'))
    
    # Bottleneck indicator
    ax2.text(5, 5.3, '⚠ Network\nBottleneck', ha='center', va='center', fontsize=10,
             color='#C62828', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#FFEBEE', edgecolor='#C62828'))
    ax2.text(5, 3.7, 'InfiniBand\n(200 GB/s)', ha='center', va='center', fontsize=8,
             color='#C62828')
    
    # Bandwidth comparison box
    bw_box = FancyBboxPatch((0.5, 0.3), 9, 1.4,
                             boxstyle="round,pad=0.05,rounding_size=0.1",
                             facecolor='#FFF8E1', edgecolor='#FF8F00', linewidth=2)
    ax2.add_patch(bw_box)
    ax2.text(5, 1.3, 'Bandwidth Hierarchy:', ha='center', va='center', 
             fontsize=11, fontweight='bold', color='#E65100')
    ax2.text(5, 0.7, 'NVLink (600 GB/s) >> InfiniBand (200 GB/s) >> Ethernet (100 Gbps = 12.5 GB/s)',
             ha='center', va='center', fontsize=9, color='#37474F')
    
    plt.tight_layout()
    plt.savefig('network_topology.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('network_topology.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

if __name__ == '__main__':
    create_network_topology_diagram()
    print("Generated: network_topology.png")
