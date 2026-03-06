"""
Pipeline Bubble Diagram

Illustrates the pipeline bubble problem in pipeline parallelism.
GPUs sit idle waiting for data from previous stages, creating bubbles.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from math4ai import save_figure


def draw_pipeline_bubble():
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Colors for batches
    batch_colors = ['#5CB85C', '#4A90D9', '#F5A623']  # Green, Blue, Orange for B1, B2, B3
    
    # Layout parameters
    block_width = 1.2
    block_height = 0.7
    row_gap = 1.2
    time_gap = 0.8  # Gap between stages in time
    
    # GPU labels
    gpus = ['GPU 0', 'GPU 1', 'GPU 2']
    batches = ['B1', 'B2', 'B3']
    
    # Calculate positions for each batch on each GPU
    # B1: GPU0 at t=0, GPU1 at t=1, GPU2 at t=2
    # B2: GPU0 at t=2, GPU1 at t=3, GPU2 at t=4
    # B3: GPU0 at t=4, GPU1 at t=5, GPU2 at t=6
    
    batch_positions = [
        # B1 positions (gpu_idx, time_slot)
        [(0, 0), (1, 1), (2, 2)],
        # B2 positions
        [(0, 2), (1, 3), (2, 4)],
        # B3 positions
        [(0, 4), (1, 5), (2, 6)],
    ]
    
    # Draw GPU labels
    for gpu_idx, gpu_name in enumerate(gpus):
        y = (2 - gpu_idx) * row_gap
        ax.text(-0.5, y + block_height/2, gpu_name, ha='right', va='center', 
                fontsize=12, fontweight='bold')
    
    # Draw batch blocks
    for batch_idx, positions in enumerate(batch_positions):
        color = batch_colors[batch_idx]
        label = batches[batch_idx]
        
        for gpu_idx, time_slot in positions:
            x = time_slot * (block_width + time_gap)
            y = (2 - gpu_idx) * row_gap
            
            rect = patches.FancyBboxPatch((x, y), block_width, block_height,
                                          boxstyle="round,pad=0.02,rounding_size=0.08",
                                          facecolor=color, edgecolor='white', linewidth=2)
            ax.add_patch(rect)
            ax.text(x + block_width/2, y + block_height/2, label,
                    ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    
    # Draw idle annotation for GPU 2
    gpu2_y = 0 * row_gap
    idle_start = 0
    idle_end = 2 * (block_width + time_gap) - time_gap
    
    # Draw idle arrow
    arrow_y = gpu2_y + block_height/2
    ax.annotate('', xy=(idle_end - 0.1, arrow_y), xytext=(idle_start + 0.1, arrow_y),
                arrowprops=dict(arrowstyle='<->', color='#D32F2F', lw=2))
    ax.text((idle_start + idle_end)/2, arrow_y - 0.4, 'idle',
            ha='center', va='top', fontsize=11, color='#D32F2F', style='italic')
    
    # Set axis properties
    ax.set_xlim(-1.5, 7 * (block_width + time_gap))
    ax.set_ylim(-0.8, 3 * row_gap)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout(pad=0.1)
    save_figure(__file__)


if __name__ == '__main__':
    draw_pipeline_bubble()
