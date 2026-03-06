"""
vLLM vs SGLang Architecture Comparison

Left: vLLM's single global executor with model parallelism
Right: SGLang's independent workers with router-based distribution
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from math4ai import configure_math_fonts, save_figure
configure_math_fonts()


def draw_box(ax, x, y, width, height, label, color, edge_color, fontsize=11, sublabel=None):
    """Draw a rounded box with label."""
    box = patches.FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=2, edgecolor=edge_color, facecolor=color
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + 0.15, label, ha='center', va='center', 
                fontsize=fontsize, fontweight='bold')
        ax.text(x, y - 0.2, sublabel, ha='center', va='center', 
                fontsize=fontsize-2, color='#555')
    else:
        ax.text(x, y, label, ha='center', va='center', 
                fontsize=fontsize, fontweight='bold')


def draw_arrow(ax, start, end, color='#546e7a', style='->', lw=2):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def draw_vllm_side(ax, center_x):
    """Draw vLLM architecture on the left side."""
    # Colors
    scheduler_color = '#e3f2fd'
    executor_color = '#fff3e0'
    worker_color = '#e8f5e9'
    
    # Title
    ax.text(center_x, 5.8, 'vLLM', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='#1565c0')
    ax.text(center_x, 5.4, 'Model Parallelism', ha='center', va='center', 
            fontsize=11, color='#555')
    
    # Scheduler
    draw_box(ax, center_x, 4.5, 2.2, 0.7, 'Scheduler', scheduler_color, '#1976d2')
    
    # Executor (global)
    draw_box(ax, center_x, 3.3, 2.2, 0.7, 'Executor', executor_color, '#f57c00', 
             sublabel='(global)')
    
    # Workers with sync
    worker_y = 1.8
    worker_width = 1.4
    worker_positions = [center_x - 1.2, center_x + 1.2]
    
    for i, wx in enumerate(worker_positions):
        draw_box(ax, wx, worker_y, worker_width, 0.8, f'W{i}', worker_color, '#388e3c',
                 sublabel=f'GPU {i}')
    
    # Sync arrows between workers (bidirectional)
    sync_y = worker_y
    ax.annotate('', xy=(center_x + 0.4, sync_y), xytext=(center_x - 0.4, sync_y),
                arrowprops=dict(arrowstyle='<->', color='#d32f2f', lw=2))
    ax.text(center_x, sync_y + 0.55, 'all-reduce', ha='center', va='bottom', 
            fontsize=10, color='#d32f2f', style='italic')
    
    # Arrows
    draw_arrow(ax, (center_x, 4.1), (center_x, 3.7))
    draw_arrow(ax, (center_x - 0.3, 2.9), (center_x - 1.0, 2.25))
    draw_arrow(ax, (center_x + 0.3, 2.9), (center_x + 1.0, 2.25))
    
    # Annotation
    ax.text(center_x, 0.7, 'Workers must synchronize\non every layer', 
            ha='center', va='center', fontsize=10, color='#666', style='italic')


def draw_sglang_side(ax, center_x):
    """Draw SGLang architecture on the right side."""
    # Colors
    router_color = '#fce4ec'
    worker_color = '#e8f5e9'
    
    # Title
    ax.text(center_x, 5.8, 'SGLang', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='#c62828')
    ax.text(center_x, 5.4, 'Request-Level Routing', ha='center', va='center', 
            fontsize=11, color='#555')
    
    # Router
    draw_box(ax, center_x, 4.2, 2.4, 0.9, 'Router', router_color, '#c62828',
             sublabel='(cache-aware)')
    
    # Independent workers
    worker_y = 1.8
    worker_width = 1.4
    worker_positions = [center_x - 1.5, center_x, center_x + 1.5]
    
    for i, wx in enumerate(worker_positions):
        draw_box(ax, wx, worker_y, worker_width, 0.8, f'W{i}', worker_color, '#388e3c',
                 sublabel='Full Model')
    
    # No sync - independence markers
    '''for i in range(len(worker_positions) - 1):
        mid_x = (worker_positions[i] + worker_positions[i+1]) / 2
        ax.text(mid_x, worker_y, '$\\times$', ha='center', va='center', 
                fontsize=14, color='#388e3c', fontweight='bold')'''
    
    # Arrows from router to workers
    for wx in worker_positions:
        draw_arrow(ax, (center_x + (wx - center_x) * 0.3, 3.7), (wx, 2.25))
    
    # Annotation
    ax.text(center_x, 0.7, 'Workers process requests\nindependently', 
            ha='center', va='center', fontsize=10, color='#666', style='italic')


def main():
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.set_xlim(0.8, 10.6)
    ax.set_ylim(0.3, 6.2)
    ax.axis('off')
    
    # Draw dividing line
    ax.axvline(x=5.5, color='#ccc', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Draw both sides
    draw_vllm_side(ax, center_x=2.75)
    draw_sglang_side(ax, center_x=8.25)
    
    plt.tight_layout(pad=0.1)
    save_figure(__file__)


if __name__ == '__main__':
    main()
