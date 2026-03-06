"""
Serial vs Zero-Overhead Scheduler Comparison

Shows CPU/GPU overlap in SGLang's zero-overhead scheduler vs traditional serial execution.
Timeline diagram showing how zero-overhead scheduler eliminates GPU idle time.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

from math4ai import configure_math_fonts, save_figure
configure_math_fonts()


def draw_block(ax, x, y, width, height, label, color, edge_color='white', fontsize=11):
    """Draw a rounded block with label."""
    rect = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=color, edgecolor=edge_color, linewidth=1.5
    )
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label, ha='center', va='center',
            fontsize=fontsize, color='white', fontweight='bold')


def draw_idle_block(ax, x, y, width, height):
    """Draw an idle/waiting block with hatching."""
    rect = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor='#f5f5f5', edgecolor='#999', linewidth=1.5,
        linestyle='--'
    )
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, 'idle', ha='center', va='center',
            fontsize=10, color='#999', style='italic')


def draw_serial_scheduler(ax, base_y):
    """Draw serial scheduler timeline."""
    # Colors
    cpu_color = '#4A90D9'  # Blue for CPU
    gpu_color = '#5CB85C'  # Green for GPU
    
    # Block dimensions
    block_height = 0.5
    cpu_width = 1.0
    gpu_width = 1.5
    gap = 0.1
    
    # Label
    ax.text(-0.3, base_y + 0.9, 'Serial Scheduler', ha='right', va='center',
            fontsize=12, fontweight='bold')
    
    # Row labels
    ax.text(-0.3, base_y + 0.6, 'CPU', ha='right', va='center', fontsize=11)
    ax.text(-0.3, base_y, 'GPU', ha='right', va='center', fontsize=11)
    
    # Timeline for 3 batches
    x = 0
    for batch in range(3):
        # CPU schedules (batch N)
        draw_block(ax, x, base_y + 0.6, cpu_width, block_height, f'Sched', cpu_color)
        # GPU idle during CPU work
        draw_idle_block(ax, x, base_y, cpu_width, block_height)
        x += cpu_width + gap
        
        # GPU computes (batch N)
        draw_block(ax, x, base_y, gpu_width, block_height, f'Compute', gpu_color)
        # CPU idle during GPU work
        draw_idle_block(ax, x, base_y + 0.6, gpu_width, block_height)
        x += gpu_width + gap
    
    # Draw idle annotation
    total_width = x - gap
    ax.annotate('', xy=(total_width * 0.3, base_y - 0.3), 
                xytext=(total_width * 0.1, base_y - 0.3),
                arrowprops=dict(arrowstyle='<->', color='#d32f2f', lw=1.5))
    ax.text(total_width * 0.2, base_y - 0.5, 'GPU waits for CPU', 
            ha='center', va='top', fontsize=10, color='#d32f2f', style='italic')
    
    return x


def draw_zero_overhead_scheduler(ax, base_y, total_width):
    """Draw zero-overhead scheduler timeline with overlap."""
    # Colors
    cpu_color = '#4A90D9'  # Blue for CPU
    gpu_color = '#5CB85C'  # Green for GPU
    
    # Block dimensions
    block_height = 0.5
    unit_width = 0.9
    gap = 0.05
    
    # Label
    ax.text(-0.3, base_y + 0.9, 'Zero-Overhead', ha='right', va='center',
            fontsize=12, fontweight='bold')
    
    # Row labels
    ax.text(-0.3, base_y + 0.6, 'CPU', ha='right', va='center', fontsize=11)
    ax.text(-0.3, base_y, 'GPU', ha='right', va='center', fontsize=11)
    
    # Overlapped timeline
    # CPU: Pre-sched N | Launch N | Post N-1 | Pre-sched N+1 | ...
    # GPU:             | Compute N            | Compute N+1   | ...
    
    x = 0
    batches = 4
    
    for batch in range(batches):
        # CPU pre-schedule for batch
        draw_block(ax, x, base_y + 0.6, unit_width * 0.8, block_height, 'Pre', cpu_color, fontsize=9)
        
        if batch > 0:
            # GPU computing previous batch (overlapped)
            draw_block(ax, x - unit_width * 0.3, base_y, unit_width * 1.5, block_height, 
                      'Compute', gpu_color, fontsize=10)
        
        x += unit_width * 0.8 + gap
        
        # CPU launch
        draw_block(ax, x, base_y + 0.6, unit_width * 0.5, block_height, 'L', cpu_color, fontsize=9)
        x += unit_width * 0.5 + gap
        
        if batch < batches - 1:
            # CPU post-process
            draw_block(ax, x, base_y + 0.6, unit_width * 0.6, block_height, 'Post', cpu_color, fontsize=9)
            x += unit_width * 0.6 + gap
    
    # Final GPU compute
    draw_block(ax, x - unit_width * 1.2, base_y, unit_width * 1.5, block_height, 
              'Compute', gpu_color, fontsize=10)
    
    # Overlap annotation
    ax.annotate('', xy=(unit_width * 2.5, base_y - 0.13), 
                xytext=(unit_width * 1.0, base_y - 0.13),
                arrowprops=dict(arrowstyle='<->', color='#388e3c', lw=1.5))
    ax.text(unit_width * 1.75, base_y - 0.3, 'CPU and GPU overlap', 
            ha='center', va='top', fontsize=10, color='#388e3c', style='italic')


def main():
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Draw both schedulers
    total_width = draw_serial_scheduler(ax, base_y=3.0)
    draw_zero_overhead_scheduler(ax, base_y=0.8, total_width=total_width)
    
    # Time axis
    ax.annotate('', xy=(total_width - 0.5, -0.2), xytext=(0, -0.2),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax.text(total_width / 2, 0.05, 'Time', ha='center', va='top', fontsize=11)
    
    # Legend
    legend_y = 4.5
    legend_x = 0
    draw_block(ax, legend_x, legend_y, 0.8, 0.4, 'CPU', '#4A90D9', fontsize=10)
    draw_block(ax, legend_x + 1.2, legend_y, 0.8, 0.4, 'GPU', '#5CB85C', fontsize=10)
    draw_idle_block(ax, legend_x + 2.4, legend_y, 0.8, 0.4)
    
    # Set limits
    ax.set_xlim(-1.5, total_width + 0.5)
    ax.set_ylim(-0.2, 5.2)
    ax.axis('off')
    
    plt.tight_layout(pad=0.1)
    save_figure(__file__)


if __name__ == '__main__':
    main()
