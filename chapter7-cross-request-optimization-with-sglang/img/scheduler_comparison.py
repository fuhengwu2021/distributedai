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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "serial_title": "Serial Scheduler",
        "zero_title": "Zero-Overhead",
        "cpu": "CPU",
        "gpu": "GPU",
        "sched": "Sched",
        "compute": "Compute",
        "idle": "idle",
        "wait_annot": "GPU waits for CPU",
        "pre": "Pre",
        "launch": "L",
        "post": "Post",
        "overlap_annot": "CPU and GPU overlap",
        "time": "Time",
    },
    "zh": {
        "serial_title": "传统串行调度",
        "zero_title": "零开销重叠调度",
        "cpu": "CPU",
        "gpu": "GPU",
        "sched": "调度",
        "compute": "计算",
        "idle": "空闲",
        "wait_annot": "GPU 等待 CPU 调度",
        "pre": "预调度",
        "launch": "发射",
        "post": "后处理",
        "overlap_annot": "CPU 与 GPU 重叠并行",
        "time": "时间 (Time)",
    }
}


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


def draw_idle_block(ax, x, y, width, height, label='idle'):
    """Draw an idle/waiting block with hatching."""
    rect = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor='#f5f5f5', edgecolor='#999', linewidth=1.5,
        linestyle='--'
    )
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label, ha='center', va='center',
            fontsize=10, color='#999', style='italic')


def draw_serial_scheduler(ax, base_y, text):
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
    ax.text(-0.3, base_y + 0.9, text['serial_title'], ha='right', va='center',
            fontsize=12, fontweight='bold')
    
    # Row labels
    ax.text(-0.3, base_y + 0.6, text['cpu'], ha='right', va='center', fontsize=11)
    ax.text(-0.3, base_y, text['gpu'], ha='right', va='center', fontsize=11)
    
    # Timeline for 3 batches
    x = 0
    for batch in range(3):
        # CPU schedules (batch N)
        draw_block(ax, x, base_y + 0.6, cpu_width, block_height, text['sched'], cpu_color)
        # GPU idle during CPU work
        draw_idle_block(ax, x, base_y, cpu_width, block_height, text['idle'])
        x += cpu_width + gap
        
        # GPU computes (batch N)
        draw_block(ax, x, base_y, gpu_width, block_height, text['compute'], gpu_color)
        # CPU idle during GPU work
        draw_idle_block(ax, x, base_y + 0.6, gpu_width, block_height, text['idle'])
        x += gpu_width + gap
    
    # Draw idle annotation
    total_width = x - gap
    ax.annotate('', xy=(total_width * 0.3, base_y - 0.3), 
                xytext=(total_width * 0.1, base_y - 0.3),
                arrowprops=dict(arrowstyle='<->', color='#d32f2f', lw=1.5))
    ax.text(total_width * 0.2, base_y - 0.5, text['wait_annot'], 
            ha='center', va='top', fontsize=10, color='#d32f2f', style='italic')
    
    return x


def draw_zero_overhead_scheduler(ax, base_y, total_width, text):
    """Draw zero-overhead scheduler timeline with overlap."""
    # Colors
    cpu_color = '#4A90D9'  # Blue for CPU
    gpu_color = '#5CB85C'  # Green for GPU
    
    # Block dimensions
    block_height = 0.5
    unit_width = 0.9
    gap = 0.05
    
    # Label
    ax.text(-0.3, base_y + 0.9, text['zero_title'], ha='right', va='center',
            fontsize=12, fontweight='bold')
    
    # Row labels
    ax.text(-0.3, base_y + 0.6, text['cpu'], ha='right', va='center', fontsize=11)
    ax.text(-0.3, base_y, text['gpu'], ha='right', va='center', fontsize=11)
    
    # Overlapped timeline
    x = 0
    batches = 4
    
    pre_fs = 9 if text['pre'] == 'Pre' else 8
    launch_fs = 9 if text['launch'] == 'L' else 8
    post_fs = 9 if text['post'] == 'Post' else 8
    
    for batch in range(batches):
        # CPU pre-schedule for batch
        draw_block(ax, x, base_y + 0.6, unit_width * 0.8, block_height, text['pre'], cpu_color, fontsize=pre_fs)
        
        if batch > 0:
            # GPU computing previous batch (overlapped)
            draw_block(ax, x - unit_width * 0.3, base_y, unit_width * 1.5, block_height, 
                      text['compute'], gpu_color, fontsize=10)
        
        x += unit_width * 0.8 + gap
        
        # CPU launch
        draw_block(ax, x, base_y + 0.6, unit_width * 0.5, block_height, text['launch'], cpu_color, fontsize=launch_fs)
        x += unit_width * 0.5 + gap
        
        if batch < batches - 1:
            # CPU post-process
            draw_block(ax, x, base_y + 0.6, unit_width * 0.6, block_height, text['post'], cpu_color, fontsize=post_fs)
            x += unit_width * 0.6 + gap
    
    # Final GPU compute
    draw_block(ax, x - unit_width * 1.2, base_y, unit_width * 1.5, block_height, 
              text['compute'], gpu_color, fontsize=10)
    
    # Overlap annotation
    ax.annotate('', xy=(unit_width * 2.5, base_y - 0.13), 
                xytext=(unit_width * 1.0, base_y - 0.13),
                arrowprops=dict(arrowstyle='<->', color='#388e3c', lw=1.5))
    ax.text(unit_width * 1.75, base_y - 0.3, text['overlap_annot'], 
            ha='center', va='top', fontsize=10, color='#388e3c', style='italic')


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Draw both schedulers
    total_width = draw_serial_scheduler(ax, base_y=3.0, text=text)
    draw_zero_overhead_scheduler(ax, base_y=0.8, total_width=total_width, text=text)
    
    # Time axis
    ax.annotate('', xy=(total_width - 0.5, -0.2), xytext=(0, -0.2),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax.text(total_width / 2, 0.05, text['time'], ha='center', va='top', fontsize=11)
    
    # Legend
    legend_y = 4.5
    legend_x = 0
    draw_block(ax, legend_x, legend_y, 0.8, 0.4, text['cpu'], '#4A90D9', fontsize=10)
    draw_block(ax, legend_x + 1.2, legend_y, 0.8, 0.4, text['gpu'], '#5CB85C', fontsize=10)
    draw_idle_block(ax, legend_x + 2.4, legend_y, 0.8, 0.4, text['idle'])
    
    # Set limits
    ax.set_xlim(-1.5, total_width + 0.5)
    ax.set_ylim(-0.2, 5.2)
    ax.axis('off')
    
    plt.tight_layout(pad=0.1)
    return fig


if __name__ == '__main__':
    localized_figure(draw, "scheduler_comparison", LABELS, __file__, pad_inches=0.02, use_math_fonts=True)
