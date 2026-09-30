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
from figstyle import localized_figure

LABELS = {
    "en": {
        "vllm_title": "vLLM",
        "vllm_sub": "Model Parallelism",
        "vllm_sched": "Scheduler",
        "vllm_exec": "Executor",
        "vllm_exec_sub": "(global)",
        "vllm_sync": "all-reduce",
        "vllm_note": "Workers must synchronize\non every layer",
        "sglang_title": "SGLang",
        "sglang_sub": "Request-Level Routing",
        "sglang_router": "Router",
        "sglang_router_sub": "(cache-aware)",
        "sglang_worker_sub": "Full Model",
        "sglang_note": "Workers process requests\nindependently",
        "gpu_sub": "GPU {i}",
        "worker_lbl": "W{i}",
    },
    "zh": {
        "vllm_title": "vLLM",
        "vllm_sub": "模型并行",
        "vllm_sched": "调度器 (Scheduler)",
        "vllm_exec": "执行器 (Executor)",
        "vllm_exec_sub": "（全局）",
        "vllm_sync": "All-Reduce",
        "vllm_note": "各 Worker 每层均需\n执行同步通信",
        "sglang_title": "SGLang",
        "sglang_sub": "请求级智能路由",
        "sglang_router": "Router 网关",
        "sglang_router_sub": "（Cache 感知）",
        "sglang_worker_sub": "完整模型",
        "sglang_note": "各 Worker 独立运行\n处理各自请求",
        "gpu_sub": "GPU {i}",
        "worker_lbl": "W{i}",
    }
}


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


def draw_vllm_side(ax, center_x, text):
    """Draw vLLM architecture on the left side."""
    # Colors
    scheduler_color = '#e3f2fd'
    executor_color = '#fff3e0'
    worker_color = '#e8f5e9'
    
    # Title
    ax.text(center_x, 5.8, text['vllm_title'], ha='center', va='center', 
            fontsize=14, fontweight='bold', color='#1565c0')
    ax.text(center_x, 5.4, text['vllm_sub'], ha='center', va='center', 
            fontsize=11, color='#555')
    
    # Scheduler
    draw_box(ax, center_x, 4.5, 2.2, 0.7, text['vllm_sched'], scheduler_color, '#1976d2', fontsize=10 if len(text['vllm_sched']) > 10 else 11)
    
    # Executor (global)
    draw_box(ax, center_x, 3.3, 2.2, 0.7, text['vllm_exec'], executor_color, '#f57c00', 
             sublabel=text['vllm_exec_sub'], fontsize=10 if len(text['vllm_exec']) > 10 else 11)
    
    # Workers with sync
    worker_y = 1.8
    worker_width = 1.4
    worker_positions = [center_x - 1.2, center_x + 1.2]
    
    for i, wx in enumerate(worker_positions):
        draw_box(ax, wx, worker_y, worker_width, 0.8, text['worker_lbl'].format(i=i), worker_color, '#388e3c',
                 sublabel=text['gpu_sub'].format(i=i))
    
    # Sync arrows between workers (bidirectional)
    sync_y = worker_y
    ax.annotate('', xy=(center_x + 0.4, sync_y), xytext=(center_x - 0.4, sync_y),
                arrowprops=dict(arrowstyle='<->', color='#d32f2f', lw=2))
    ax.text(center_x, sync_y + 0.55, text['vllm_sync'], ha='center', va='bottom', 
            fontsize=10, color='#d32f2f', style='italic')
    
    # Arrows
    draw_arrow(ax, (center_x, 4.1), (center_x, 3.7))
    draw_arrow(ax, (center_x - 0.3, 2.9), (center_x - 1.0, 2.25))
    draw_arrow(ax, (center_x + 0.3, 2.9), (center_x + 1.0, 2.25))
    
    # Annotation
    ax.text(center_x, 0.7, text['vllm_note'], 
            ha='center', va='center', fontsize=10, color='#666', style='italic')


def draw_sglang_side(ax, center_x, text):
    """Draw SGLang architecture on the right side."""
    # Colors
    router_color = '#fce4ec'
    worker_color = '#e8f5e9'
    
    # Title
    ax.text(center_x, 5.8, text['sglang_title'], ha='center', va='center', 
            fontsize=14, fontweight='bold', color='#c62828')
    ax.text(center_x, 5.4, text['sglang_sub'], ha='center', va='center', 
            fontsize=11, color='#555')
    
    # Router
    draw_box(ax, center_x, 4.2, 2.4, 0.9, text['sglang_router'], router_color, '#c62828',
             sublabel=text['sglang_router_sub'])
    
    # Independent workers
    worker_y = 1.8
    worker_width = 1.4
    worker_positions = [center_x - 1.5, center_x, center_x + 1.5]
    
    for i, wx in enumerate(worker_positions):
        draw_box(ax, wx, worker_y, worker_width, 0.8, text['worker_lbl'].format(i=i), worker_color, '#388e3c',
                 sublabel=text['sglang_worker_sub'])
    
    # Arrows from router to workers
    for wx in worker_positions:
        draw_arrow(ax, (center_x + (wx - center_x) * 0.3, 3.7), (wx, 2.25))
    
    # Annotation
    ax.text(center_x, 0.7, text['sglang_note'], 
            ha='center', va='center', fontsize=10, color='#666', style='italic')


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.set_xlim(0.8, 10.6)
    ax.set_ylim(0.3, 6.2)
    ax.axis('off')
    
    # Draw dividing line
    ax.axvline(x=5.5, color='#ccc', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Draw both sides
    draw_vllm_side(ax, center_x=2.75, text=text)
    draw_sglang_side(ax, center_x=8.25, text=text)
    
    plt.tight_layout(pad=0.1)
    return fig


if __name__ == '__main__':
    localized_figure(draw, "vllm_vs_sglang", LABELS, __file__, pad_inches=0.02, use_math_fonts=True)
