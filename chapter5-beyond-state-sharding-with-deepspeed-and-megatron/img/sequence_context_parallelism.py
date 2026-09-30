import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import sys

# Ensure shared directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

COLORS = {
    'gpu0': '#4CAF50',      # Green
    'gpu1': '#2196F3',      # Blue
    'gpu2': '#FF9800',      # Orange
    'gpu3': '#9C27B0',      # Purple
    'arrow': '#424242',     # Dark gray
    'bg': '#FAFAFA',        # Light gray
    'border': '#757575',    # Gray
    'kv_transfer': '#F44336',  # Red for KV transfer
}

LABELS = {
    "en": {
        "title_sp": "Sequence Parallelism",
        "full_seq": "Full Sequence (8K tokens)",
        "activations_full": "Activations: (batch, 8K, hidden)",
        "split_label": "split",
        "split_desc": "Split along sequence dimension",
        "gpu0_tokens": "GPU 0: tokens 0-4K",
        "gpu1_tokens": "GPU 1: tokens 4K-8K",
        "activations_chunk": "(batch, 4K, hidden)",
        "title_cp": "Context Parallelism (Ring Attention)",
        "gpu_labels": ['GPU 0\nQ₀, K₀, V₀', 'GPU 1\nQ₁, K₁, V₁', 'GPU 2\nQ₂, K₂, V₂', 'GPU 3\nQ₃, K₃, V₃'],
        "pass_kv_right": "pass K,V →",
        "pass_kv_left": "← pass K,V",
        "ring_topology": "Ring Topology",
        "ring_desc": "K,V chunks rotate\naround the ring",
    },
    "zh": {
        "title_sp": "序列并行 (Sequence Parallelism)",
        "full_seq": "完整序列 (8K Tokens)",
        "activations_full": "激活值张量：(batch, 8K, hidden)",
        "split_label": "切分",
        "split_desc": "沿序列（Sequence）维度切分",
        "gpu0_tokens": "GPU 0: Tokens 0-4K",
        "gpu1_tokens": "GPU 1: Tokens 4K-8K",
        "activations_chunk": "(batch, 4K, hidden)",
        "title_cp": "上下文并行 (Context Parallelism / Ring Attention)",
        "gpu_labels": ['GPU 0\n$Q_0, K_0, V_0$', 'GPU 1\n$Q_1, K_1, V_1$', 'GPU 2\n$Q_2, K_2, V_2$', 'GPU 3\n$Q_3, K_3, V_3$'],
        "pass_kv_right": "传递 K, V →",
        "pass_kv_left": "← 传递 K, V",
        "ring_topology": "环形拓扑 (Ring Topology)",
        "ring_desc": "K, V 分块沿环状拓扑\n循环轮转与计算",
    },
}

def draw_sequence_parallelism(ax, text):
    """Draw sequence parallelism: activations split along sequence dimension."""
    ax.set_xlim(0.8, 11.2)
    ax.set_ylim(2, 6)
    ax.axis('off')
    ax.set_title(text["title_sp"], fontsize=14, fontweight='bold', pad=15)
    
    # Full sequence (top)
    ax.text(6, 5.7, text["full_seq"], ha='center', fontsize=13, fontweight='bold')
    
    # Draw full sequence bar
    full_seq = patches.FancyBboxPatch((1, 4.8), 10, 0.6, boxstyle="round,pad=0.02",
                                       facecolor='#E0E0E0', edgecolor=COLORS['border'], linewidth=1.5)
    ax.add_patch(full_seq)
    ax.text(6, 5.1, text["activations_full"], ha='center', fontsize=13, style='italic')
    
    # Arrow down
    ax.annotate("", xy=(6, 4.3), xytext=(6, 4.7),
                arrowprops=dict(arrowstyle="->", color=COLORS['arrow'], lw=2))
    ax.text(6.2, 4.5, text["split_label"], ha='left', fontsize=13)
    
    # Split across 2 GPUs
    ax.text(6, 3.9, text["split_desc"], ha='center', fontsize=13)
    
    # GPU 0 chunk
    gpu0_box = patches.FancyBboxPatch((1.5, 2.35), 4, 1, boxstyle="round,pad=0.02",
                                       facecolor=COLORS['gpu0'], edgecolor='white', linewidth=2, alpha=0.8)
    ax.add_patch(gpu0_box)
    ax.text(3.5, 3.0, text["gpu0_tokens"], ha='center', va='center', fontsize=13, fontweight='bold', color='white')
    ax.text(3.5, 2.7, text["activations_chunk"], ha='center', va='center', fontsize=12, color='white')
    
    # GPU 1 chunk
    gpu1_box = patches.FancyBboxPatch((6.5, 2.35), 4, 1, boxstyle="round,pad=0.02",
                                       facecolor=COLORS['gpu1'], edgecolor='white', linewidth=2, alpha=0.8)
    ax.add_patch(gpu1_box)
    ax.text(8.5, 3.0, text["gpu1_tokens"], ha='center', va='center', fontsize=13, fontweight='bold', color='white')
    ax.text(8.5, 2.7, text["activations_chunk"], ha='center', va='center', fontsize=12, color='white')


def draw_context_parallelism(ax, text):
    """Draw context parallelism with ring attention."""
    ax.set_xlim(1.5, 12.5)
    ax.set_ylim(1, 7)
    ax.axis('off')
    ax.set_title(text["title_cp"], fontsize=14, fontweight='bold', pad=15)
    
    # Draw 4 GPUs in a ring layout
    gpu_positions = [(3, 5.5), (11, 5.5), (11, 2), (3, 2)]  # top-left, top-right, bottom-right, bottom-left
    gpu_labels = text["gpu_labels"]
    gpu_colors = [COLORS['gpu0'], COLORS['gpu1'], COLORS['gpu2'], COLORS['gpu3']]
    
    for i, (pos, label, color) in enumerate(zip(gpu_positions, gpu_labels, gpu_colors)):
        box = patches.FancyBboxPatch((pos[0]-1.2, pos[1]-0.6), 2.4, 1.2, boxstyle="round,pad=0.02",
                                      facecolor=color, edgecolor='white', linewidth=2, alpha=0.85)
        ax.add_patch(box)
        ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=13, fontweight='bold', color='white')
    
    # Draw ring arrows (KV passing)
    # Top: GPU 0 -> GPU 1
    ax.annotate("", xy=(9.5, 5.8), xytext=(4.5, 5.8),
                arrowprops=dict(arrowstyle="->", color=COLORS['kv_transfer'], lw=2.5,
                               connectionstyle="arc3,rad=-0.1"))
    ax.text(7, 6.3, text["pass_kv_right"], ha='center', fontsize=12, color=COLORS['kv_transfer'])
    
    # Right: GPU 1 -> GPU 2
    ax.annotate("", xy=(11.3, 3.3), xytext=(11.3, 4.7),
                arrowprops=dict(arrowstyle="->", color=COLORS['kv_transfer'], lw=2.5))
    ax.text(12, 4, "↓", ha='center', fontsize=13, color=COLORS['kv_transfer'])
    
    # Bottom: GPU 2 -> GPU 3
    ax.annotate("", xy=(4.5, 1.7), xytext=(9.5, 1.7),
                arrowprops=dict(arrowstyle="->", color=COLORS['kv_transfer'], lw=2.5,
                               connectionstyle="arc3,rad=-0.1"))
    ax.text(7, 1.2, text["pass_kv_left"], ha='center', fontsize=12, color=COLORS['kv_transfer'])
    
    # Left: GPU 3 -> GPU 0
    ax.annotate("", xy=(2.7, 4.7), xytext=(2.7, 3.3),
                arrowprops=dict(arrowstyle="->", color=COLORS['kv_transfer'], lw=2.5))
    ax.text(2, 4, "↑", ha='center', fontsize=13, color=COLORS['kv_transfer'])
    
    # Center explanation
    ax.text(7, 3.95, text["ring_topology"], ha='center', fontsize=13, fontweight='bold')
    ax.text(7, 3.3, text["ring_desc"], ha='center', fontsize=13, style='italic')


def draw(text: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    draw_sequence_parallelism(ax1, text)
    draw_context_parallelism(ax2, text)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    localized_figure(draw, "sequence_context_parallelism", LABELS, __file__, dpi=150, pad_inches=0.10)
