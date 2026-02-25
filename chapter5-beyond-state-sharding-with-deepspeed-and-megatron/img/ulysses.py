"""
DeepSpeed-Ulysses visualization: all-to-all transpose between
sequence-parallel and head-parallel layouts.

Shows the 2D transpose:
  (S_local, H_all) -> (S_all, H_local)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

COLORS = {
    'gpu0': '#4CAF50',
    'gpu1': '#2196F3',
    'gpu2': '#FF9800',
    'gpu3': '#9C27B0',
    'arrow': '#424242',
    'alltoall': '#E91E63',
    'tensor': '#ECEFF1',
}


def draw_ulysses(ax):
    ax.set_xlim(1, 12)
    ax.set_ylim(0.2, 5)
    ax.axis('off')
    
    gpu_colors = [COLORS['gpu0'], COLORS['gpu1'], COLORS['gpu2'], COLORS['gpu3']]
    
    # Offsets to move left/right sides toward center
    left_offset = 1.0
    right_offset = -1.0
    
    # Left side: Sequence-parallel layout
    ax.text(2.5 + left_offset, 5., "Sequence-Parallel", ha='center', fontsize=12, fontweight='bold')
    
    # Draw 4 GPUs, each with a tensor block showing (local_seq × all_heads)
    for i in range(4):
        y = 4.2 - i * 1.02
        # GPU label
        gpu_box = patches.FancyBboxPatch((0.3 + left_offset, y - 0.35), 0.9, 0.7,
                                          boxstyle="round,pad=0.02",
                                          facecolor=gpu_colors[i], edgecolor='white',
                                          linewidth=1.5, alpha=0.9)
        ax.add_patch(gpu_box)
        ax.text(0.75 + left_offset, y, f"GPU{i}", ha='center', va='center', fontsize=12,
                fontweight='bold', color='white')
        
        # Tensor block: wide (all heads) but short height (local seq)
        tensor_box = patches.FancyBboxPatch((1.4 + left_offset, y - 0.3), 2.2, 0.6,
                                             boxstyle="round,pad=0.01",
                                             facecolor=gpu_colors[i], edgecolor='gray',
                                             linewidth=1, alpha=0.3)
        ax.add_patch(tensor_box)
        ax.text(2.5 + left_offset, y, f"S{i} × H_all", ha='center', va='center', fontsize=12)
    
    # Shape annotation for left side
    ax.text(2.5 + left_offset, 0.3, "(local_seq, all_heads)", ha='center', fontsize=12,
            style='italic', color='#616161')
    
    # All-to-all in the middle
    ax.annotate("", xy=(7.8, 2.6), xytext=(5.2, 2.6),
                arrowprops=dict(arrowstyle="->", color=COLORS['alltoall'], lw=3))
    ax.text(6.5, 3.2, "all-to-all", ha='center', fontsize=12, fontweight='bold',
            color=COLORS['alltoall'])
    ax.text(6.5, 2.0, "2D transpose", ha='center', fontsize=12, style='italic',
            color=COLORS['alltoall'])
    
    # Right side: Head-parallel layout
    ax.text(11.5 + right_offset, 5., "Head-Parallel", ha='center', fontsize=12, fontweight='bold')
    
    # Draw 4 GPUs, each with a tensor block showing (full_seq × local_heads)
    for i in range(4):
        y = 4.2 - i * 1.02
        # GPU label
        gpu_box = patches.FancyBboxPatch((9.3 + right_offset, y - 0.35), 0.9, 0.7,
                                          boxstyle="round,pad=0.02",
                                          facecolor=gpu_colors[i], edgecolor='white',
                                          linewidth=1.5, alpha=0.9)
        ax.add_patch(gpu_box)
        ax.text(9.75 + right_offset, y, f"GPU{i}", ha='center', va='center', fontsize=12,
                fontweight='bold', color='white')
        
        # Tensor block: narrow (local heads) but tall height (full seq)
        tensor_box = patches.FancyBboxPatch((10.4 + right_offset, y - 0.3), 2.2, 0.6,
                                             boxstyle="round,pad=0.01",
                                             facecolor=gpu_colors[i], edgecolor='gray',
                                             linewidth=1, alpha=0.3)
        ax.add_patch(tensor_box)
        ax.text(11.5 + right_offset, y, f"S_all × H{i}", ha='center', va='center', fontsize=12)
    
    # Shape annotation for right side
    ax.text(11.5 + right_offset, 0.3, "(full_seq, local_heads)", ha='center', fontsize=12,
            style='italic', color='#616161')


def main():
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    
    draw_ulysses(ax)
    
    plt.tight_layout()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_path = os.path.join(script_dir, f"{script_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
