import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

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

def draw_sequence_parallelism(ax):
    """Draw sequence parallelism: activations split along sequence dimension."""
    ax.set_xlim(0.8, 11.2)
    ax.set_ylim(2, 6)
    ax.axis('off')
    ax.set_title("Sequence Parallelism", fontsize=14, fontweight='bold', pad=15)
    
    # Full sequence (top)
    ax.text(6, 5.7, "Full Sequence (8K tokens)", ha='center', fontsize=13, fontweight='bold')
    
    # Draw full sequence bar
    full_seq = patches.FancyBboxPatch((1, 4.8), 10, 0.6, boxstyle="round,pad=0.02",
                                       facecolor='#E0E0E0', edgecolor=COLORS['border'], linewidth=1.5)
    ax.add_patch(full_seq)
    ax.text(6, 5.1, "Activations: (batch, 8K, hidden)", ha='center', fontsize=13, style='italic')
    
    # Arrow down
    ax.annotate("", xy=(6, 4.3), xytext=(6, 4.7),
                arrowprops=dict(arrowstyle="->", color=COLORS['arrow'], lw=2))
    ax.text(6.2, 4.5, "split", ha='left', fontsize=13)
    
    # Split across 2 GPUs
    ax.text(6, 3.9, "Split along sequence dimension", ha='center', fontsize=13)
    
    # GPU 0 chunk
    gpu0_box = patches.FancyBboxPatch((1.5, 2.35), 4, 1, boxstyle="round,pad=0.02",
                                       facecolor=COLORS['gpu0'], edgecolor='white', linewidth=2, alpha=0.8)
    ax.add_patch(gpu0_box)
    ax.text(3.5, 3.0, "GPU 0: tokens 0-4K", ha='center', va='center', fontsize=13, fontweight='bold', color='white')
    ax.text(3.5, 2.7, "(batch, 4K, hidden)", ha='center', va='center', fontsize=12, color='white')
    
    # GPU 1 chunk
    gpu1_box = patches.FancyBboxPatch((6.5, 2.35), 4, 1, boxstyle="round,pad=0.02",
                                       facecolor=COLORS['gpu1'], edgecolor='white', linewidth=2, alpha=0.8)
    ax.add_patch(gpu1_box)
    ax.text(8.5, 3.0, "GPU 1: tokens 4K-8K", ha='center', va='center', fontsize=13, fontweight='bold', color='white')
    ax.text(8.5, 2.7, "(batch, 4K, hidden)", ha='center', va='center', fontsize=12, color='white')
    
    # Memory savings note
    '''ax.text(6, 1.5, "Each GPU stores 50% of activation memory", ha='center', fontsize=13, 
            bbox=dict(boxstyle='round', facecolor='#FFF9C4', edgecolor='#F57F17', alpha=0.8))
    
    # Operations note
    ax.text(6, 0.7, "LayerNorm & Dropout operate on local chunks\nNo communication needed for these ops", 
            ha='center', fontsize=13, style='italic')'''


def draw_context_parallelism(ax):
    """Draw context parallelism with ring attention."""
    ax.set_xlim(1.5, 12.5)
    ax.set_ylim(1, 7)
    ax.axis('off')
    ax.set_title("Context Parallelism (Ring Attention)", fontsize=14, fontweight='bold', pad=15)
    
    # Title for the ring
    #ax.text(7, 7.5, "Each GPU holds local Q, K, V chunks", ha='center', fontsize=13)
    
    # Draw 4 GPUs in a ring layout
    gpu_positions = [(3, 5.5), (11, 5.5), (11, 2), (3, 2)]  # top-left, top-right, bottom-right, bottom-left
    gpu_labels = ['GPU 0\nQ₀, K₀, V₀', 'GPU 1\nQ₁, K₁, V₁', 'GPU 2\nQ₂, K₂, V₂', 'GPU 3\nQ₃, K₃, V₃']
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
    ax.text(7, 6.3, "pass K,V →", ha='center', fontsize=12, color=COLORS['kv_transfer'])
    
    # Right: GPU 1 -> GPU 2
    ax.annotate("", xy=(11.3, 3.3), xytext=(11.3, 4.7),
                arrowprops=dict(arrowstyle="->", color=COLORS['kv_transfer'], lw=2.5))
    ax.text(12, 4, "↓", ha='center', fontsize=13, color=COLORS['kv_transfer'])
    
    # Bottom: GPU 2 -> GPU 3
    ax.annotate("", xy=(4.5, 1.7), xytext=(9.5, 1.7),
                arrowprops=dict(arrowstyle="->", color=COLORS['kv_transfer'], lw=2.5,
                               connectionstyle="arc3,rad=-0.1"))
    ax.text(7, 1.2, "← pass K,V", ha='center', fontsize=12, color=COLORS['kv_transfer'])
    
    # Left: GPU 3 -> GPU 0
    ax.annotate("", xy=(2.7, 4.7), xytext=(2.7, 3.3),
                arrowprops=dict(arrowstyle="->", color=COLORS['kv_transfer'], lw=2.5))
    ax.text(2, 4, "↑", ha='center', fontsize=13, color=COLORS['kv_transfer'])
    
    # Center explanation
    ax.text(7, 3.95, "Ring Topology", ha='center', fontsize=13, fontweight='bold')
    ax.text(7, 3.3, "K,V chunks rotate\naround the ring", ha='center', fontsize=13, style='italic')
    
    # Bottom note
    '''ax.text(7, 0.5, "After one full rotation: every Q has attended to all K,V\nNo GPU ever holds the full sequence!", 
            ha='center', fontsize=13,
            bbox=dict(boxstyle='round', facecolor='#FFF9C4', edgecolor='#F57F17', alpha=0.8))'''


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    draw_sequence_parallelism(ax1)
    draw_context_parallelism(ax2)
    
    plt.tight_layout()
    
    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_path = os.path.join(script_dir, f"{script_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
