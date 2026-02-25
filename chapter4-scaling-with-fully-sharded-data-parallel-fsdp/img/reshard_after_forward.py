"""
reshard_after_forward comparison: True (ZeRO-3 style) vs False (ZeRO-2 style).
Shows memory vs communication tradeoff during forward and backward passes.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

COLORS = {
    "params_sharded": "#3498db",
    "params_full": "#2ecc71",
    "grads": "#e74c3c",
    "compute": "#f1c40f",
    "comm": "#9b59b6",
    "text": "#2c3e50",
    "arrow": "#7f8c8d",
    "memory_bar": "#3498db",
    "comm_bar": "#e74c3c",
}

def draw_timeline_block(ax, x, y, w, h, color, label="", fontsize=9):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01",
        linewidth=1, edgecolor="black", facecolor=color
    )
    ax.add_patch(rect)
    if label:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", 
                fontsize=fontsize, fontweight="bold", color="white")

def panel_reshard_true(ax):
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-1, 5)
    ax.axis("off")
    ax.set_title("reshard_after_forward=True (ZeRO-3 style)", fontsize=13, fontweight="bold", pad=5)
    
    # Timeline
    y_fwd = 3.5
    y_bwd = 1.5
    
    ax.text(-0.3, y_fwd + 0.2, "Forward", fontsize=11, fontweight="bold", ha="right")
    ax.text(-0.3, y_bwd + 0.2, "Backward", fontsize=11, fontweight="bold", ha="right")
    
    # Forward: all-gather -> compute -> free (reshard)
    draw_timeline_block(ax, 0, y_fwd, 1.5, 0.5, COLORS["comm"], "All-Gather", 8)
    draw_timeline_block(ax, 1.6, y_fwd, 2.0, 0.5, COLORS["compute"], "Compute L1", 8)
    ax.text(3.8, y_fwd + 0.25, "free", fontsize=8, color=COLORS["arrow"], style="italic")
    draw_timeline_block(ax, 4.2, y_fwd, 1.5, 0.5, COLORS["comm"], "All-Gather", 8)
    draw_timeline_block(ax, 5.8, y_fwd, 2.0, 0.5, COLORS["compute"], "Compute L2", 8)
    ax.text(8.0, y_fwd + 0.25, "free", fontsize=8, color=COLORS["arrow"], style="italic")
    
    # Backward: all-gather -> compute grad -> reduce-scatter
    draw_timeline_block(ax, 0, y_bwd, 1.5, 0.5, COLORS["comm"], "All-Gather", 8)
    draw_timeline_block(ax, 1.6, y_bwd, 1.5, 0.5, COLORS["compute"], "Grad L2", 8)
    draw_timeline_block(ax, 3.2, y_bwd, 1.5, 0.5, COLORS["grads"], "Red-Scat", 8)
    draw_timeline_block(ax, 4.8, y_bwd, 1.5, 0.5, COLORS["comm"], "All-Gather", 8)
    draw_timeline_block(ax, 6.4, y_bwd, 1.5, 0.5, COLORS["compute"], "Grad L1", 8)
    draw_timeline_block(ax, 8.0, y_bwd, 1.5, 0.5, COLORS["grads"], "Red-Scat", 8)
    
    # Memory indicator
    ax.text(5, 0.3, "Memory: Low (only 1 layer params at a time)", 
            ha="center", fontsize=10, color=COLORS["memory_bar"])
    ax.text(5, -0.2, "Communication: High (all-gather in both fwd & bwd)", 
            ha="center", fontsize=10, color=COLORS["comm_bar"])

def panel_reshard_false(ax):
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-1, 5)
    ax.axis("off")
    ax.set_title("reshard_after_forward=False (ZeRO-2 style)", fontsize=13, fontweight="bold", pad=5)
    
    y_fwd = 3.5
    y_bwd = 1.5
    
    ax.text(-0.3, y_fwd + 0.2, "Forward", fontsize=11, fontweight="bold", ha="right")
    ax.text(-0.3, y_bwd + 0.2, "Backward", fontsize=11, fontweight="bold", ha="right")
    
    # Forward: all-gather -> compute -> KEEP (no reshard)
    draw_timeline_block(ax, 0, y_fwd, 1.5, 0.5, COLORS["comm"], "All-Gather", 8)
    draw_timeline_block(ax, 1.6, y_fwd, 2.0, 0.5, COLORS["compute"], "Compute L1", 8)
    ax.text(3.8, y_fwd + 0.25, "keep", fontsize=8, color="#2ecc71", style="italic", fontweight="bold")
    draw_timeline_block(ax, 4.2, y_fwd, 1.5, 0.5, COLORS["comm"], "All-Gather", 8)
    draw_timeline_block(ax, 5.8, y_fwd, 2.0, 0.5, COLORS["compute"], "Compute L2", 8)
    ax.text(8.0, y_fwd + 0.25, "keep", fontsize=8, color="#2ecc71", style="italic", fontweight="bold")
    
    # Backward: NO all-gather needed (params still in memory) -> compute grad -> reduce-scatter
    draw_timeline_block(ax, 0, y_bwd, 1.8, 0.5, COLORS["compute"], "Grad L2", 8)
    draw_timeline_block(ax, 1.9, y_bwd, 1.5, 0.5, COLORS["grads"], "Red-Scat", 8)
    draw_timeline_block(ax, 3.5, y_bwd, 1.8, 0.5, COLORS["compute"], "Grad L1", 8)
    draw_timeline_block(ax, 5.4, y_bwd, 1.5, 0.5, COLORS["grads"], "Red-Scat", 8)
    ax.text(8.0, y_bwd + 0.25, "(no all-gather)", fontsize=8, color="#2ecc71", style="italic")
    
    # Memory indicator
    ax.text(5, 0.3, "Memory: High (all layer params kept after fwd)", 
            ha="center", fontsize=10, color=COLORS["comm_bar"])
    ax.text(5, -0.2, "Communication: Low (no all-gather in backward)", 
            ha="center", fontsize=10, color=COLORS["memory_bar"])

def main():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    panel_reshard_true(ax1)
    panel_reshard_false(ax2)
    
    plt.tight_layout(pad=1.5)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_path = os.path.join(script_dir, f"{script_name}.png")
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.08,
    )
    print(f"Saved figure to: {output_path}")
    plt.close()

if __name__ == "__main__":
    main()
