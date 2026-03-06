"""
reshard_after_forward comparison: True (ZeRO-3 style) vs False (ZeRO-2 style).
Shows memory vs communication tradeoff during forward and backward passes.
"""
import os
import matplotlib.pyplot as plt

from math4ai import save_figure
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
    ax.set_ylim(0.5, 5)
    ax.axis("off")
    ax.set_title("reshard_after_forward=True", fontsize=16, fontweight="bold", pad=10)
    
    # Timeline
    y_fwd = 3.5
    y_bwd = 1.5
    block_h = 0.7
    
    ax.text(-0.3, y_fwd + 0.3, "Fwd", fontsize=13, fontweight="bold", ha="right")
    ax.text(-0.3, y_bwd + 0.3, "Bwd", fontsize=13, fontweight="bold", ha="right")
    
    # Forward: all-gather -> compute -> free (reshard)
    draw_timeline_block(ax, 0, y_fwd, 1.8, block_h, COLORS["comm"], "AG", 11)
    draw_timeline_block(ax, 2.0, y_fwd, 2.2, block_h, COLORS["compute"], "L1", 11)
    ax.text(4.4, y_fwd + 0.35, "free", fontsize=10, color=COLORS["arrow"], style="italic")
    draw_timeline_block(ax, 5.0, y_fwd, 1.8, block_h, COLORS["comm"], "AG", 11)
    draw_timeline_block(ax, 7.0, y_fwd, 2.2, block_h, COLORS["compute"], "L2", 11)
    
    # Backward: all-gather -> compute grad -> reduce-scatter
    draw_timeline_block(ax, 0, y_bwd, 1.5, block_h, COLORS["comm"], "AG", 11)
    draw_timeline_block(ax, 1.6, y_bwd, 1.5, block_h, COLORS["compute"], "L2", 11)
    draw_timeline_block(ax, 3.2, y_bwd, 1.5, block_h, COLORS["grads"], "RS", 11)
    draw_timeline_block(ax, 4.8, y_bwd, 1.5, block_h, COLORS["comm"], "AG", 11)
    draw_timeline_block(ax, 6.4, y_bwd, 1.5, block_h, COLORS["compute"], "L1", 11)
    draw_timeline_block(ax, 8.0, y_bwd, 1.5, block_h, COLORS["grads"], "RS", 11)

def panel_reshard_false(ax):
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(0.5, 5)
    ax.axis("off")
    ax.set_title("reshard_after_forward=False", fontsize=16, fontweight="bold", pad=10)
    
    y_fwd = 3.5
    y_bwd = 1.5
    block_h = 0.7
    
    ax.text(-0.3, y_fwd + 0.3, "Fwd", fontsize=13, fontweight="bold", ha="right")
    ax.text(-0.3, y_bwd + 0.3, "Bwd", fontsize=13, fontweight="bold", ha="right")
    
    # Forward: all-gather -> compute -> KEEP (no reshard)
    draw_timeline_block(ax, 0, y_fwd, 1.8, block_h, COLORS["comm"], "AG", 11)
    draw_timeline_block(ax, 2.0, y_fwd, 2.2, block_h, COLORS["compute"], "L1", 11)
    ax.text(4.4, y_fwd + 0.35, "keep", fontsize=10, color="#2ecc71", style="italic", fontweight="bold")
    draw_timeline_block(ax, 5.0, y_fwd, 1.8, block_h, COLORS["comm"], "AG", 11)
    draw_timeline_block(ax, 7.0, y_fwd, 2.2, block_h, COLORS["compute"], "L2", 11)
    
    # Backward: NO all-gather needed (params still in memory) -> compute grad -> reduce-scatter
    draw_timeline_block(ax, 0, y_bwd, 2.2, block_h, COLORS["compute"], "L2", 11)
    draw_timeline_block(ax, 2.4, y_bwd, 1.8, block_h, COLORS["grads"], "RS", 11)
    draw_timeline_block(ax, 4.4, y_bwd, 2.2, block_h, COLORS["compute"], "L1", 11)
    draw_timeline_block(ax, 6.8, y_bwd, 1.8, block_h, COLORS["grads"], "RS", 11)

def main():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    panel_reshard_true(ax1)
    panel_reshard_false(ax2)
    
    plt.tight_layout(pad=1.0)
    save_figure(__file__)

if __name__ == "__main__":
    main()
