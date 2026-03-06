"""
Tensor Parallelism: Column-parallel and Row-parallel linear layers.
Shows how weight matrices are split across GPUs.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from math4ai import save_figure

COLORS = {
    "input": "#3498db",
    "weight": "#2ecc71",
    "output": "#e74c3c",
    "gpu0": "#3498db",
    "gpu1": "#e74c3c",
    "comm": "#9b59b6",
    "text": "#2c3e50",
}

def draw_matrix(ax, x, y, w, h, color, label="", fontsize=10, alpha=1.0):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01",
        linewidth=1.5, edgecolor="black", facecolor=color, alpha=alpha
    )
    ax.add_patch(rect)
    if label:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", 
                fontsize=fontsize, fontweight="bold", color="white")

def panel_column_parallel(ax):
    ax.set_xlim(0.4, 8.5)
    ax.set_ylim(2, 4)
    ax.axis("off")
    ax.set_title("Column-Parallel Linear", fontsize=14, fontweight="bold", pad=10)
    
    # Input X (replicated on both GPUs)
    draw_matrix(ax, 0.5, 2.5, 1.5, 1.0, COLORS["input"], "X", 11)
    ax.text(1.25, 2.2, "(full)", fontsize=9, ha="center", color=COLORS["text"])
    
    # Weight matrix split column-wise
    draw_matrix(ax, 3.0, 2.8, 0.8, 0.7, COLORS["gpu0"], "W₀", 10)
    draw_matrix(ax, 3.9, 2.8, 0.8, 0.7, COLORS["gpu1"], "W₁", 10)
    ax.text(3.85, 2.5, "columns", fontsize=9, ha="center", color=COLORS["text"])
    
    # Arrows
    ax.annotate("", xy=(2.8, 3.0), xytext=(2.1, 3.0),
                arrowprops=dict(arrowstyle="->", color=COLORS["text"], lw=1.5))
    ax.annotate("", xy=(5.0, 3.1), xytext=(4.8, 3.1),
                arrowprops=dict(arrowstyle="->", color=COLORS["text"], lw=1.5))
    
    # Output (split)
    draw_matrix(ax, 5.3, 2.8, 0.6, 0.7, COLORS["gpu0"], "Y₀", 9)
    draw_matrix(ax, 6.0, 2.8, 0.6, 0.7, COLORS["gpu1"], "Y₁", 9)
    ax.text(6.0, 2.5, "(split)", fontsize=9, ha="center", color=COLORS["text"])
    
    # GPU labels
    ax.text(3.4, 3.7, "GPU 0", fontsize=9, ha="center", color=COLORS["gpu0"], fontweight="bold")
    ax.text(4.3, 3.7, "GPU 1", fontsize=9, ha="center", color=COLORS["gpu1"], fontweight="bold")
    
    # No communication needed
    ax.text(7.5, 3.0, "No comm", fontsize=10, ha="center", color="#27ae60", fontweight="bold")

def panel_row_parallel(ax):
    ax.set_xlim(0.4, 8.5)
    ax.set_ylim(2, 4)
    ax.axis("off")
    ax.set_title("Row-Parallel Linear", fontsize=14, fontweight="bold", pad=10)
    
    # Input X (already split from column-parallel)
    draw_matrix(ax, 0.5, 2.8, 0.6, 0.7, COLORS["gpu0"], "X₀", 9)
    draw_matrix(ax, 1.2, 2.8, 0.6, 0.7, COLORS["gpu1"], "X₁", 9)
    ax.text(1.15, 2.5, "(split)", fontsize=9, ha="center", color=COLORS["text"])
    
    # Weight matrix split row-wise
    draw_matrix(ax, 2.8, 3.0, 0.8, 0.5, COLORS["gpu0"], "W₀", 9)
    draw_matrix(ax, 2.8, 2.4, 0.8, 0.5, COLORS["gpu1"], "W₁", 9)
    ax.text(3.2, 2.1, "rows", fontsize=9, ha="center", color=COLORS["text"])
    
    # Arrows
    ax.annotate("", xy=(2.6, 3.0), xytext=(1.9, 3.0),
                arrowprops=dict(arrowstyle="->", color=COLORS["text"], lw=1.5))
    ax.annotate("", xy=(4.0, 2.75), xytext=(3.7, 2.75),
                arrowprops=dict(arrowstyle="->", color=COLORS["text"], lw=1.5))
    
    # Partial outputs
    draw_matrix(ax, 4.3, 2.8, 0.6, 0.7, COLORS["gpu0"], "Y₀'", 9)
    draw_matrix(ax, 5.0, 2.8, 0.6, 0.7, COLORS["gpu1"], "Y₁'", 9)
    
    # All-reduce
    ax.annotate("", xy=(6.5, 3.1), xytext=(5.7, 3.1),
                arrowprops=dict(arrowstyle="->", color=COLORS["comm"], lw=2))
    ax.text(6.1, 3.5, "All-Reduce", fontsize=9, ha="center", color=COLORS["comm"], fontweight="bold")
    
    # Final output (full)
    draw_matrix(ax, 6.8, 2.5, 1.5, 1.0, COLORS["output"], "Y", 11)
    ax.text(7.55, 2.2, "(full)", fontsize=9, ha="center", color=COLORS["text"])

def main():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    
    panel_column_parallel(ax1)
    panel_row_parallel(ax2)
    
    plt.tight_layout()
    save_figure(__file__)

if __name__ == "__main__":
    main()
