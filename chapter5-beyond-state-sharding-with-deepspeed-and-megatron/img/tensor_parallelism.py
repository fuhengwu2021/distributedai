"""
Tensor Parallelism: Column-parallel and Row-parallel linear layers.
Shows how weight matrices are split across GPUs.
"""
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure shared directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

COLORS = {
    "input": "#3498db",
    "weight": "#2ecc71",
    "output": "#e74c3c",
    "gpu0": "#3498db",
    "gpu1": "#e74c3c",
    "comm": "#9b59b6",
    "text": "#2c3e50",
}

LABELS = {
    "en": {
        "title_col": "Column-Parallel Linear",
        "title_row": "Row-Parallel Linear",
        "full": "(full)",
        "split": "(split)",
        "columns": "columns",
        "rows": "rows",
        "no_comm": "No comm",
        "all_reduce": "All-Reduce",
        "gpu0": "GPU 0",
        "gpu1": "GPU 1",
        "w0": "W₀",
        "w1": "W₁",
        "x0": "X₀",
        "x1": "X₁",
        "y0": "Y₀",
        "y1": "Y₁",
        "y0_prime": "Y₀'",
        "y1_prime": "Y₁'",
    },
    "zh": {
        "title_col": "列并行线性层 (Column-Parallel Linear)",
        "title_row": "行并行线性层 (Row-Parallel Linear)",
        "full": "（全量）",
        "split": "（分片）",
        "columns": "按列切分",
        "rows": "按行切分",
        "no_comm": "无需通信",
        "all_reduce": "All-Reduce 通信",
        "gpu0": "GPU 0",
        "gpu1": "GPU 1",
        "w0": "$W_0$",
        "w1": "$W_1$",
        "x0": "$X_0$",
        "x1": "$X_1$",
        "y0": "$Y_0$",
        "y1": "$Y_1$",
        "y0_prime": "$Y_0'$",
        "y1_prime": "$Y_1'$",
    },
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

def panel_column_parallel(ax, text):
    ax.set_xlim(0.4, 8.5)
    ax.set_ylim(2, 4)
    ax.axis("off")
    ax.set_title(text["title_col"], fontsize=14, fontweight="bold", pad=10)
    
    # Input X (replicated on both GPUs)
    draw_matrix(ax, 0.5, 2.5, 1.5, 1.0, COLORS["input"], "X", 11)
    ax.text(1.25, 2.2, text["full"], fontsize=9, ha="center", color=COLORS["text"])
    
    # Weight matrix split column-wise
    draw_matrix(ax, 3.0, 2.8, 0.8, 0.7, COLORS["gpu0"], text["w0"], 10)
    draw_matrix(ax, 3.9, 2.8, 0.8, 0.7, COLORS["gpu1"], text["w1"], 10)
    ax.text(3.85, 2.5, text["columns"], fontsize=9, ha="center", color=COLORS["text"])
    
    # Arrows
    ax.annotate("", xy=(2.8, 3.0), xytext=(2.1, 3.0),
                arrowprops=dict(arrowstyle="->", color=COLORS["text"], lw=1.5))
    ax.annotate("", xy=(5.0, 3.1), xytext=(4.8, 3.1),
                arrowprops=dict(arrowstyle="->", color=COLORS["text"], lw=1.5))
    
    # Output (split)
    draw_matrix(ax, 5.3, 2.8, 0.6, 0.7, COLORS["gpu0"], text["y0"], 9)
    draw_matrix(ax, 6.0, 2.8, 0.6, 0.7, COLORS["gpu1"], text["y1"], 9)
    ax.text(6.0, 2.5, text["split"], fontsize=9, ha="center", color=COLORS["text"])
    
    # GPU labels
    ax.text(3.4, 3.7, text["gpu0"], fontsize=9, ha="center", color=COLORS["gpu0"], fontweight="bold")
    ax.text(4.3, 3.7, text["gpu1"], fontsize=9, ha="center", color=COLORS["gpu1"], fontweight="bold")
    
    # No communication needed
    ax.text(7.5, 3.0, text["no_comm"], fontsize=10, ha="center", color="#27ae60", fontweight="bold")

def panel_row_parallel(ax, text):
    ax.set_xlim(0.4, 8.5)
    ax.set_ylim(2, 4)
    ax.axis("off")
    ax.set_title(text["title_row"], fontsize=14, fontweight="bold", pad=10)
    
    # Input X (already split from column-parallel)
    draw_matrix(ax, 0.5, 2.8, 0.6, 0.7, COLORS["gpu0"], text["x0"], 9)
    draw_matrix(ax, 1.2, 2.8, 0.6, 0.7, COLORS["gpu1"], text["x1"], 9)
    ax.text(1.15, 2.5, text["split"], fontsize=9, ha="center", color=COLORS["text"])
    
    # Weight matrix split row-wise
    draw_matrix(ax, 2.8, 3.0, 0.8, 0.5, COLORS["gpu0"], text["w0"], 9)
    draw_matrix(ax, 2.8, 2.4, 0.8, 0.5, COLORS["gpu1"], text["w1"], 9)
    ax.text(3.2, 2.1, text["rows"], fontsize=9, ha="center", color=COLORS["text"])
    
    # Arrows
    ax.annotate("", xy=(2.6, 3.0), xytext=(1.9, 3.0),
                arrowprops=dict(arrowstyle="->", color=COLORS["text"], lw=1.5))
    ax.annotate("", xy=(4.0, 2.75), xytext=(3.7, 2.75),
                arrowprops=dict(arrowstyle="->", color=COLORS["text"], lw=1.5))
    
    # Partial outputs
    draw_matrix(ax, 4.3, 2.8, 0.6, 0.7, COLORS["gpu0"], text["y0_prime"], 9)
    draw_matrix(ax, 5.0, 2.8, 0.6, 0.7, COLORS["gpu1"], text["y1_prime"], 9)
    
    # All-reduce
    ax.annotate("", xy=(6.5, 3.1), xytext=(5.7, 3.1),
                arrowprops=dict(arrowstyle="->", color=COLORS["comm"], lw=2))
    ax.text(6.1, 3.5, text["all_reduce"], fontsize=9, ha="center", color=COLORS["comm"], fontweight="bold")
    
    # Final output (full)
    draw_matrix(ax, 6.8, 2.5, 1.5, 1.0, COLORS["output"], "Y", 11)
    ax.text(7.55, 2.2, text["full"], fontsize=9, ha="center", color=COLORS["text"])

def draw(text: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    
    panel_column_parallel(ax1, text)
    panel_row_parallel(ax2, text)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    localized_figure(draw, "tensor_parallelism", LABELS, __file__, pad_inches=0.08)
