"""
ZeRO Stages Comparison: Memory layout across ZeRO-1, ZeRO-2, ZeRO-3.
Shows what each GPU stores at each stage.
"""
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure shared directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

COLORS = {
    "params": "#3498db",
    "grads": "#e74c3c", 
    "optimizer": "#2ecc71",
    "sharded": "#9b59b6",
    "text": "#2c3e50",
}

LABELS = {
    "en": {
        "title_ddp": "DDP (Baseline)",
        "title_zero1": "ZeRO-1",
        "title_zero2": "ZeRO-2",
        "title_zero3": "ZeRO-3",
        "params": "Params",
        "params_sharded": "P",
        "grads": "Grads",
        "grads_sharded": "G",
        "optim": "Optim",
        "optim_sharded": "O",
        "legend_params": "■ Params",
        "legend_grads": "■ Grads",
        "legend_optim": "■ Optimizer",
        "legend_note": "(smaller = sharded)",
        "rank": "R{i}",
    },
    "zh": {
        "title_ddp": "DDP（基线）",
        "title_zero1": "ZeRO-1",
        "title_zero2": "ZeRO-2",
        "title_zero3": "ZeRO-3",
        "params": "参数",
        "params_sharded": "P",
        "grads": "梯度",
        "grads_sharded": "G",
        "optim": "优化器",
        "optim_sharded": "O",
        "legend_params": "■ 参数 (Params)",
        "legend_grads": "■ 梯度 (Grads)",
        "legend_optim": "■ 优化器 (Optimizer)",
        "legend_note": "（小色块代表分片存储）",
        "rank": "R{i}",
    },
}

def draw_block(ax, x, y, w, h, color, label="", fontsize=13, alpha=1.0):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        linewidth=1.5, edgecolor="black", facecolor=color, alpha=alpha
    )
    ax.add_patch(rect)
    if label:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", 
                fontsize=fontsize, fontweight="bold", color="white")

def draw_gpu_row(ax, y, params_sharded, grads_sharded, opt_sharded, gpu_label, text):
    """Draw a single GPU's memory layout."""
    x = 1.0
    block_h = 0.6
    spacing = 0.1
    
    # GPU label
    ax.text(0.5, y + block_h/2, gpu_label, ha="center", va="center", 
            fontsize=13, fontweight="bold")
    
    # Parameters
    if params_sharded:
        draw_block(ax, x, y, 0.8, block_h, COLORS["params"], text["params_sharded"], 10, alpha=0.7)
    else:
        draw_block(ax, x, y, 2.0, block_h, COLORS["params"], text["params"], 10)
    
    # Gradients
    grad_x = x + (0.9 if params_sharded else 2.1)
    if grads_sharded:
        draw_block(ax, grad_x, y, 0.8, block_h, COLORS["grads"], text["grads_sharded"], 10, alpha=0.7)
    else:
        draw_block(ax, grad_x, y, 2.0, block_h, COLORS["grads"], text["grads"], 10)
    
    # Optimizer states
    opt_x = grad_x + (0.9 if grads_sharded else 2.1)
    if opt_sharded:
        draw_block(ax, opt_x, y, 0.8, block_h, COLORS["optimizer"], text["optim_sharded"], 10, alpha=0.7)
    else:
        draw_block(ax, opt_x, y, 2.0, block_h, COLORS["optimizer"], text["optim"], 10)

def panel_ddp(ax, text):
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    ax.set_title(text["title_ddp"], fontsize=14, fontweight="bold", pad=10)
    
    for i in range(2):
        draw_gpu_row(ax, 1.5 - i * 1.0, False, False, False, text["rank"].format(i=i), text)

def panel_zero1(ax, text):
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    ax.set_title(text["title_zero1"], fontsize=14, fontweight="bold", pad=10)
    
    for i in range(2):
        draw_gpu_row(ax, 1.5 - i * 1.0, False, False, True, text["rank"].format(i=i), text)

def panel_zero2(ax, text):
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    ax.set_title(text["title_zero2"], fontsize=14, fontweight="bold", pad=10)
    
    for i in range(2):
        draw_gpu_row(ax, 1.5 - i * 1.0, False, True, True, text["rank"].format(i=i), text)

def panel_zero3(ax, text):
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    ax.set_title(text["title_zero3"], fontsize=14, fontweight="bold", pad=10)
    
    for i in range(2):
        draw_gpu_row(ax, 1.5 - i * 1.0, True, True, True, text["rank"].format(i=i), text)

def draw(text: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(14, 2.5))
    
    panel_ddp(axes[0], text)
    panel_zero1(axes[1], text)
    panel_zero2(axes[2], text)
    panel_zero3(axes[3], text)
    
    # Legend
    legend_y = 0.1
    fig.text(0.25, legend_y, text["legend_params"], color=COLORS["params"], fontsize=13, 
             ha="center", fontweight="bold")
    fig.text(0.45, legend_y, text["legend_grads"], color=COLORS["grads"], fontsize=13,
             ha="center", fontweight="bold")
    fig.text(0.65, legend_y, text["legend_optim"], color=COLORS["optimizer"], fontsize=13,
             ha="center", fontweight="bold")
    fig.text(0.85, legend_y, text["legend_note"], color=COLORS["text"], fontsize=13,
             ha="center", style="italic")
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    return fig

if __name__ == "__main__":
    localized_figure(draw, "zero_stages_comparison", LABELS, __file__, pad_inches=0.08)
