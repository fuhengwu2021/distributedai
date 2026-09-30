#!/usr/bin/env python3
"""
Visualize the FSDP All-Gather and Reduce-Scatter communication flow.
Left: Forward pass showing sharded parameters gathered into full parameters.
Right: Backward pass showing full gradients reduced and scattered back to shards.
"""
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure shared directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

# Colors: more professional palette
COLORS = {
    "ranks": ["#3498db", "#e74c3c", "#2ecc71", "#f1c40f"],
    "arrow": "#7f8c8d",
    "text": "#2c3e50"
}
N = 4
BLOCK_W = 0.8
BLOCK_H = 0.5

LABELS = {
    "en": {
        "title_allgather": "Forward Pass: All-Gather",
        "step1_allgather": "1. Parameters are Sharded ($1/N$)",
        "rank_label": "Rank {r}",
        "param_sharded": "P{r}",
        "arrow_allgather": "Collectives: All-Gather",
        "step2_allgather": "2. Parameters Reconstructed for Forward",
        "full_p": "Full $P$",
        "title_reducescatter": "Backward Pass: Reduce-Scatter",
        "step1_reducescatter": "1. Gradients Computed for Full Layer",
        "grad_full": r"$\nabla$Full",
        "arrow_reducescatter": "Collectives: Reduce-Scatter",
        "step2_reducescatter": "2. Gradients Reduced & Sharded ($1/N$)",
        "sum_grad": r"$\sum G_{r}$",
    },
    "zh": {
        "title_allgather": "前向传播：All-Gather 聚合",
        "step1_allgather": "1. 参数处于分片状态 ($1/N$)",
        "rank_label": "Rank {r}",
        "param_sharded": "P{r}",
        "arrow_allgather": "集合通信：All-Gather",
        "step2_allgather": "2. 前向计算重构完整参数",
        "full_p": "完整参数 $P$",
        "title_reducescatter": "反向传播：Reduce-Scatter 规约切分",
        "step1_reducescatter": "1. 计算完整层的局部梯度",
        "grad_full": r"$\nabla$Full",
        "arrow_reducescatter": "集合通信：Reduce-Scatter",
        "step2_reducescatter": "2. 梯度规约并分片存储 ($1/N$)",
        "sum_grad": r"$\sum G_{r}$",
    },
}


def draw_block(ax, x, y, color, label, hatch=None, alpha=1.0):
    rect = patches.Rectangle(
        (x, y), BLOCK_W, BLOCK_H,
        linewidth=1.5, edgecolor="black", facecolor=color, hatch=hatch, alpha=alpha
    )
    ax.add_patch(rect)
    ax.text(
        x + BLOCK_W / 2, y + BLOCK_H / 2, label,
        ha="center", va="center", fontsize=13, fontweight="bold", color="white" if alpha > 0.8 else "black"
    )


def panel_allgather(ax, text: dict):
    ax.set_xlim(0.4, 5.4)
    ax.set_ylim(-.5, 4.5)
    ax.axis("off")
    ax.set_title(text["title_allgather"], fontsize=14, fontweight="bold", pad=2)

    # 1. Before: Sharded (1/N Memory)
    ax.text(2.75, 4.0, text["step1_allgather"], ha="center", fontsize=13, style="italic")
    for r in range(N):
        x_pos = 0.5 + r * 1.3
        ax.text(x_pos + BLOCK_W/2, 3.5, text["rank_label"].format(r=r), ha="center", fontsize=13)
        draw_block(ax, x_pos, 2.8, COLORS["ranks"][r], text["param_sharded"].format(r=r))

    # Arrow
    ax.annotate(text["arrow_allgather"], xy=(2.75, 1.5), xytext=(2.75, 2.6),
                arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=2),
                ha="center", fontsize=13, color=COLORS["arrow"])

    # 2. After: Gathered (Full Model State)
    ax.text(2.75, 1.2, text["step2_allgather"], ha="center", fontsize=13, style="italic")
    for r in range(N):
        x_base = 0.5 + r * 1.3
        ax.text(x_base + BLOCK_W/2, 0.7, text["rank_label"].format(r=r), ha="center", fontsize=13)
        # Draw the "full" parameter as a horizontal combined block
        for i in range(N):
            mini_w = BLOCK_W / N
            rect = patches.Rectangle((x_base + i*mini_w, 0.0), mini_w, BLOCK_H, 
                                     facecolor=COLORS["ranks"][i], edgecolor="black", linewidth=0.5)
            ax.add_patch(rect)
        ax.text(x_base + BLOCK_W/2, 0.0 - 0.3, text["full_p"], ha="center", fontsize=12)


def panel_reducescatter(ax, text: dict):
    ax.set_xlim(0.4, 5.4)
    ax.set_ylim(-0.5, 4.5)
    ax.axis("off")
    ax.set_title(text["title_reducescatter"], fontsize=14, fontweight="bold", pad=2)

    # 1. Before: Full Gradients computed locally
    ax.text(2.75, 4.0, text["step1_reducescatter"], ha="center", fontsize=13, style="italic")
    for r in range(N):
        x_pos = 0.5 + r * 1.3
        ax.text(x_pos + BLOCK_W/2, 3.5, text["rank_label"].format(r=r), ha="center", fontsize=13)
        draw_block(ax, x_pos, 2.8, COLORS["ranks"][r], text["grad_full"], hatch="///")

    # Arrow
    ax.annotate(text["arrow_reducescatter"], xy=(2.75, 1.5), xytext=(2.75, 2.6),
                arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=2),
                ha="center", fontsize=13, color=COLORS["arrow"])

    # 2. After: Sharded Gradients (Averaged/Summed)
    ax.text(2.75, 1.2, text["step2_reducescatter"], ha="center", fontsize=13, style="italic")
    for r in range(N):
        x_pos = 0.5 + r * 1.3
        ax.text(x_pos + BLOCK_W/2, 0.7, text["rank_label"].format(r=r), ha="center", fontsize=13)
        draw_block(ax, x_pos, 0.0, COLORS["ranks"][r], text["sum_grad"].format(r=r), hatch="...")


def draw(text: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    panel_allgather(ax1, text)
    panel_reducescatter(ax2, text)

    plt.tight_layout(pad=0.5)
    return fig


if __name__ == "__main__":
    localized_figure(draw, "fsdp_allgather_reducescatter", LABELS, __file__, pad_inches=0.08)