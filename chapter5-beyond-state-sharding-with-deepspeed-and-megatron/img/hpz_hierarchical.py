"""
hpZ (Hierarchical Partitioning): ZeRO-3 vs hpZ communication pattern.
Shows how hpZ reduces inter-node communication.
"""
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure shared directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

COLORS = {
    "shard_colors": ["#3498db", "#e74c3c", "#2ecc71", "#f1c40f"],
    "node_border": "#2c3e50",
    "node": "#ecf0f1",
    "nvlink": "#27ae60",
    "infiniband": "#e74c3c",
    "text": "#2c3e50",
}

LABELS = {
    "en": {
        "title_zero3": "ZeRO-3 (Full Sharding)",
        "title_hpz": "hpZ (Hierarchical)",
        "node_fmt": "Node {i}",
        "shard_fmt": "S{i}",
    },
    "zh": {
        "title_zero3": "ZeRO-3（全局完全分片）",
        "title_hpz": "hpZ（分层分片与节点间复制）",
        "node_fmt": "节点 {i} (Node {i})",
        "shard_fmt": "S{i}",
    },
}

def draw_gpu(ax, x, y, label, color, size=0.4):
    rect = patches.FancyBboxPatch(
        (x - size/2, y - size/2), size, size,
        boxstyle="round,pad=0.02",
        linewidth=1.5, edgecolor="black", facecolor=color
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=9, fontweight="bold", color="white")

def panel_zero3(ax, text):
    ax.set_xlim(0.3, 2.7)
    ax.set_ylim(-0.5, 5.5)
    ax.axis("off")
    ax.set_title(text["title_zero3"], fontsize=14, fontweight="bold", pad=10)
    
    # Two nodes with more vertical spacing
    node_w, node_h = 2.0, 1.3
    
    # Node 0 (top)
    node0 = patches.FancyBboxPatch(
        (0.5, 3.3), node_w, node_h,
        boxstyle="round,pad=0.05",
        linewidth=2, edgecolor=COLORS["node_border"], facecolor=COLORS["node"], alpha=0.5
    )
    ax.add_patch(node0)
    ax.text(1.5, 4.8, text["node_fmt"].format(i=0), ha="center", fontsize=13, fontweight="normal")
    
    # GPUs in Node 0 - each has different shard (2 GPUs)
    for i in range(2):
        draw_gpu(ax, 0.95 + i * 0.9, 3.9, text["shard_fmt"].format(i=i), COLORS["shard_colors"][i], size=0.45)
    
    # Node 1 (bottom)
    node1 = patches.FancyBboxPatch(
        (0.5, 0.5), node_w, node_h,
        boxstyle="round,pad=0.05",
        linewidth=2, edgecolor=COLORS["node_border"], facecolor=COLORS["node"], alpha=0.5
    )
    ax.add_patch(node1)
    ax.text(1.5, 0.1, text["node_fmt"].format(i=1), ha="center", fontsize=13, fontweight="normal")
    
    # GPUs in Node 1 - each has different shard (2 GPUs)
    for i in range(2):
        draw_gpu(ax, 0.95 + i * 0.9, 1.1, text["shard_fmt"].format(i=i+2), COLORS["shard_colors"][i+2], size=0.45)
    
    # Inter-node communication arrows - all GPUs communicate with each other
    # Arrows between GPUs in Node 0 and Node 1
    ax.annotate("", xy=(0.95, 3.25), xytext=(0.95, 1.85),
                arrowprops=dict(arrowstyle="<->", color=COLORS["infiniband"], lw=2))
    ax.annotate("", xy=(1.85, 3.25), xytext=(1.85, 1.85),
                arrowprops=dict(arrowstyle="<->", color=COLORS["infiniband"], lw=2))
    ax.annotate("", xy=(0.95, 3.25), xytext=(1.85, 1.85),
                arrowprops=dict(arrowstyle="<->", color=COLORS["infiniband"], lw=2))
    ax.annotate("", xy=(1.85, 3.25), xytext=(0.95, 1.85),
                arrowprops=dict(arrowstyle="<->", color=COLORS["infiniband"], lw=2))

def panel_hpz(ax, text):
    ax.set_xlim(0.3, 2.7)
    ax.set_ylim(-0.5, 5.5)
    ax.axis("off")
    ax.set_title(text["title_hpz"], fontsize=14, fontweight="bold", pad=10)
    
    node_w, node_h = 2.0, 1.3
    
    # Node 0 - all GPUs have same shard (S0)
    node0 = patches.FancyBboxPatch(
        (0.5, 3.3), node_w, node_h,
        boxstyle="round,pad=0.05",
        linewidth=2, edgecolor=COLORS["node_border"], facecolor=COLORS["node"], alpha=0.5
    )
    ax.add_patch(node0)
    ax.text(1.5, 4.8, text["node_fmt"].format(i=0), ha="center", fontsize=13, fontweight="normal")
    
    # All GPUs in Node 0 have same shard (replicated) - 2 GPUs
    for i in range(2):
        draw_gpu(ax, 0.95 + i * 0.9, 3.9, text["shard_fmt"].format(i=0), COLORS["shard_colors"][0], size=0.45)
    
    # Node 1 - all GPUs have same shard (S1)
    node1 = patches.FancyBboxPatch(
        (0.5, 0.5), node_w, node_h,
        boxstyle="round,pad=0.05",
        linewidth=2, edgecolor=COLORS["node_border"], facecolor=COLORS["node"], alpha=0.5
    )
    ax.add_patch(node1)
    ax.text(1.5, 0.1, text["node_fmt"].format(i=1), ha="center", fontsize=13, fontweight="normal")
    
    # All GPUs in Node 1 have same shard - 2 GPUs
    for i in range(2):
        draw_gpu(ax, 0.95 + i * 0.9, 1.1, text["shard_fmt"].format(i=1), COLORS["shard_colors"][1], size=0.45)
    
    # Inter-node communication - only one arrow between nodes
    ax.annotate("", xy=(1.4, 3.25), xytext=(1.4, 1.85),
                arrowprops=dict(arrowstyle="<->", color=COLORS["nvlink"], lw=3))

def draw(text: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    
    panel_zero3(ax1, text)
    panel_hpz(ax2, text)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    localized_figure(draw, "hpz_hierarchical", LABELS, __file__, pad_inches=0.08)
