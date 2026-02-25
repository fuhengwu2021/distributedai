"""
hpZ (Hierarchical Partitioning): ZeRO-3 vs hpZ communication pattern.
Shows how hpZ reduces inter-node communication.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

COLORS = {
    "shard_colors": ["#3498db", "#e74c3c", "#2ecc71", "#f1c40f"],
    "node_border": "#2c3e50",
    "node": "#ecf0f1",
    "nvlink": "#27ae60",
    "infiniband": "#e74c3c",
    "text": "#2c3e50",
}

def draw_gpu(ax, x, y, label, color, size=0.4):
    rect = patches.FancyBboxPatch(
        (x - size/2, y - size/2), size, size,
        boxstyle="round,pad=0.02",
        linewidth=1.5, edgecolor="black", facecolor=color
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=9, fontweight="bold", color="white")

def panel_zero3(ax):
    ax.set_xlim(0.3, 2.7)
    ax.set_ylim(-0.5, 5.5)
    ax.axis("off")
    ax.set_title("ZeRO-3 (Full Sharding)", fontsize=14, fontweight="bold", pad=10)
    
    # Two nodes with more vertical spacing
    node_w, node_h = 2.0, 1.3
    
    # Node 0 (top)
    node0 = patches.FancyBboxPatch(
        (0.5, 3.3), node_w, node_h,
        boxstyle="round,pad=0.05",
        linewidth=2, edgecolor=COLORS["node_border"], facecolor=COLORS["node"], alpha=0.5
    )
    ax.add_patch(node0)
    ax.text(1.5, 4.8, "Node 0", ha="center", fontsize=13, fontweight="normal")
    
    # GPUs in Node 0 - each has different shard (2 GPUs)
    for i in range(2):
        draw_gpu(ax, 0.95 + i * 0.9, 3.9, f"S{i}", COLORS["shard_colors"][i], size=0.45)
    
    # Node 1 (bottom)
    node1 = patches.FancyBboxPatch(
        (0.5, 0.5), node_w, node_h,
        boxstyle="round,pad=0.05",
        linewidth=2, edgecolor=COLORS["node_border"], facecolor=COLORS["node"], alpha=0.5
    )
    ax.add_patch(node1)
    ax.text(1.5, 0.1, "Node 1", ha="center", fontsize=13, fontweight="normal")
    
    # GPUs in Node 1 - each has different shard (2 GPUs)
    for i in range(2):
        draw_gpu(ax, 0.95 + i * 0.9, 1.1, f"S{i+2}", COLORS["shard_colors"][i+2], size=0.45)
    
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

def panel_hpz(ax):
    ax.set_xlim(0.3, 2.7)
    ax.set_ylim(-0.5, 5.5)
    ax.axis("off")
    ax.set_title("hpZ (Hierarchical)", fontsize=14, fontweight="bold", pad=10)
    
    node_w, node_h = 2.0, 1.3
    
    # Node 0 - all GPUs have same shard (S0)
    node0 = patches.FancyBboxPatch(
        (0.5, 3.3), node_w, node_h,
        boxstyle="round,pad=0.05",
        linewidth=2, edgecolor=COLORS["node_border"], facecolor=COLORS["node"], alpha=0.5
    )
    ax.add_patch(node0)
    ax.text(1.5, 4.8, "Node 0", ha="center", fontsize=13, fontweight="normal")
    
    # All GPUs in Node 0 have same shard (replicated) - 2 GPUs
    for i in range(2):
        draw_gpu(ax, 0.95 + i * 0.9, 3.9, f"S0", COLORS["shard_colors"][0], size=0.45)
    
    # Node 1 - all GPUs have same shard (S1)
    node1 = patches.FancyBboxPatch(
        (0.5, 0.5), node_w, node_h,
        boxstyle="round,pad=0.05",
        linewidth=2, edgecolor=COLORS["node_border"], facecolor=COLORS["node"], alpha=0.5
    )
    ax.add_patch(node1)
    ax.text(1.5, 0.1, "Node 1", ha="center", fontsize=13, fontweight="normal")
    
    # All GPUs in Node 1 have same shard - 2 GPUs
    for i in range(2):
        draw_gpu(ax, 0.95 + i * 0.9, 1.1, f"S1", COLORS["shard_colors"][1], size=0.45)
    
    # Inter-node communication - only one arrow between nodes
    ax.annotate("", xy=(1.4, 3.25), xytext=(1.4, 1.85),
                arrowprops=dict(arrowstyle="<->", color=COLORS["nvlink"], lw=3))

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    
    panel_zero3(ax1)
    panel_hpz(ax2)
    
    plt.tight_layout()
    
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
