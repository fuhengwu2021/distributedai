"""
Device Mesh visualization: 1D mesh for FSDP vs 2D mesh for HSDP (Hybrid Sharding).
Left: 1D mesh - all GPUs in one dimension, full sharding across all.
Right: 2D mesh - shard within node (dim 1), replicate across nodes (dim 0).
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

COLORS = {
    "gpu": "#3498db",
    "node": "#ecf0f1",
    "node_border": "#bdc3c7",
    "highlight": "#e74c3c",
    "text": "#2c3e50",
    "arrow": "#7f8c8d",
    "shard_colors": ["#3498db", "#e74c3c", "#2ecc71", "#f1c40f", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"],
}

def draw_gpu(ax, x, y, label, color="#3498db", size=0.4):
    rect = patches.FancyBboxPatch(
        (x - size/2, y - size/2), size, size,
        boxstyle="round,pad=0.02",
        linewidth=1.5, edgecolor="black", facecolor=color
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=9, fontweight="bold", color="white")

def panel_1d_mesh(ax):
    ax.set_xlim(-0.5, 9)
    ax.set_ylim(-1.5, 3)
    ax.axis("off")
    ax.set_title("1D Mesh: Full Sharding (FSDP)", fontsize=14, fontweight="bold", pad=10)
    
    # Draw 8 GPUs in a row
    ax.text(4, 2.5, "init_device_mesh(\"cuda\", (8,))", ha="center", fontsize=11, 
            family="monospace", style="italic")
    
    for i in range(8):
        draw_gpu(ax, 0.5 + i * 1.0, 1.5, f"GPU{i}", COLORS["shard_colors"][i])
    
    # Bracket showing single dimension
    ax.annotate("", xy=(0.2, 0.9), xytext=(8.3, 0.9),
                arrowprops=dict(arrowstyle="-", color=COLORS["arrow"], lw=1.5))
    ax.text(4.25, 0.5, "dim 0: world_size=8 (shard across all)", ha="center", fontsize=11)
    
    # Description
    ax.text(4, -0.3, "Parameters sharded across all 8 GPUs", ha="center", fontsize=11, color=COLORS["text"])
    ax.text(4, -0.8, "Each GPU holds 1/8 of parameters", ha="center", fontsize=10, color=COLORS["arrow"])

def panel_2d_mesh(ax):
    ax.set_xlim(-0.5, 9)
    ax.set_ylim(-1.5, 4.5)
    ax.axis("off")
    ax.set_title("2D Mesh: Hybrid Sharding (HSDP)", fontsize=14, fontweight="bold", pad=10)
    
    ax.text(4, 4.0, "init_device_mesh(\"cuda\", (2, 4))", ha="center", fontsize=11,
            family="monospace", style="italic")
    
    # Draw 2 nodes x 4 GPUs
    node_w, node_h = 4.2, 1.0
    gpu_spacing = 1.0
    
    for node_idx in range(2):
        node_y = 2.5 - node_idx * 1.5
        # Node box
        node_rect = patches.FancyBboxPatch(
            (0.1, node_y - 0.3), node_w, node_h,
            boxstyle="round,pad=0.05",
            linewidth=2, edgecolor=COLORS["node_border"], facecolor=COLORS["node"], alpha=0.5
        )
        ax.add_patch(node_rect)
        ax.text(-0.3, node_y + 0.2, f"Node {node_idx}", ha="right", fontsize=10, fontweight="bold")
        
        # GPUs in node
        for gpu_idx in range(4):
            color_idx = gpu_idx  # Same color for same position across nodes (replicated)
            draw_gpu(ax, 0.5 + gpu_idx * gpu_spacing, node_y + 0.2, 
                    f"GPU{node_idx*4 + gpu_idx}", COLORS["shard_colors"][color_idx], size=0.35)
    
    # Annotations for dimensions
    # Dim 0: across nodes (replicate)
    ax.annotate("", xy=(-0.1, 2.7), xytext=(-0.1, 1.2),
                arrowprops=dict(arrowstyle="<->", color="#e74c3c", lw=2))
    ax.text(-0.5, 1.95, "dim 0\n(replicate)", ha="right", fontsize=9, color="#e74c3c")
    
    # Dim 1: within node (shard)
    ax.annotate("", xy=(0.3, 3.3), xytext=(4.1, 3.3),
                arrowprops=dict(arrowstyle="<->", color="#2ecc71", lw=2))
    ax.text(2.2, 3.6, "dim 1 (shard within node)", ha="center", fontsize=9, color="#2ecc71")
    
    # Description
    ax.text(4, -0.3, "Shard within node (4 GPUs), replicate across nodes (2 nodes)", 
            ha="center", fontsize=10, color=COLORS["text"])
    ax.text(4, -0.8, "Reduces inter-node communication; each node has full model copy", 
            ha="center", fontsize=9, color=COLORS["arrow"])

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    panel_1d_mesh(ax1)
    panel_2d_mesh(ax2)
    
    plt.tight_layout(pad=1.0)
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
