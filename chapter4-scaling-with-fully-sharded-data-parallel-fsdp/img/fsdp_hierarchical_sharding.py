"""
Hierarchical sharding in FSDP: wrapping at different levels.
Shows how applying fully_shard to individual blocks vs the whole model
creates different FSDP units and communication boundaries.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

COLORS = {
    "embedding": "#95a5a6",
    "block": "#3498db",
    "block_sharded": "#2980b9",
    "lm_head": "#95a5a6",
    "fsdp_boundary": "#e74c3c",
    "text": "#2c3e50",
    "arrow": "#7f8c8d",
    "ranks": ["#3498db", "#e74c3c", "#2ecc71", "#f1c40f"],
}

def draw_module(ax, x, y, w, h, color, label, fontsize=10, edgecolor="black", linestyle="-", linewidth=1.5):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        linewidth=linewidth, edgecolor=edgecolor, facecolor=color, linestyle=linestyle
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", 
            fontsize=fontsize, fontweight="bold", color="white")

def draw_fsdp_boundary(ax, x, y, w, h, label=""):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03",
        linewidth=2.5, edgecolor=COLORS["fsdp_boundary"], facecolor="none", linestyle="--"
    )
    ax.add_patch(rect)
    if label:
        ax.text(x + w + 0.1, y + h/2, label, fontsize=9, color=COLORS["fsdp_boundary"], 
                va="center", fontweight="bold")

def panel_hierarchical(ax):
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-0.5, 6)
    ax.axis("off")
    ax.set_title("Hierarchical Sharding: Block-Level FSDP Units", fontsize=13, fontweight="bold", pad=10)
    
    # Model structure
    x_start = 0.5
    block_w, block_h = 1.8, 0.6
    spacing = 0.15
    
    # Embedding (not sharded in this example)
    draw_module(ax, x_start, 5.0, block_w, block_h, COLORS["embedding"], "Embedding")
    ax.text(x_start + block_w + 0.2, 5.3, "(not sharded)", fontsize=9, color=COLORS["arrow"], style="italic")
    
    # Transformer blocks (each is an FSDP unit)
    for i in range(4):
        y_pos = 4.2 - i * (block_h + spacing)
        draw_module(ax, x_start, y_pos, block_w, block_h, COLORS["block"], f"Block {i}")
        draw_fsdp_boundary(ax, x_start - 0.1, y_pos - 0.08, block_w + 0.2, block_h + 0.16, "FSDP unit")
    
    # LM Head (not sharded)
    draw_module(ax, x_start, 0.8, block_w, block_h, COLORS["lm_head"], "LM Head")
    ax.text(x_start + block_w + 0.2, 1.1, "(not sharded)", fontsize=9, color=COLORS["arrow"], style="italic")
    
    # Root FSDP
    draw_fsdp_boundary(ax, x_start - 0.3, 0.6, block_w + 0.6, 5.0, "Root FSDP")
    
    # Explanation
    ax.text(5.5, 4.5, "Code pattern:", fontsize=11, fontweight="bold")
    ax.text(5.5, 4.0, "for block in model.blocks:", fontsize=10, family="monospace")
    ax.text(5.5, 3.6, "    fully_shard(block)", fontsize=10, family="monospace")
    ax.text(5.5, 3.2, "fully_shard(model)  # root", fontsize=10, family="monospace")
    
    ax.text(5.5, 2.4, "Benefits:", fontsize=11, fontweight="bold")
    ax.text(5.5, 2.0, "• All-gather/reduce-scatter\n  at block boundaries", fontsize=10)
    ax.text(5.5, 1.2, "• Fine-grained memory\n  management", fontsize=10)
    ax.text(5.5, 0.4, "• Prefetching between\n  blocks possible", fontsize=10)

def panel_flat(ax):
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-0.5, 6)
    ax.axis("off")
    ax.set_title("Flat Sharding: Single FSDP Unit", fontsize=13, fontweight="bold", pad=10)
    
    x_start = 0.5
    block_w, block_h = 1.8, 0.6
    spacing = 0.15
    
    # All modules in one FSDP unit
    draw_module(ax, x_start, 5.0, block_w, block_h, COLORS["embedding"], "Embedding")
    for i in range(4):
        y_pos = 4.2 - i * (block_h + spacing)
        draw_module(ax, x_start, y_pos, block_w, block_h, COLORS["block"], f"Block {i}")
    draw_module(ax, x_start, 0.8, block_w, block_h, COLORS["lm_head"], "LM Head")
    
    # Single FSDP boundary around everything
    draw_fsdp_boundary(ax, x_start - 0.2, 0.6, block_w + 0.4, 5.0, "Single FSDP unit")
    
    # Explanation
    ax.text(5.5, 4.5, "Code pattern:", fontsize=11, fontweight="bold")
    ax.text(5.5, 4.0, "fully_shard(model)  # only", fontsize=10, family="monospace")
    
    ax.text(5.5, 3.0, "Characteristics:", fontsize=11, fontweight="bold")
    ax.text(5.5, 2.5, "• All params in one group", fontsize=10)
    ax.text(5.5, 2.0, "• Single all-gather for\n  entire model", fontsize=10)
    ax.text(5.5, 1.2, "• Higher peak memory\n  (all params at once)", fontsize=10)
    ax.text(5.5, 0.4, "• Simpler but less\n  memory-efficient", fontsize=10)

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    panel_hierarchical(ax1)
    panel_flat(ax2)
    
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
