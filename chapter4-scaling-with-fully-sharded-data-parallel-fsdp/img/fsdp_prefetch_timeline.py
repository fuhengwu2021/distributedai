"""
FSDP prefetching timeline: shows how communication overlaps with computation.
Compares no prefetching (sequential) vs prefetching (overlapped).
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

COLORS = {
    "compute": "#f1c40f",
    "allgather": "#9b59b6",
    "idle": "#ecf0f1",
    "text": "#2c3e50",
}

def draw_block(ax, x, y, w, h, color, label="", fontsize=11):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        linewidth=1.5, edgecolor="black", facecolor=color
    )
    ax.add_patch(rect)
    if label:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", 
                fontsize=fontsize, fontweight="bold", color="white")

def panel_no_prefetch(ax):
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.5, 2.5)
    ax.axis("off")
    ax.set_title("Without Prefetching", fontsize=16, fontweight="bold", pad=10)
    
    block_h = 0.8
    y = 1.0
    
    # Layer 0: AG -> Compute
    draw_block(ax, 0, y, 1.2, block_h, COLORS["allgather"], "AG₀", 10)
    draw_block(ax, 1.3, y, 2.0, block_h, COLORS["compute"], "L₀", 12)
    
    # Layer 1: AG -> Compute (sequential, starts after L0)
    draw_block(ax, 3.4, y, 1.2, block_h, COLORS["allgather"], "AG₁", 10)
    draw_block(ax, 4.7, y, 2.0, block_h, COLORS["compute"], "L₁", 12)
    
    # Layer 2: AG -> Compute
    draw_block(ax, 6.8, y, 1.2, block_h, COLORS["allgather"], "AG₂", 10)
    draw_block(ax, 8.1, y, 2.0, block_h, COLORS["compute"], "L₂", 12)
    
    # Time arrow
    ax.annotate("", xy=(11, 0.3), xytext=(0, 0.3),
                arrowprops=dict(arrowstyle="->", color=COLORS["text"], lw=1.5))
    ax.text(5.5, 0.0, "time", ha="center", fontsize=11, color=COLORS["text"])

def panel_with_prefetch(ax):
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.5, 2.5)
    ax.axis("off")
    ax.set_title("With Prefetching", fontsize=16, fontweight="bold", pad=10)
    
    block_h = 0.8
    y_compute = 1.2
    y_comm = 0.3
    
    # Layer 0: AG then Compute
    draw_block(ax, 0, y_compute, 1.0, block_h, COLORS["allgather"], "AG₀", 10)
    draw_block(ax, 1.1, y_compute, 2.0, block_h, COLORS["compute"], "L₀", 12)
    
    # AG₁ overlaps with L₀ compute
    draw_block(ax, 1.1, y_comm, 1.0, block_h, COLORS["allgather"], "AG₁", 10)
    
    # Layer 1: Compute (AG already done)
    draw_block(ax, 3.2, y_compute, 2.0, block_h, COLORS["compute"], "L₁", 12)
    
    # AG₂ overlaps with L₁ compute
    draw_block(ax, 3.2, y_comm, 1.0, block_h, COLORS["allgather"], "AG₂", 10)
    
    # Layer 2: Compute (AG already done)
    draw_block(ax, 5.3, y_compute, 2.0, block_h, COLORS["compute"], "L₂", 12)
    
    # Time arrow
    ax.annotate("", xy=(8, -0.3), xytext=(0, -0.3),
                arrowprops=dict(arrowstyle="->", color=COLORS["text"], lw=1.5))
    ax.text(4, -0.6, "time", ha="center", fontsize=11, color=COLORS["text"])
    
    # Overlap annotation
    ax.annotate("", xy=(2.0, 1.1), xytext=(2.0, 1.1 - 0.15),
                arrowprops=dict(arrowstyle="-", color="#27ae60", lw=2, ls="--"))

def main():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4))
    panel_no_prefetch(ax1)
    panel_with_prefetch(ax2)
    
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
