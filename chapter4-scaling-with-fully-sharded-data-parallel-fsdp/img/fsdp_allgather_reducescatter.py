import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Colors: more professional palette
COLORS = {
    "ranks": ["#3498db", "#e74c3c", "#2ecc71", "#f1c40f"],
    "arrow": "#7f8c8d",
    "text": "#2c3e50"
}
N = 4
BLOCK_W = 0.8
BLOCK_H = 0.5

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

def panel_allgather(ax):
    ax.set_xlim(0.4, 5.4)
    ax.set_ylim(-.5, 4.5)
    ax.axis("off")
    ax.set_title("Forward Pass: All-Gather", fontsize=14, fontweight="bold", pad=2)

    # 1. Before: Sharded (1/N Memory)
    ax.text(2.75, 4.0, "1. Parameters are Sharded ($1/N$)", ha="center", fontsize=13, style="italic")
    for r in range(N):
        x_pos = 0.5 + r * 1.3
        ax.text(x_pos + BLOCK_W/2, 3.5, f"Rank {r}", ha="center", fontsize=13)
        draw_block(ax, x_pos, 2.8, COLORS["ranks"][r], f"P{r}")

    # Arrow
    ax.annotate("Collectives: All-Gather", xy=(2.75, 1.5), xytext=(2.75, 2.6),
                arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=2),
                ha="center", fontsize=13, color=COLORS["arrow"])

    # 2. After: Gathered (Full Model State)
    ax.text(2.75, 1.2, "2. Parameters Reconstructed for Forward", ha="center", fontsize=13, style="italic")
    for r in range(N):
        x_base = 0.5 + r * 1.3
        ax.text(x_base + BLOCK_W/2, 0.7, f"Rank {r}", ha="center", fontsize=13)
        # Draw the "full" parameter as a horizontal combined block
        for i in range(N):
            mini_w = BLOCK_W / N
            rect = patches.Rectangle((x_base + i*mini_w, 0.0), mini_w, BLOCK_H, 
                                     facecolor=COLORS["ranks"][i], edgecolor="black", linewidth=0.5)
            ax.add_patch(rect)
        ax.text(x_base + BLOCK_W/2, 0.0 - 0.3, "Full $P$", ha="center", fontsize=12)

def panel_reducescatter(ax):
    ax.set_xlim(0.4, 5.4)
    ax.set_ylim(-0.5, 4.5)
    ax.axis("off")
    ax.set_title("Backward Pass: Reduce-Scatter", fontsize=14, fontweight="bold", pad=2)

    # 1. Before: Full Gradients computed locally
    ax.text(2.75, 4.0, "1. Gradients Computed for Full Layer", ha="center", fontsize=13, style="italic")
    for r in range(N):
        x_pos = 0.5 + r * 1.3
        ax.text(x_pos + BLOCK_W/2, 3.5, f"Rank {r}", ha="center", fontsize=13)
        draw_block(ax, x_pos, 2.8, COLORS["ranks"][r], r"$\nabla$Full", hatch="///")

    # Arrow
    ax.annotate("Collectives: Reduce-Scatter", xy=(2.75, 1.5), xytext=(2.75, 2.6),
                arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=2),
                ha="center", fontsize=13, color=COLORS["arrow"])

    # 2. After: Sharded Gradients (Averaged/Summed)
    ax.text(2.75, 1.2, "2. Gradients Reduced & Sharded ($1/N$)", ha="center", fontsize=13, style="italic")
    for r in range(N):
        x_pos = 0.5 + r * 1.3
        ax.text(x_pos + BLOCK_W/2, 0.7, f"Rank {r}", ha="center", fontsize=13)
        draw_block(ax, x_pos, 0.0, COLORS["ranks"][r], f"$\sum G_{r}$", hatch="...")

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    panel_allgather(ax1)
    panel_reducescatter(ax2)
    
    plt.tight_layout(pad=0.5)
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