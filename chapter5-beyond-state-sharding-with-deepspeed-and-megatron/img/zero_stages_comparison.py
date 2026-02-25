"""
ZeRO Stages Comparison: Memory layout across ZeRO-1, ZeRO-2, ZeRO-3.
Shows what each GPU stores at each stage.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

COLORS = {
    "params": "#3498db",
    "grads": "#e74c3c", 
    "optimizer": "#2ecc71",
    "sharded": "#9b59b6",
    "text": "#2c3e50",
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

def draw_gpu_row(ax, y, params_sharded, grads_sharded, opt_sharded, gpu_label):
    """Draw a single GPU's memory layout."""
    x = 1.0
    block_h = 0.6
    spacing = 0.1
    
    # GPU label
    ax.text(0.5, y + block_h/2, gpu_label, ha="center", va="center", 
            fontsize=13, fontweight="bold")
    
    # Parameters
    if params_sharded:
        draw_block(ax, x, y, 0.8, block_h, COLORS["params"], "P", 10, alpha=0.7)
    else:
        draw_block(ax, x, y, 2.0, block_h, COLORS["params"], "Params", 10)
    
    # Gradients
    grad_x = x + (0.9 if params_sharded else 2.1)
    if grads_sharded:
        draw_block(ax, grad_x, y, 0.8, block_h, COLORS["grads"], "G", 10, alpha=0.7)
    else:
        draw_block(ax, grad_x, y, 2.0, block_h, COLORS["grads"], "Grads", 10)
    
    # Optimizer states
    opt_x = grad_x + (0.9 if grads_sharded else 2.1)
    if opt_sharded:
        draw_block(ax, opt_x, y, 0.8, block_h, COLORS["optimizer"], "O", 10, alpha=0.7)
    else:
        draw_block(ax, opt_x, y, 2.0, block_h, COLORS["optimizer"], "Optim", 10)

def panel_ddp(ax):
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    ax.set_title("DDP (Baseline)", fontsize=14, fontweight="bold", pad=10)
    
    for i in range(2):
        draw_gpu_row(ax, 1.5 - i * 1.0, False, False, False, f"R{i}")

def panel_zero1(ax):
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    ax.set_title("ZeRO-1", fontsize=14, fontweight="bold", pad=10)
    
    for i in range(2):
        draw_gpu_row(ax, 1.5 - i * 1.0, False, False, True, f"R{i}")

def panel_zero2(ax):
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    ax.set_title("ZeRO-2", fontsize=14, fontweight="bold", pad=10)
    
    for i in range(2):
        draw_gpu_row(ax, 1.5 - i * 1.0, False, True, True, f"R{i}")

def panel_zero3(ax):
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    ax.set_title("ZeRO-3", fontsize=14, fontweight="bold", pad=10)
    
    for i in range(2):
        draw_gpu_row(ax, 1.5 - i * 1.0, True, True, True, f"R{i}")

def main():
    fig, axes = plt.subplots(1, 4, figsize=(14, 2.5))
    
    panel_ddp(axes[0])
    panel_zero1(axes[1])
    panel_zero2(axes[2])
    panel_zero3(axes[3])
    
    # Legend
    legend_y = 0.1
    fig.text(0.25, legend_y, "■ Params", color=COLORS["params"], fontsize=13, 
             ha="center", fontweight="bold")
    fig.text(0.45, legend_y, "■ Grads", color=COLORS["grads"], fontsize=13,
             ha="center", fontweight="bold")
    fig.text(0.65, legend_y, "■ Optimizer", color=COLORS["optimizer"], fontsize=13,
             ha="center", fontweight="bold")
    fig.text(0.85, legend_y, "(smaller = sharded)", color=COLORS["text"], fontsize=13,
             ha="center", style="italic")
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
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
