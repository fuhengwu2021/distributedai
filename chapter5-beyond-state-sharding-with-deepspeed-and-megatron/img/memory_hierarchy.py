"""
Memory Hierarchy for ZeRO-Infinity: GPU → CPU → NVMe.
Shows bandwidth and capacity at each level.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from math4ai import save_figure

COLORS = {
    "gpu": "#e74c3c",
    "cpu": "#3498db",
    "nvme": "#2ecc71",
    "arrow": "#7f8c8d",
    "text": "#2c3e50",
}

def draw_tier(ax, x, y, w, h, color, label, capacity, bandwidth):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03",
        linewidth=2, edgecolor="black", facecolor=color
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h*0.65, label, ha="center", va="center", 
            fontsize=14, fontweight="bold", color="white")
    ax.text(x + w/2, y + h*0.35, capacity, ha="center", va="center", 
            fontsize=11, color="white")

def main():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 5)
    ax.axis("off")
    ax.set_title("ZeRO-Infinity Memory Hierarchy", fontsize=16, fontweight="bold", pad=15)
    
    # Draw tiers (top to bottom: GPU → CPU → NVMe)
    tier_w = 3.0
    tier_h = 1.2
    center_x = 3.5
    
    # GPU tier
    draw_tier(ax, center_x, 3.5, tier_w, tier_h, COLORS["gpu"], 
              "GPU HBM", "80 GB", "1.5 TB/s")
    
    # CPU tier
    draw_tier(ax, center_x, 1.8, tier_w, tier_h, COLORS["cpu"],
              "CPU RAM", "512 GB", "100 GB/s")
    
    # NVMe tier
    draw_tier(ax, center_x, 0.1, tier_w, tier_h, COLORS["nvme"],
              "NVMe SSD", "4+ TB", "7 GB/s")
    
    # Arrows with bandwidth labels
    arrow_x = center_x + tier_w + 0.3
    
    # GPU ↔ CPU
    ax.annotate("", xy=(arrow_x, 3.5), xytext=(arrow_x, 3.0),
                arrowprops=dict(arrowstyle="<->", color=COLORS["arrow"], lw=2))
    ax.text(arrow_x + 0.3, 3.25, "PCIe\n32 GB/s", fontsize=10, va="center", color=COLORS["text"])
    
    # CPU ↔ NVMe
    ax.annotate("", xy=(arrow_x, 1.8), xytext=(arrow_x, 1.3),
                arrowprops=dict(arrowstyle="<->", color=COLORS["arrow"], lw=2))
    ax.text(arrow_x + 0.3, 1.55, "NVMe\n7 GB/s", fontsize=10, va="center", color=COLORS["text"])
    
    # What's stored at each level (right side)
    info_x = 8.0
    ax.text(info_x, 4.0, "Active params\n+ activations", fontsize=10, 
            ha="center", va="center", color=COLORS["gpu"], fontweight="bold")
    ax.text(info_x, 2.3, "Optimizer states\n+ param buffer", fontsize=10,
            ha="center", va="center", color=COLORS["cpu"], fontweight="bold")
    ax.text(info_x, 0.6, "Cold params\n+ checkpoints", fontsize=10,
            ha="center", va="center", color=COLORS["nvme"], fontweight="bold")
    
    plt.tight_layout()
    
    save_figure(__file__)

if __name__ == "__main__":
    main()
