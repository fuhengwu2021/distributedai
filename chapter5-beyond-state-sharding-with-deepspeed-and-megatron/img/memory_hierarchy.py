"""
Memory Hierarchy for ZeRO-Infinity: GPU → CPU → NVMe.
Shows bandwidth and capacity at each level.
"""
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure shared directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

COLORS = {
    "gpu": "#e74c3c",
    "cpu": "#3498db",
    "nvme": "#2ecc71",
    "arrow": "#7f8c8d",
    "text": "#2c3e50",
}

LABELS = {
    "en": {
        "title": "ZeRO-Infinity Memory Hierarchy",
        "gpu_label": "GPU HBM",
        "cpu_label": "CPU RAM",
        "nvme_label": "NVMe SSD",
        "pcie": "PCIe\n32 GB/s",
        "nvme_bw": "NVMe\n7 GB/s",
        "gpu_info": "Active params\n+ activations",
        "cpu_info": "Optimizer states\n+ param buffer",
        "nvme_info": "Cold params\n+ checkpoints",
    },
    "zh": {
        "title": "ZeRO-Infinity 异构存储分层架构",
        "gpu_label": "GPU 显存 (HBM)",
        "cpu_label": "CPU 内存 (RAM)",
        "nvme_label": "NVMe 固态硬盘 (SSD)",
        "pcie": "PCIe\n32 GB/s",
        "nvme_bw": "NVMe\n7 GB/s",
        "gpu_info": "活跃参数\n+ 激活值 (Activations)",
        "cpu_info": "优化器状态\n+ 参数缓冲区 (Buffer)",
        "nvme_info": "冷参数分片\n+ 检查点 (Checkpoints)",
    },
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

def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 5)
    ax.axis("off")
    ax.set_title(text["title"], fontsize=16, fontweight="bold", pad=15)
    
    # Draw tiers (top to bottom: GPU → CPU → NVMe)
    tier_w = 3.0
    tier_h = 1.2
    center_x = 3.5
    
    # GPU tier
    draw_tier(ax, center_x, 3.5, tier_w, tier_h, COLORS["gpu"], 
              text["gpu_label"], "80 GB", "1.5 TB/s")
    
    # CPU tier
    draw_tier(ax, center_x, 1.8, tier_w, tier_h, COLORS["cpu"],
              text["cpu_label"], "512 GB", "100 GB/s")
    
    # NVMe tier
    draw_tier(ax, center_x, 0.1, tier_w, tier_h, COLORS["nvme"],
              text["nvme_label"], "4+ TB", "7 GB/s")
    
    # Arrows with bandwidth labels
    arrow_x = center_x + tier_w + 0.3
    
    # GPU ↔ CPU
    ax.annotate("", xy=(arrow_x, 3.5), xytext=(arrow_x, 3.0),
                arrowprops=dict(arrowstyle="<->", color=COLORS["arrow"], lw=2))
    ax.text(arrow_x + 0.3, 3.25, text["pcie"], fontsize=10, va="center", color=COLORS["text"])
    
    # CPU ↔ NVMe
    ax.annotate("", xy=(arrow_x, 1.8), xytext=(arrow_x, 1.3),
                arrowprops=dict(arrowstyle="<->", color=COLORS["arrow"], lw=2))
    ax.text(arrow_x + 0.3, 1.55, text["nvme_bw"], fontsize=10, va="center", color=COLORS["text"])
    
    # What's stored at each level (right side)
    info_x = 8.0
    ax.text(info_x, 4.0, text["gpu_info"], fontsize=10, 
            ha="center", va="center", color=COLORS["gpu"], fontweight="bold")
    ax.text(info_x, 2.3, text["cpu_info"], fontsize=10,
            ha="center", va="center", color=COLORS["cpu"], fontweight="bold")
    ax.text(info_x, 0.6, text["nvme_info"], fontsize=10,
            ha="center", va="center", color=COLORS["nvme"], fontweight="bold")
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    localized_figure(draw, "memory_hierarchy", LABELS, __file__, pad_inches=0.08)
