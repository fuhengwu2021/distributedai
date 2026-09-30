import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "dataloader": "Dataloader",
        "gpu": "GPU{idx}",
        "model": "Model",
        "pass": "Forward/\nBackward pass",
        "sync_title": "Synchronize",
        "sync_box": "Synchronize\ngradients",
        "update": "Update\nModel",
    },
    "zh": {
        "dataloader": "数据加载器",
        "gpu": "GPU{idx}",
        "model": "模型",
        "pass": "前向传播/\n反向求导",
        "sync_title": "梯度同步",
        "sync_box": "同步\n梯度",
        "update": "更新\n模型",
    },
}


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis('off')

    # Helper function to draw labeled rectangles
    def draw_box(x, y, w, h, box_text, color, text_color='black', fontsize=14):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                                      linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, box_text, color=text_color, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', wrap=True)

    # 1. Dataloader (The Source)
    draw_box(2, 25, 12, 10, text["dataloader"], "#ffdb99")

    # 2. GPU Rows (Sharding and Processing)
    shard_colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#f9ca24"]  # Red, Teal, Blue, Yellow
    y_offsets = [45, 32, 19, 6]

    for i, y in enumerate(y_offsets):
        # Data Shard
        draw_box(18, y + 2, 6, 6, "", "white")
        ax.add_patch(patches.Rectangle((18 + (i * 1.5), y + 2), 1.5, 6, color=shard_colors[i]))

        # GPU / Model
        draw_box(28, y - 0.4, 10, 10, "", "#f2f2f2")
        ax.text(33, y - 0.4 + 3, text["gpu"].format(idx=3 - i), color='black', ha='center', va='center',
                fontsize=14, fontweight='bold')
        ax.add_patch(plt.Circle((33, y + 8), 2.8, color="#b19cd9", ec='black'))
        ax.text(33, y + 8, text["model"], fontsize=12, ha='center', va='center', fontweight='bold', color='black')

        # Forward/Backward Pass
        draw_box(42, y, 15, 10, text["pass"], "#99bcff", text_color="black")

        # Connections: Dataloader -> Shard -> GPU -> Pass
        ax.annotate('', xy=(18, y + 5), xytext=(14, 30), arrowprops=dict(arrowstyle='->', lw=0.5, color='gray'))
        ax.annotate('', xy=(28, y + 5), xytext=(24, y + 5), arrowprops=dict(arrowstyle='->', lw=0.5))
        ax.annotate('', xy=(42, y + 5), xytext=(38, y + 5), arrowprops=dict(arrowstyle='->', lw=0.5))

    # 3. Synchronize Gradients (The Bottleneck/Communication)
    ax.text(70, 55, text["sync_title"], fontsize=18, fontweight='bold', ha='center', color='black')
    draw_box(65, 23, 14, 14, text["sync_box"], "#b19cd9", text_color="black", fontsize=14)

    # Connections: Passes -> Synchronize
    for y in y_offsets:
        ax.annotate('', xy=(65, 30), xytext=(57, y + 5), arrowprops=dict(arrowstyle='->', lw=0.8, color='gray'))

    # 4. Update Model (Final Step)
    for y in y_offsets:
        draw_box(85, y, 12, 10, text["update"], "#e69183", text_color="black")
        # Connection: Synchronize -> Update
        ax.annotate('', xy=(85, y + 5), xytext=(79, 30), arrowprops=dict(arrowstyle='->', lw=0.8, color='gray'))

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    localized_figure(draw, "ddp_workflow", LABELS, __file__)