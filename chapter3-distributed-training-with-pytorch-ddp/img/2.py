import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "top": "torchrun / TDE\nProcess orchestration, rendezvous, fault tolerance",
        "agent": "Elastic Agent",
        "worker": "Rank {i}\nDDP: gradient sync",
    },
    "zh": {
        "top": "torchrun / TDE\n进程编排、Rendezvous 汇合与容错管理",
        "agent": "Elastic Agent（弹性代理）",
        "worker": "Rank {i}\nDDP：梯度同步",
    },
}


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, box_text, fontsize=11):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03",
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, box_text,
                ha="center", va="center", fontsize=fontsize)

    # Top layer
    box(1, 4.8, 8, 0.9, text["top"])

    # Elastic Agent
    box(3.5, 3.6, 3, 0.8, text["agent"])

    # Worker boxes
    worker_centers = np.linspace(2, 8, 4)
    worker_width = 1.5
    worker_height = 1.0
    worker_y = 1.8

    for i, cx in enumerate(worker_centers):
        box(cx - worker_width / 2, worker_y,
            worker_width, worker_height,
            text["worker"].format(i=i))

    # Vertical arrow from TDE to Agent
    ax.annotate("", xy=(5, 4.8), xytext=(5, 4.4),
                arrowprops=dict(arrowstyle="->"))

    # Vertical arrows from Agent to workers
    for cx in worker_centers:
        ax.annotate("",
            xy=(cx, worker_y + worker_height),
            xytext=(5, 3.6),
            arrowprops=dict(arrowstyle="->")
        )

    # Clean DDP ring (circular arrows)
    ring_y = worker_y - 0.3

    for i in range(3):
        ax.annotate("",
            xy=(worker_centers[i + 1], ring_y),
            xytext=(worker_centers[i], ring_y),
            arrowprops=dict(arrowstyle="->")
        )

    # Close ring with smooth arc
    ax.annotate("",
        xy=(worker_centers[0], ring_y),
        xytext=(worker_centers[-1], ring_y),
        arrowprops=dict(arrowstyle="->",
                        connectionstyle="arc3,rad=-0.4")
    )

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    localized_figure(draw, "2", LABELS, __file__)
