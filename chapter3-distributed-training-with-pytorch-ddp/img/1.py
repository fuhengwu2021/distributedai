import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "title": "2-Node Heterogeneous GPU DDP Architecture",
        "node0": "Node 0 (Flower)",
        "rank0": "Rank 0",
        "gpu0": "RTX 4090\n(sm_89)",
        "node1": "Node 1 (Potato)",
        "rank1": "Rank 1",
        "gpu1": "RTX 5070\n(sm_120)",
        "nccl": "NCCL AllReduce",
        "output": "Output: Rank 0 & Rank 1 print all_reduce_sum=1.0",
    },
    "zh": {
        "title": "双节点异构 GPU DDP 架构",
        "node0": "Node 0 (Flower)",
        "rank0": "Rank 0",
        "gpu0": "RTX 4090\n(sm_89)",
        "node1": "Node 1 (Potato)",
        "rank1": "Rank 1",
        "gpu1": "RTX 5070\n(sm_120)",
        "nccl": "NCCL AllReduce",
        "output": "输出结果：Rank 0 与 Rank 1 打印 all_reduce_sum=1.0",
    },
}


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))

    # ---- Node 0 ----
    node0 = patches.Rectangle((0.1, 0.3), 0.3, 0.4, fill=False)
    ax.add_patch(node0)
    ax.text(0.25, 0.65, text["node0"], ha='center', fontsize=12)
    ax.text(0.25, 0.53, text["rank0"], ha='center')

    gpu0 = patches.Rectangle((0.17, 0.4), 0.16, 0.12, fill=False)
    ax.add_patch(gpu0)
    ax.text(0.25, 0.42, text["gpu0"], ha='center')

    # ---- Node 1 ----
    node1 = patches.Rectangle((0.6, 0.3), 0.3, 0.4, fill=False)
    ax.add_patch(node1)
    ax.text(0.75, 0.65, text["node1"], ha='center', fontsize=12)
    ax.text(0.75, 0.53, text["rank1"], ha='center')

    gpu1 = patches.Rectangle((0.67, 0.4), 0.16, 0.12, fill=False)
    ax.add_patch(gpu1)
    ax.text(0.75, 0.42, text["gpu1"], ha='center')

    # ---- Communication Arrow ----
    ax.annotate(
        text["nccl"],
        xy=(0.4, 0.5),
        xytext=(0.6, 0.5),
        arrowprops=dict(arrowstyle="<->"),
        ha='center',
        va='center'
    )

    # ---- Output ----
    ax.text(0.5, 0.15, text["output"], ha='center')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.title(text["title"])
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    localized_figure(draw, "1", LABELS, __file__)