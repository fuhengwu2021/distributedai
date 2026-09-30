"""fig_distai_stack.png / _zh -- The Distributed AI Stack Architecture.

Learned from ~/mmb bilingual pattern:
One drawing implementation, localized bilingual outputs for English and Chinese.
"""

from __future__ import annotations

import sys
from pathlib import Path
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from figstyle import (
    COMPUTE,
    NETWORK,
    MEMORY,
    INFERENCE,
    SYSTEM,
    PROFILING,
    GREY,
    INK,
    arrow,
    box,
    localized_figure,
)

LABELS = {
    "en": {
        "title": "The Distributed AI Stack",
        "layers": [
            ("Framework Layer", "PyTorch, Megatron-LM, DeepSpeed, vLLM", COMPUTE),
            ("Messaging Layer", "ProcessGroup, RPC, Rendezvous, Store", SYSTEM),
            ("Collective Operations Layer", "AllReduce, AllGather, ReduceScatter, AlltoAll", MEMORY),
            ("Data Transfer Layer", "NCCL, Gloo, RCCL, OneCCL", NETWORK),
            ("Topology Layer", "NVLink Mesh, NVSwitch, Ring / Tree Logical Graphs", INFERENCE),
            ("Link Layer", "InfiniBand Verbs, RoCE v2, RDMA, TCP/IP", PROFILING),
            ("Physical Layer", "GPU Chips (H100/B200), PCIe Gen5, Optical Transceivers", INK),
        ],
    },
    "zh": {
        "title": "分布式 AI 核心技术分层栈",
        "layers": [
            ("应用框架层 (Framework Layer)", "PyTorch, Megatron-LM, DeepSpeed, vLLM", COMPUTE),
            ("进程与控制层 (Messaging Layer)", "ProcessGroup, RPC, Rendezvous, 键值存储", SYSTEM),
            ("集合通信原语层 (Collective Ops)", "AllReduce, AllGather, ReduceScatter, AlltoAll", MEMORY),
            ("底层传输驱动层 (Data Transfer)", "NCCL, Gloo, RCCL, OneCCL 高性能库", NETWORK),
            ("物理拓扑组织层 (Topology Layer)", "NVLink 网格, NVSwitch, 逻辑环/树通信图", INFERENCE),
            ("链路协议层 (Link Layer)", "InfiniBand Verbs, RoCE v2, RDMA, TCP/IP", PROFILING),
            ("物理硬件层 (Physical Layer)", "GPU 计算芯片 (H100/B200), PCIe Gen5, 高速光模块", INK),
        ],
    },
}


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.6, 7.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    layers = text["layers"]
    n_layers = len(layers)
    box_w = 8.6
    box_h = 0.72
    box_x = 0.7
    spacing = 1.05
    start_y = 0.45

    # Draw from bottom (Physical) to top (Framework)
    # layers is defined top-to-bottom, so reverse for ascending Y
    for i, (title, desc, color) in enumerate(reversed(layers)):
        y = start_y + i * spacing
        # Main layer box
        box(
            ax,
            (box_x, y),
            box_w,
            box_h,
            f"{title}\n{desc}",
            color=color,
            text_color="white",
            fontsize=10.5,
            pad=0.015,
            rounding=0.05,
            lw=1.2,
            edgecolor=INK,
            linespacing=1.2,
        )

        # Draw upward flow arrow between layers
        if i < n_layers - 1:
            arrow(
                ax,
                (5.0, y + box_h + 0.02),
                (5.0, y + spacing - 0.02),
                color=GREY,
                arrowstyle="->",
                mutation_scale=13,
                lw=1.6,
                zorder=2,
            )

    # Title at the top
    ax.text(
        5.0,
        8.05,
        text["title"],
        ha="center",
        va="center",
        fontsize=13.5,
        weight="bold",
        color=INK,
    )

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    localized_figure(draw, "fig_distai_stack", LABELS, __file__)
