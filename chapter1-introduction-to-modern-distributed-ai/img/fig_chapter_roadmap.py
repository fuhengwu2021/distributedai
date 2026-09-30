"""fig_chapter_roadmap.png / _zh -- Technical Roadmap for Distributed AI Systems.

Learned directly from ~/mmb/chapter1-us-mortgage-market/img/fig_chapter_roadmap.py.
One drawing implementation, one PNG per language. The strings live in LABELS;
nothing about the layout knows which edition it is drawing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path so figstyle can be imported anywhere
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
    localized_figure,
)

LABELS = {
    "en": {
        "title": "Where the book goes next: The Distributed AI Journey",
        "subtitle": "From single-device limits to planetary-scale training, inference, and serving",
        "nodes": [
            "Hardware &\nInterconnects\n(Ch 2)",
            "DDP & FSDP\nState Sharding\n(Ch 3–4)",
            "3D Parallelism &\nMegatron-DeepSpeed\n(Ch 5)",
            "Distributed Inference\n& vLLM / SGLang\n(Ch 6–7)",
            "Cluster Orchestration\nSlurm & Kubernetes\n(Ch 8–9)",
            "Profiling, Tuning\n& Frontier MoE\n(Ch 10–11)",
        ],
    },
    "zh": {
        "title": "本书接下来要走的路：分布式 AI 系统工程路线图",
        "subtitle": "从单卡硬件极限到万卡级集群训练、高性能推理与工业级生产栈",
        "nodes": [
            "硬件架构与\n高速网络互联\n(第2章)",
            "DDP 与 FSDP\n显存状态分片\n(第3–4章)",
            "3D 混合并行与\nMegatron 体系\n(第5章)",
            "分布式推理加速\nvLLM 与 SGLang\n(第6–7章)",
            "集群调度与编排\nSlurm 与 K8s 架构\n(第8–9章)",
            "全链路性能调优\n与前沿 MoE 演进\n(第10–11章)",
        ],
    },
}

PALETTE = [COMPUTE, NETWORK, MEMORY, INFERENCE, SYSTEM, PROFILING]
COL_X = (1.7, 5.0, 8.3)
ROW_Y = (0.70, 0.26)


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for idx, label in enumerate(text["nodes"]):
        x, y = COL_X[idx % 3], ROW_Y[idx // 3]
        color = PALETTE[idx]

        # Circular stage marker
        ax.scatter([x], [y], s=600, color=color, zorder=3)
        # Step number inside the circle
        ax.text(
            x,
            y,
            str(idx + 1),
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            weight="bold",
            zorder=4,
        )
        # Stage title below the circle in matching color
        ax.text(
            x,
            y - 0.08,
            label,
            ha="center",
            va="top",
            color=color,
            fontsize=10.5,
            linespacing=1.25,
            weight="bold",
            zorder=4,
        )

        # Horizontal flow arrows within each row
        if idx % 3 < 2:
            ax.annotate(
                "",
                xy=(COL_X[idx % 3 + 1] - 0.45, y),
                xytext=(x + 0.45, y),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=GREY),
                zorder=2,
            )

    # Wrap lane connecting the end of Row 1 (node 3) to the start of Row 2 (node 4)
    lane_y = (ROW_Y[0] + ROW_Y[1]) / 2 - 0.05
    ax.plot([COL_X[2], COL_X[2]], [ROW_Y[0] - 0.18, lane_y], color=GREY, lw=1.5, zorder=1)
    ax.plot([COL_X[2], COL_X[0]], [lane_y, lane_y], color=GREY, lw=1.5, zorder=1)
    ax.annotate(
        "",
        xy=(COL_X[0], ROW_Y[1] + 0.06),
        xytext=(COL_X[0], lane_y),
        arrowprops=dict(arrowstyle="->", lw=1.5, color=GREY),
        zorder=1,
    )

    # Header title and subtitle
    ax.text(
        5.0,
        0.96,
        text["title"],
        ha="center",
        va="center",
        fontsize=13,
        weight="bold",
        color=INK,
    )
    ax.text(
        5.0,
        0.90,
        text["subtitle"],
        ha="center",
        va="center",
        fontsize=9.5,
        color=GREY,
    )

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    localized_figure(draw, "fig_chapter_roadmap", LABELS, __file__)
