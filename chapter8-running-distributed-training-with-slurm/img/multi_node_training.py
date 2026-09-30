"""
Multi-node distributed training with SLURM: showing torchrun/srun launching
processes across nodes with NCCL communication.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "slurm": "SLURM",
        "launcher": "srun / torchrun",
        "node0": "Node 0 (MASTER_ADDR)",
        "node1": "Node 1",
        "gpu": "GPU {gpu_idx}",
        "rank": "RANK={rank}",
        "local_rank": "LOCAL_RANK={local_rank}",
        "script": "train.py",
        "process": "Process {rank}",
        "nccl": "NCCL",
        "allreduce": "AllReduce",
        "world_size": "WORLD_SIZE = 4 (2 nodes × 2 GPUs)",
    },
    "zh": {
        "slurm": "SLURM",
        "launcher": "srun / torchrun",
        "node0": "节点 0 (MASTER_ADDR)",
        "node1": "节点 1",
        "gpu": "GPU {gpu_idx}",
        "rank": "RANK={rank}",
        "local_rank": "LOCAL_RANK={local_rank}",
        "script": "train.py",
        "process": "进程 {rank}",
        "nccl": "NCCL",
        "allreduce": "AllReduce",
        "world_size": "WORLD_SIZE = 4 (2 节点 × 2 GPU)",
    }
}


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.)

    # Colors
    node_color = "#e8f4f8"
    gpu_color = "#99bcff"
    process_color = "#d4edda"
    nccl_color = "#ffc107"
    slurm_color = "#f8d7da"

    # SLURM controller at top
    slurm_box = mpatches.FancyBboxPatch((3.5, 5.0), 3.0, 0.9, boxstyle="round,pad=0.08",
                                         facecolor=slurm_color, edgecolor="black", linewidth=1.5)
    ax.add_patch(slurm_box)
    ax.text(5.0, 5.6, text["slurm"], ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(5.0, 5.2, text["launcher"], ha="center", va="center", fontsize=13, family="monospace")

    # Two nodes (moved apart: node0 to left, node1 to right)
    nodes = [(0.1, text["node0"]), (5.4, text["node1"])]
    for node_idx, (node_x, node_name) in enumerate(nodes):
        # Node box
        node_box = mpatches.FancyBboxPatch((node_x, 0.5), 4.5, 4.0, boxstyle="round,pad=0.1",
                                            facecolor=node_color, edgecolor="grey", linewidth=1.5)
        ax.add_patch(node_box)
        ax.text(node_x + 2.25, 4.25, node_name, ha="center", va="center", 
                fontsize=13, fontweight="bold")
        
        # Two GPUs per node
        for gpu_idx in range(2):
            gpu_x = node_x + 0.4 + gpu_idx * 2.1
            
            # GPU box
            gpu_box = mpatches.FancyBboxPatch((gpu_x, 0.8), 1.8, 3.0, boxstyle="round,pad=0.05",
                                               facecolor=gpu_color, edgecolor="black", linewidth=1)
            ax.add_patch(gpu_box)
            
            # Global rank calculation
            global_rank = node_idx * 2 + gpu_idx
            
            # Process box inside GPU
            proc_box = mpatches.FancyBboxPatch((gpu_x + 0.15, 1.5), 1.5, 1.8, boxstyle="round,pad=0.05",
                                                facecolor=process_color, edgecolor="black", linewidth=0.8)
            ax.add_patch(proc_box)
            
            # Labels
            ax.text(gpu_x + 0.9, 3.55, text["gpu"].format(gpu_idx=gpu_idx), ha="center", va="center", fontsize=13)
            ax.text(gpu_x + 0.9, 2.9, text["rank"].format(rank=global_rank), ha="center", va="center", 
                    fontsize=13, fontweight="bold")
            ax.text(gpu_x + 0.9, 2.5, text["local_rank"].format(local_rank=gpu_idx), ha="center", va="center", 
                    fontsize=12, color="#0066cc")
            ax.text(gpu_x + 0.9, 2.1, text["script"], ha="center", va="center", 
                    fontsize=12, family="monospace")
            ax.text(gpu_x + 0.9, 1.0, text["process"].format(rank=global_rank), ha="center", va="center", fontsize=12)

    # SLURM arrows to nodes
    ax.annotate("", xy=(2.35, 4.5), xytext=(4.2, 5.0),
                arrowprops=dict(arrowstyle="->", color="#666666", lw=1.5,
                               connectionstyle="arc3,rad=0.2"))
    ax.annotate("", xy=(7.65, 4.5), xytext=(5.8, 5.0),
                arrowprops=dict(arrowstyle="->", color="#666666", lw=1.5,
                               connectionstyle="arc3,rad=-0.2"))

    # NCCL communication (horizontal double arrow between nodes)
    ax.annotate("", xy=(5.4, 2.5), xytext=(4.6, 2.5),
                arrowprops=dict(arrowstyle="<->", color=nccl_color, lw=3))
    ax.text(5.0, 2.9, text["nccl"], ha="center", va="center", fontsize=13, fontweight="bold",
            color="#856404")
    ax.text(5.0, 2.15, text["allreduce"], ha="center", va="center", fontsize=12, color="red", fontweight="bold")

    # WORLD_SIZE annotation
    ax.text(5.0, 0.2, text["world_size"], ha="center", va="center", 
            fontsize=13, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd", edgecolor="gray"))

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    localized_figure(draw, "multi_node_training", LABELS, __file__)
