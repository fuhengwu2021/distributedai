"""
SLURM Architecture: slurmctld, slurmd, slurmdbd and job flow.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "user": "User",
        "user_cmds": "sbatch\nsrun\nsqueue",
        "head_node": "Head Node",
        "slurmctld": "slurmctld",
        "ctld_desc": "Job Queue\nScheduler\nResource Mgr",
        "slurmdbd": "slurmdbd",
        "dbd_desc": "Accounting",
        "compute_node": "Compute Node {i}",
        "slurmd": "slurmd",
        "tasks": "Tasks",
        "gpu": "GPU{j}",
        "submit": "submit",
        "allocate_launch": "allocate &\nlaunch",
    },
    "zh": {
        "user": "用户",
        "user_cmds": "sbatch\nsrun\nsqueue",
        "head_node": "管理节点",
        "slurmctld": "slurmctld",
        "ctld_desc": "作业队列\n调度器\n资源管理",
        "slurmdbd": "slurmdbd",
        "dbd_desc": "数据库记账",
        "compute_node": "计算节点 {i}",
        "slurmd": "slurmd",
        "tasks": "任务执行",
        "gpu": "GPU{j}",
        "submit": "提交",
        "allocate_launch": "分配资源与\n拉起进程",
    }
}


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axis_off()
    ax.set_xlim(0.2, 9.6)
    ax.set_ylim(0, 6.2)

    # Colors
    head_color = "#e8f4f8"
    compute_color = "#d4edda"
    db_color = "#fff3cd"
    user_color = "#f8d7da"
    arrow_color = "#666666"

    # User/Client box (left side)
    user_box = mpatches.FancyBboxPatch((0.3, 2.5), 1.72, 1.45, boxstyle="round,pad=0.08",
                                        facecolor=user_color, edgecolor="black", linewidth=1.2)
    ax.add_patch(user_box)
    ax.text(1.2, 3.5, text["user"], ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(1.2, 3.1, text["user_cmds"], ha="center", va="center", fontsize=12, family="monospace")

    # Head Node with slurmctld
    head_box = mpatches.FancyBboxPatch((2.85, 2.0), 2.25, 2.4, boxstyle="round,pad=0.08",
                                        facecolor=head_color, edgecolor="black", linewidth=1.2)
    ax.add_patch(head_box)
    ax.text(4.0, 4.2, text["head_node"], ha="center", va="center", fontsize=11, fontweight="bold")

    # slurmctld daemon box
    ctld_box = mpatches.FancyBboxPatch((3.1, 2.3), 1.8, 1.5, boxstyle="round,pad=0.05",
                                        facecolor="#b8daff", edgecolor="black", linewidth=1)
    ax.add_patch(ctld_box)
    ax.text(4.0, 3.4, text["slurmctld"], ha="center", va="center", fontsize=13, fontweight="bold", family="monospace")
    ax.text(4.0, 2.9, text["ctld_desc"], ha="center", va="center", fontsize=12)

    # Database with slurmdbd (bottom)
    db_box = mpatches.FancyBboxPatch((3.3, 0.3), 1.4, 1.2, boxstyle="round,pad=0.05",
                                      facecolor=db_color, edgecolor="black", linewidth=1)
    ax.add_patch(db_box)
    ax.text(4.0, 1.1, text["slurmdbd"], ha="center", va="center", fontsize=12, fontweight="bold", family="monospace")
    ax.text(4.0, 0.7, text["dbd_desc"], ha="center", va="center", fontsize=12)

    # Compute Nodes (right side)
    nodes = [(4.2, text["compute_node"].format(i=0)), 
             (2.2, text["compute_node"].format(i=1)), 
             (0.2, text["compute_node"].format(i="N"))]
    for i, (y_pos, node_name) in enumerate(nodes):
        if i == 2:  # Add dots before last node
            ax.text(7.8, 1.6, "...", ha="center", va="center", fontsize=16, fontweight="bold")
        
        node_box = mpatches.FancyBboxPatch((6.0, y_pos), 3.5, 1.8, boxstyle="round,pad=0.08",
                                            facecolor=compute_color, edgecolor="black", linewidth=1.2)
        ax.add_patch(node_box)
        ax.text(7.75, y_pos + 1.55, node_name, ha="center", va="center", fontsize=13, fontweight="bold")
        
        # slurmd daemon
        slurmd_box = mpatches.FancyBboxPatch((6.3, y_pos + 0.2), 1.3, 1.0, boxstyle="round,pad=0.05",
                                              facecolor="#c3e6cb", edgecolor="black", linewidth=1)
        ax.add_patch(slurmd_box)
        ax.text(6.95, y_pos + 0.9, text["slurmd"], ha="center", va="center", fontsize=12, fontweight="bold", family="monospace")
        ax.text(6.95, y_pos + 0.5, text["tasks"], ha="center", va="center", fontsize=12)
        
        # GPU boxes
        for j in range(2):
            gpu_box = mpatches.FancyBboxPatch((7.9 + j * 0.75, y_pos + 0.3), 0.65, 0.8, boxstyle="round,pad=0.03",
                                               facecolor="#99bcff", edgecolor="black", linewidth=0.8)
            ax.add_patch(gpu_box)
            ax.text(8.22 + j * 0.75, y_pos + 0.7, text["gpu"].format(j=j), ha="center", va="center", fontsize=12)

    # Arrows
    # User -> slurmctld
    ax.annotate("", xy=(2.8, 3.25), xytext=(2.1, 3.25),
                arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.5))
    ax.text(2.45, 3.5, text["submit"], ha="center", va="center", fontsize=10, color=arrow_color)

    # slurmctld -> slurmd (multiple arrows)
    for y_offset in [4.8, 2.8]:
        ax.annotate("", xy=(5.925, y_offset), xytext=(5.2, 3.25),
                    arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.2,
                                   connectionstyle="arc3,rad=-0.01"))

    ax.text(5.4, 4.8, text["allocate_launch"], ha="center", va="center", fontsize=12, color=arrow_color)

    # slurmctld <-> slurmdbd
    ax.annotate("", xy=(4.0, 1.5), xytext=(4.0, 2.3),
                arrowprops=dict(arrowstyle="<->", color=arrow_color, lw=1.2))

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    localized_figure(draw, "slurm_architecture", LABELS, __file__)
