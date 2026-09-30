"""
SLURM job lifecycle: PENDING -> RUNNING -> COMPLETING -> COMPLETED
with possible states like FAILED, CANCELLED, TIMEOUT.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "title": "SLURM Job State Lifecycle",
        "pending": "PENDING",
        "pending_desc": "Waiting for\nresources",
        "running": "RUNNING",
        "running_desc": "Executing\non nodes",
        "completing": "COMPLETING",
        "completing_desc": "Cleaning up\nprocesses",
        "completed": "COMPLETED",
        "completed_desc": "Finished\nsuccessfully",
        "failed": "FAILED",
        "failed_desc": "Error occurred",
        "cancelled": "CANCELLED",
        "cancelled_desc": "User cancelled",
        "timeout": "TIMEOUT",
        "timeout_desc": "Time limit hit",
        "submit_cmd": "sbatch/srun",
    },
    "zh": {
        "title": "SLURM 作业状态生命周期",
        "pending": "PENDING",
        "pending_desc": "等待资源分配",
        "running": "RUNNING",
        "running_desc": "在计算节点执行",
        "completing": "COMPLETING",
        "completing_desc": "清理释放进程",
        "completed": "COMPLETED",
        "completed_desc": "作业成功完成",
        "failed": "FAILED",
        "failed_desc": "运行发生错误",
        "cancelled": "CANCELLED",
        "cancelled_desc": "用户手动取消",
        "timeout": "TIMEOUT",
        "timeout_desc": "达到时间限制",
        "submit_cmd": "sbatch/srun",
    }
}


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_axis_off()
    ax.set_xlim(0.2, 9.1)
    ax.set_ylim(0.2, 4.5)

    # Colors
    pending_color = "#fff3cd"
    running_color = "#d4edda"
    completing_color = "#cce5ff"
    completed_color = "#c3e6cb"
    failed_color = "#f8d7da"
    arrow_color = "#666666"

    # Title
    ax.text(5.0, 4.2, text["title"], ha="center", va="center", 
            fontsize=13, fontweight="bold")

    # Main flow states (horizontal)
    states = [
        (0.5, 2.2, text["pending"], pending_color, text["pending_desc"]),
        (2.7, 2.2, text["running"], running_color, text["running_desc"]),
        (4.9, 2.2, text["completing"], completing_color, text["completing_desc"]),
        (7.1, 2.2, text["completed"], completed_color, text["completed_desc"]),
    ]

    for x, y, label, color, desc in states:
        box = mpatches.FancyBboxPatch((x, y), 1.8, 1.4, boxstyle="round,pad=0.08",
                                       facecolor=color, edgecolor="black", linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + 0.9, y + 1.0, label, ha="center", va="center", 
                fontsize=13, fontweight="bold")
        ax.text(x + 0.9, y + 0.4, desc, ha="center", va="center", fontsize=12)

    # Arrows between main states
    for i in range(3):
        x_start = states[i][0] + 1.8
        x_end = states[i+1][0]
        y = 2.9
        ax.annotate("", xy=(x_end, y), xytext=(x_start, y),
                    arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2))

    # Alternative end states (bottom)
    alt_states = [
        (4.9, 0.4, text["failed"], failed_color, text["failed_desc"]),
        (7.1, 0.4, text["cancelled"], "#e2e3e5", text["cancelled_desc"]),
        (2.7, 0.4, text["timeout"], "#ffeeba", text["timeout_desc"]),
    ]

    for x, y, label, color, desc in alt_states:
        box = mpatches.FancyBboxPatch((x, y), 1.8, 1.0, boxstyle="round,pad=0.08",
                                       facecolor=color, edgecolor="black", linewidth=1)
        ax.add_patch(box)
        ax.text(x + 0.9, y + 0.65, label, ha="center", va="center", 
                fontsize=12, fontweight="bold")
        ax.text(x + 0.9, y + 0.25, desc, ha="center", va="center", fontsize=13)

    # Arrows to alternative states
    # RUNNING -> FAILED
    ax.annotate("", xy=(5.8, 1.4), xytext=(5.8, 2.2),
                arrowprops=dict(arrowstyle="-|>", color="#dc3545", lw=1.5))

    # RUNNING -> CANCELLED
    ax.annotate("", xy=(8.0, 1.4), xytext=(4.5, 2.5),
                arrowprops=dict(arrowstyle="-|>", color="#6c757d", lw=1.2,
                               connectionstyle="arc3,rad=-0.013"))

    # RUNNING -> TIMEOUT
    ax.annotate("", xy=(3.6, 1.4), xytext=(3.6, 2.2),
                arrowprops=dict(arrowstyle="-|>", color="#856404", lw=1.2))

    # sbatch annotation
    ax.text(0.3, 4.4, text["submit_cmd"], ha="left", va="center", fontsize=12, 
            family="monospace", style="italic")
    ax.annotate("", xy=(1.4, 3.7), xytext=(1.4, 4.4),
                arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=1.5))

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    localized_figure(draw, "job_lifecycle", LABELS, __file__)
