#!/usr/bin/env python3
"""
Per-GPU memory comparison: DDP (full replicate per GPU) vs FSDP (sharded) for a
7B parameter model in BF16 with Adam. Left: memory per GPU vs number of GPUs
(log scale) with an 80 GB capacity line. Right: stacked bar breakdown of
parameters, gradients, and optimizer state for DDP vs FSDP at N=8.
Figure captions are set in the chapter markdown; no titles in the plot.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure shared directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "ddp_label": "DDP (full model/opt per GPU)",
        "fsdp_label": "FSDP (sharded per GPU)",
        "capacity_label": "H100/A100 (80 GB)",
        "xlabel_gpus": "Number of GPUs",
        "ylabel_mem_gpu": "Memory per GPU (GB)",
        "bar_ddp": "DDP",
        "bar_fsdp": "FSDP (N=8)",
        "params": "Parameters",
        "grads": "Gradients",
        "opt_states": "Optimizer states",
        "limit_80gb": "80 GB limit",
        "ylabel_mem": "Memory (GB)",
    },
    "zh": {
        "ddp_label": "DDP（单卡全量模型/优化器）",
        "fsdp_label": "FSDP（每卡分片）",
        "capacity_label": "H100/A100 (80 GB)",
        "xlabel_gpus": "GPU 数量",
        "ylabel_mem_gpu": "单卡显存 (GB)",
        "bar_ddp": "DDP",
        "bar_fsdp": "FSDP (N=8)",
        "params": "模型参数",
        "grads": "梯度",
        "opt_states": "优化器状态",
        "limit_80gb": "80 GB 上限",
        "ylabel_mem": "显存 (GB)",
    },
}

# Data based on the text for a 7B model (BF16, Adam): 14 + 14 + 56 = 84 GB
param_mem_gb = 14
grad_mem_gb = 14
opt_state_mem_gb = 56
total_mem_gb = param_mem_gb + grad_mem_gb + opt_state_mem_gb  # 84 GB

gpu_counts = [1, 2, 4, 8, 16, 32, 64]
ddp_mem = [total_mem_gb] * len(gpu_counts)
fsdp_mem = [total_mem_gb / n for n in gpu_counts]


def draw(text: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Memory per GPU vs Number of GPUs (no title; caption in markdown)
    ax1.plot(gpu_counts, ddp_mem, "o-", label=text["ddp_label"], color="red", linewidth=2)
    ax1.plot(gpu_counts, fsdp_mem, "o-", label=text["fsdp_label"], color="green", linewidth=2)
    ax1.axhline(y=80, color="blue", linestyle="--", label=text["capacity_label"])
    ax1.set_xlabel(text["xlabel_gpus"], fontsize=12)
    ax1.set_ylabel(text["ylabel_mem_gpu"], fontsize=12)
    ax1.set_yscale("log")
    ax1.set_ylim(1, 100)
    ax1.set_yticks([2, 5, 10, 20, 40, 80])
    ax1.set_yticklabels(["2", "5", "10", "20", "40", "80"])
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=11)
    ax1.tick_params(labelsize=11)

    # Right: Breakdown at N=8 (no title; caption in markdown)
    labels = [text["bar_ddp"], text["bar_fsdp"]]
    params = np.array([param_mem_gb, param_mem_gb / 8])
    grads = np.array([grad_mem_gb, grad_mem_gb / 8])
    opts = np.array([opt_state_mem_gb, opt_state_mem_gb / 8])
    width = 0.5
    ax2.bar(labels, params, width, label=text["params"], color="#3498db")
    ax2.bar(labels, grads, width, bottom=params, label=text["grads"], color="#e74c3c")
    ax2.bar(labels, opts, width, bottom=params + grads, label=text["opt_states"], color="#f1c40f")
    ax2.axhline(y=80, color="blue", linestyle="--", label=text["limit_80gb"])
    ax2.set_ylabel(text["ylabel_mem"], fontsize=12)
    ax2.set_ylim(0, 95)
    ax2.set_yticks([0, 20, 40, 60, 80])
    ax2.legend(fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.tick_params(labelsize=11)

    plt.tight_layout(pad=0.1)
    return fig


if __name__ == '__main__':
    localized_figure(draw, "ddp_fsdp_mem", LABELS, __file__)
