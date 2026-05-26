"""
GPU Memory Capacity vs Model Memory Requirements

This plot shows actual GPU memory capacity over time versus actual model memory
requirements (in BF16/FP16), using real data from NVIDIA GPU specifications and
published model sizes from @tbl:model-comparison (disclosed counts only).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "shared",
    ),
)
from math4ai import configure_math_fonts, save_figure

configure_math_fonts()

# GPU memory capacity (GB) — NVIDIA specifications
gpu_years = np.array([2020, 2022, 2023, 2024])
gpu_memory = np.array([80, 80, 141, 192])  # A100, H100, H200, B200

# Model weight memory (GB, BF16 = 2 bytes/param) — disclosed milestones from chapter 1
model_years = np.array([2020, 2023, 2024, 2025, 2026])
model_memory = np.array([
    350.0,   # GPT-3 175B
    140.0,   # LLaMA-2 70B
    472.0,   # DeepSeek-V2 236B
    1342.0,  # DeepSeek-V3 671B
    3200.0,  # DeepSeek-V4-Pro 1.6T
])

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    gpu_years,
    gpu_memory,
    "b-o",
    linewidth=2.5,
    markersize=8,
    label="Single GPU Memory Capacity (A100/H100/H200/B200)",
    zorder=3,
)
ax.plot(
    model_years,
    model_memory,
    "r-s",
    linewidth=2.5,
    markersize=8,
    label="Model Memory Requirements (BF16, disclosed milestones)",
    zorder=3,
)

gpu_interp = np.interp(model_years, gpu_years, gpu_memory)
gap_mask = model_memory > gpu_interp
ax.fill_between(
    model_years,
    gpu_interp,
    model_memory,
    where=gap_mask,
    alpha=0.3,
    color="red",
    label="Memory Gap (requires distributed training)",
    zorder=1,
)

ax.annotate(
    "GPT-3\n(175B)",
    xy=(2020, 350),
    xytext=(2020.4, 550),
    arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"),
    fontsize=9,
    ha="left",
)
ax.annotate(
    "LLaMA-2\n(70B)",
    xy=(2023, 140),
    xytext=(2023.3, 280),
    arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"),
    fontsize=9,
    ha="left",
)
ax.annotate(
    "DeepSeek-V3\n(671B)",
    xy=(2025, 1342),
    xytext=(2024.5, 2200),
    arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"),
    fontsize=9,
    ha="right",
)
ax.annotate(
    "DeepSeek-V4-Pro\n(1.6T)",
    xy=(2026, 3200),
    xytext=(2025.4, 4500),
    arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"),
    fontsize=9,
    ha="right",
)

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Memory (GB)", fontsize=12)
ax.grid(True, alpha=0.3, linestyle="--", zorder=0)
ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
ax.tick_params(labelsize=11)
ax.set_yscale("log")
ax.set_xlim(2019.5, 2026.5)
ax.set_ylim(bottom=50, top=8000)

plt.tight_layout()
save_figure(__file__)
