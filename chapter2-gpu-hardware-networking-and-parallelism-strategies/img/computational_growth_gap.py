"""
GPU Memory Capacity vs Model Memory Requirements

Mainstream single-GPU HBM vs model weight memory (BF16) per year.
GPU data from NVIDIA SKUs; models from @tbl:model-comparison plus MT-NLG (2021)
and PaLM (2022).

Red solid (top): frontier scale including approximate (~) entries.
Yellow dashed (middle): disclosed models only (no ~).
Blue solid (bottom): mainstream single-GPU HBM.
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

# Mainstream single-GPU HBM (GB) at year-end deployment reality
# 2020: A100 40GB (May 2020); 80GB announced Nov 2020, widely deployed from 2021
# 2022–2023: H100 80GB; H200 141GB mainstream from 2024; B200 192GB from 2025
gpu_years = np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026])
gpu_memory = np.array([40, 80, 80, 80, 141, 192, 192])  # 2026: B200 still mainstream

model_years = np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026])

# Disclosed representative per year (no ~)
disclosed_memory = np.array([
    350.0,    # GPT-3 175B
    1060.0,   # MT-NLG 530B
    1080.0,   # PaLM 540B
    2170.0,   # PanGu-Σ 1.085T
    472.0,    # DeepSeek-V2 236B
    1342.0,   # DeepSeek-V3 671B
    2000.0,   # Kimi K2.6 1T
])

# Frontier per year (incl. ~); largest in @tbl:model-comparison per release year
frontier_memory = np.array([
    350.0,    # GPT-3 175B
    1060.0,   # MT-NLG 530B
    1080.0,   # PaLM 540B
    3200.0,   # Gemini-1 1.6T
    3600.0,   # GPT-4V ~1.8T
    4000.0,   # GPT-5 ~2T (low bound of ~2–5T)
    20000.0,  # Claude Mythos 5 ~10T
])

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    gpu_years,
    gpu_memory,
    "b-o",
    linewidth=2.5,
    markersize=8,
    label="Mainstream Single-GPU HBM (A100/H100/H200/B200)",
    zorder=2,
)
ax.plot(
    model_years,
    disclosed_memory,
    color="gold",
    linestyle="--",
    marker="s",
    linewidth=2.5,
    markersize=8,
    label="Disclosed Representative (BF16)",
    zorder=3,
)
ax.plot(
    model_years,
    frontier_memory,
    "r-^",
    linewidth=2.5,
    markersize=8,
    label="Frontier Estimate incl. Undisclosed (~, BF16)",
    zorder=4,
)

gpu_interp = np.interp(model_years, gpu_years, gpu_memory)
gap_memory = np.maximum(disclosed_memory, frontier_memory)
gap_mask = gap_memory > gpu_interp
ax.fill_between(
    model_years,
    gpu_interp,
    gap_memory,
    where=gap_mask,
    alpha=0.25,
    color="red",
    label="Memory Gap (requires distributed training)",
    zorder=1,
)

disclosed_annotations = [
    (2020, 350, "GPT-3\n(175B)", (2020.04, 550), "left"),
    (2021, 1060, "MT-NLG\n(530B)", (2021.35, 1700), "left"),
    (2022, 1080, "PaLM\n(540B)", (2022.035, 1650), "left"),
    (2023, 2170, "PanGu-Σ\n(1.085T)", (2022.15, 3100), "left"),
    (2024, 472, "DeepSeek-V2\n(236B)", (2024.0, 680), "left"),
    (2025, 1342, "DeepSeek-V3\n(671B)", (2025.0, 2000), "left"),
    (2026, 2000, "Kimi K2.6\n(1T)", (2025.45, 3200), "left"),
]
frontier_annotations = [
    (2023, 3200, "Gemini-1\n(1.6T)", (2023.55, 5200), "left"),
    (2024, 3600, "GPT-4V\n(~1.8T)", (2024.55, 5800), "left"),
    (2025, 4000, "GPT-5\n(~2T)", (2025.55, 6500), "left"),
    (2026, 20000, "Claude Mythos 5\n(~10T)", (2026.05, 28000), "right"),
]
for x, y, label, (tx, ty), ha in disclosed_annotations:
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(tx, ty),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"),
        fontsize=8,
        ha=ha,
        color="black",
    )
for x, y, label, (tx, ty), ha in frontier_annotations:
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(tx, ty),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"),
        fontsize=8,
        ha=ha,
        color="black",
    )

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Memory (GB)", fontsize=12)
ax.grid(True, alpha=0.3, linestyle="--", zorder=0)
handles, labels = ax.get_legend_handles_labels()
order = [2, 1, 0, 3]  # frontier, disclosed, GPU, gap
ax.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    loc="upper left",
    fontsize=9,
    framealpha=0.9,
)
ax.tick_params(labelsize=11)
ax.set_yscale("log")
ax.set_xlim(2019.5, 2026.1)
ax.set_ylim(bottom=30, top=41000)

plt.tight_layout()
save_figure(__file__)
