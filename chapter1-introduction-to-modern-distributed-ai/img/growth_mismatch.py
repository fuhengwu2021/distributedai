"""
Visualize the mismatch between exponential model growth and linear GPU growth.
Two subplots: (1) Model parameters over time (exponential), (2) GPU memory over time (linear).
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from math4ai import save_figure

# Model data: [Year, Parameters (in billions), Model Name]
model_data = [
    [2020, 175, 'GPT-3'],
    [2022, 540, 'PaLM'],
    [2023.25, 22, 'ViT-22B'],
    [2023.54, 700, 'LLaMA-2'],
    [2023.93, 1600, 'Gemini-1'],
    [2024.33, 1800, 'GPT-4o'],
    [2024.98, 671, 'DeepSeek-V3'],
    [2025.59, 4000, 'GPT-5*'],
    [2025.88, 2500, 'Gemini-3*'],
    # 2026 (from chapter model comparison table, disclosed counts)
    [2026.0, 1000, 'Kimi K2.6'],
    [2026.15, 1600, 'DeepSeek-V4-Pro'],
    [2026.3, 1500, 'Grok V9 Medium'],
]

# GPU memory data: [Year, Memory (GB), GPU Name]
gpu_data = [
    [2016, 16, 'P100'],
    [2017, 16, 'V100'],
    [2020, 40, 'A100-40'],
    [2020, 80, 'A100-80'],
    [2022, 80, 'H100'],
    [2024, 141, 'H200'],
    [2025, 192, 'B200'],
]


def create_growth_mismatch_plot():
    """Create a two-subplot figure showing exponential vs linear growth."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    years_models = [row[0] for row in model_data]
    params_b = [row[1] for row in model_data]
    model_names = [row[2] for row in model_data]

    ax1.scatter(years_models, params_b, s=120, alpha=0.7, color='#d62728',
                edgecolors='black', linewidths=1.2, zorder=3)

    for i, (year, param, name) in enumerate(zip(years_models, params_b, model_names)):
        offset_x = 10 if i % 3 == 0 else (-10 if i % 3 == 1 else 0)
        offset_y = 10 if i % 4 == 0 else (-10 if i % 4 == 1 else 5 if i % 4 == 2 else -5)
        ha_align = 'left' if offset_x > 0 else ('right' if offset_x < 0 else 'center')
        va_align = 'bottom' if offset_y > 0 else ('top' if offset_y < 0 else 'center')
        ax1.annotate(name, (year, param),
                     xytext=(offset_x, offset_y), textcoords='offset points',
                     fontsize=8, ha=ha_align, va=va_align,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               alpha=0.9, edgecolor='#d62728', linewidth=1),
                     zorder=4)

    log_params = np.log(params_b)
    coeffs = np.polyfit(years_models, log_params, 1)
    x_fit = np.linspace(2020, 2026, 100)
    y_fit = np.exp(coeffs[1] + coeffs[0] * x_fit)
    ax1.plot(x_fit, y_fit, '--', color='#d62728', alpha=0.5, linewidth=1.5,
             label='Exponential trend')

    ax1.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Model Parameters (log scale)', fontsize=11, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_xlim(2019.5, 2026.5)

    def format_params(value, pos):
        if value >= 1000:
            return f'{value/1000:.1f}T'
        elif value >= 1:
            return f'{value:.0f}B'
        else:
            return f'{value:.1f}B'

    ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_params))

    years_gpu = [row[0] for row in gpu_data]
    memory_gb = [row[1] for row in gpu_data]
    gpu_names = [row[2] for row in gpu_data]

    ax2.scatter(years_gpu, memory_gb, s=120, alpha=0.7, color='#2ca02c',
                edgecolors='black', linewidths=1.2, zorder=3)

    for i, (year, memory, name) in enumerate(zip(years_gpu, memory_gb, gpu_names)):
        offset_x = 10 if i % 3 == 0 else (-10 if i % 3 == 1 else 0)
        offset_y = 10 if i % 4 == 0 else (-10 if i % 4 == 1 else 5 if i % 4 == 2 else -5)
        ha_align = 'left' if offset_x > 0 else ('right' if offset_x < 0 else 'center')
        va_align = 'bottom' if offset_y > 0 else ('top' if offset_y < 0 else 'center')
        ax2.annotate(f'{name}\n{memory}GB', (year, memory),
                     xytext=(offset_x, offset_y), textcoords='offset points',
                     fontsize=8, ha=ha_align, va=va_align,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               alpha=0.9, edgecolor='#2ca02c', linewidth=1),
                     zorder=4)

    x_fit_gpu = np.linspace(2016, 2026, 100)
    coeffs_gpu = np.polyfit(years_gpu, memory_gb, 1)
    y_fit_gpu = np.maximum(np.polyval(coeffs_gpu, x_fit_gpu), 0)
    ax2.plot(x_fit_gpu, y_fit_gpu, '--', color='#2ca02c', alpha=0.5, linewidth=1.5,
             label='Linear trend')

    ax2.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax2.set_ylabel('GPU Memory (GB)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_xlim(2015.5, 2026.2)
    ax2.set_ylim(0, 220)

    plt.tight_layout()
    save_figure(__file__)


if __name__ == '__main__':
    create_growth_mismatch_plot()
