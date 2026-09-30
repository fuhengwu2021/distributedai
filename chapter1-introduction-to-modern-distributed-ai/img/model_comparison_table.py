#!/usr/bin/env python3
"""
Visualize the comparison of large AI models using matplotlib.
This script creates a scatter plot with Year on X-axis and Parameters on Y-axis,
with model names as text labels for each point.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from collections import defaultdict
import re
import os
import sys

# Ensure shared directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

# Table data: [Model Name, Parameters, Company, Year]
data = [
    ['ViT-22B', '22B', 'Google', 2023, '2023'],
    ['Grok-1', '314B', 'xAI', 2023, '2023'],
    ['Gemini-1', '1.6T', 'Google', 2023, '2023'],
    ['LLaMA-2', '70B', 'Meta', 2023, '2023'],
    ['PanGu-Σ', '1.085T', 'Huawei', 2023, '2023'],
    ['DeepSeek-V1', '6.7B', 'DeepSeek', 2023, '2023'],
    ['GPT-4V', '~1.8T', 'OpenAI', 2024, '2024'],
    ['DeepSeek-V2', '236B', 'DeepSeek', 2024, '2024'],
    ['Qwen-Max', '~1.2T', 'Alibaba', 2025, '2025'],
    ['GPT-5', '~2–5T', 'OpenAI', 2025, '2025'],
    ['DeepSeek-V3', '671B', 'DeepSeek', 2025, '2025'],
    ['Gemini 3.1 Pro', '~2–3T', 'Google', 2026, '2026'],
    ['Grok 4.3', '~3–6T', 'xAI', 2026, '2026'],
    ['Claude Opus 4.7', '~1T+', 'Anthropic', 2026, '2026'],
    ['GPT-5.5', '~2–5T', 'OpenAI', 2026, '2026'],
    ['Kimi K2.6', '1T', 'Moonshot AI', 2026, '2026'],
    ['DeepSeek-V4-Pro', '1.6T', 'DeepSeek', 2026, '2026'],
    ['Grok V9 Medium', '1.5T', 'xAI', 2026, '2026'],
]

CLUSTER_OFFSETS = [
    (6, 2, 'left', 'bottom'),
    (-6, 2, 'right', 'bottom'),
    (6, -2, 'left', 'top'),
    (-6, -2, 'right', 'top'),
    (6, 0, 'left', 'center'),
    (-6, 0, 'right', 'center'),
]
RIGHT_ONLY_OFFSETS = [
    (6, 2, 'left', 'bottom'),
    (6, -2, 'left', 'top'),
    (6, 0, 'left', 'center'),
    (8, 4, 'left', 'bottom'),
    (8, -4, 'left', 'top'),
]

LABELS = {
    "en": {
        "x_label": "Year",
        "y_label": "Parameters (log scale)",
    },
    "zh": {
        "x_label": "年份",
        "y_label": "模型参数量（对数尺度）",
    }
}


def parse_parameters(param_str):
    range_match = re.search(r'~?(\d+\.?\d*)\s*[–-]\s*(\d+\.?\d*)\s*T', param_str, re.IGNORECASE)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        return ((low + high) / 2) * 1e12

    main_part = param_str.split('(')[0].strip()
    match = re.search(r'[~>]?\s*(\d+\.?\d*)\s*([BMKT])', main_part, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(2).upper()
        multipliers = {'B': 1e9, 'M': 1e6, 'K': 1e3, 'T': 1e12}
        return value * multipliers[unit]
    return None


def build_label(name, param_label):
    display_name = name.rstrip('*')
    param_in_name = re.search(r'\d+\.?\d*\s*[BMKT]', display_name, re.IGNORECASE) is not None
    if param_in_name:
        return display_name
    return f'{display_name} ({param_label})'


def format_parameter_label(param_value):
    if param_value >= 1e12:
        return f'{param_value/1e12:.1f}T'.rstrip('0').rstrip('.')
    elif param_value >= 1e9:
        return f'{param_value/1e9:.0f}B'
    elif param_value >= 1e6:
        return f'{param_value/1e6:.0f}M'
    else:
        return f'{param_value/1e3:.0f}K'


def spread_years(years, parameters):
    by_year = defaultdict(list)
    for i, year in enumerate(years):
        by_year[int(year)].append(i)

    plot_years = list(years)
    for indices in by_year.values():
        n = len(indices)
        if n == 1:
            continue
        spread = 0.11 if n >= 3 else 0.06
        sorted_idx = sorted(indices, key=lambda i: parameters[i])
        for rank, i in enumerate(sorted_idx):
            plot_years[i] = years[i] + (rank - (n - 1) / 2) * (spread / max(n - 1, 1))
    return plot_years


def label_offset_for_index(rank, n, prefer_right=True):
    pool = RIGHT_ONLY_OFFSETS if prefer_right else CLUSTER_OFFSETS
    return pool[rank % len(pool)]


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))

    years = []
    parameters = []
    model_names = []
    param_labels = []

    for row in data:
        model_name, param_str = row[0], row[1]
        year_decimal = row[3]
        param_value = parse_parameters(param_str)
        if param_value is not None:
            years.append(year_decimal)
            parameters.append(param_value)
            model_names.append(model_name)
            param_labels.append(format_parameter_label(param_value))

    plot_years = spread_years(years, parameters)
    year_colors = {2023: '#1f77b4', 2024: '#ff7f0e', 2025: '#2ca02c', 2026: '#d62728'}
    colors = [year_colors[int(y)] for y in years]

    ax.scatter(plot_years, parameters, s=260, alpha=0.75, c=colors,
               edgecolors='red', linewidths=2, zorder=3)

    ax.set_xticks([2023, 2024, 2025, 2026])
    ax.set_xlim(2022.82, 2026.18)
    ax.set_yscale('log')
    ymin, ymax = min(parameters), max(parameters)
    ax.set_ylim(ymin / 1.6, ymax * 1.22)

    by_year = defaultdict(list)
    for i, year in enumerate(years):
        by_year[int(year)].append(i)
    leftmost_year = min(by_year)

    for year, indices in by_year.items():
        sorted_idx = sorted(indices, key=lambda i: parameters[i])
        n = len(sorted_idx)
        prefer_right = year == leftmost_year
        for rank, i in enumerate(sorted_idx):
            dx, dy, ha, va = label_offset_for_index(rank, n, prefer_right=prefer_right)
            ax.annotate(
                build_label(model_names[i], param_labels[i]),
                (plot_years[i], parameters[i]),
                xytext=(dx, dy),
                textcoords='offset points',
                fontsize=8.5,
                ha=ha,
                va=va,
                color=colors[i],
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          alpha=0.8, edgecolor=colors[i], linewidth=0.7),
                zorder=4,
            )

    ax.set_xlabel(text["x_label"], fontsize=12, fontweight='bold')
    ax.set_ylabel(text["y_label"], fontsize=12, fontweight='bold')

    def format_y_axis(value, pos):
        if value >= 1e12:
            return f'{value/1e12:.1f}T'
        elif value >= 1e9:
            return f'{value/1e9:.0f}B'
        else:
            return f'{value/1e6:.0f}M'

    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y_axis))
    ax.xaxis.set_minor_locator(MultipleLocator(1 / 12))
    ax.grid(True, alpha=0.3, linestyle='--', which='both')

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    localized_figure(draw, "model_comparison_table", LABELS, __file__)
