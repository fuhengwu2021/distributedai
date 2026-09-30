#!/usr/bin/env python3
"""
Generate TTFT (Time to First Token) pipeline diagram.
Shows the stages from input to first output token.

Follows ~/mmb's localized_figure standard:
- Single implementation, multiple outputs (<stem>.png for English, <stem>_zh.png for Chinese)
- High-resolution (300 DPI) PNG exports
"""

import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Import localized_figure and styling from shared/figstyle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "title": "Time to First Token (TTFT)",
        "tokenization": "Tokenization",
        "prefill_l1": "Initial Prompt",
        "prefill_l2": "Processing",
        "prefill_l3": "(Prefill)",
        "decode": "decode/generation",
        "detokenization": "De-Tokenization",
        "first_output": "First Output Token",
    },
    "zh": {
        "title": "首字时延全链路时序拆解 (TTFT Pipeline)",
        "tokenization": "输入分词编码",
        "prefill_l1": "初始提示词",
        "prefill_l2": "前向计算",
        "prefill_l3": "(Prefill)",
        "decode": "自回归解码 (Decode)",
        "detokenization": "文本反分词还原",
        "first_output": "生成首个 Token",
    }
}


def draw_box(ax, x, y, width, height, text, facecolor, textcolor='white', fontsize=11):
    """Draw a rounded rectangle with text."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=facecolor,
        edgecolor=facecolor,
        linewidth=2
    )
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', fontsize=fontsize,
            color=textcolor, fontweight='bold')


def draw_arrow(ax, x_start, x_end, y, color='#DC2626'):
    """Draw a horizontal arrow."""
    ax.annotate('', xy=(x_end, y), xytext=(x_start, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5))


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Colors
    purple = '#8B5CF6'
    pink = '#F5A5A5'
    green_light = '#90EE90'
    green_text = '#16A34A'
    red_arrow = '#DC2626'
    text_dark = '#1a1a1a'

    y_center = 1.5
    box_height = 1.0

    # Stage 1: Tokenization
    draw_box(ax, 0.5, y_center, 1.8, box_height, text["tokenization"], purple)

    # Arrow 1
    draw_arrow(ax, 2.4, 3.2, y_center + box_height/2, color=red_arrow)

    # Stage 2: Main processing box (contains Prefill and Decode)
    outer_box = FancyBboxPatch(
        (3.3, y_center - 0.3), 5.0, box_height + 0.6,
        boxstyle="round,pad=0.02,rounding_size=0.2",
        facecolor='#f5f5f5',
        edgecolor='#404040',
        linewidth=2
    )
    ax.add_patch(outer_box)

    # Prefill box inside
    prefill_box = FancyBboxPatch(
        (3.6, y_center), 2.2, box_height,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=pink,
        edgecolor=pink,
        linewidth=2
    )
    ax.add_patch(prefill_box)
    ax.text(3.6 + 2.2/2, y_center + box_height/2 + 0.15, text["prefill_l1"], 
            ha='center', va='center', fontsize=10, color='#1a1a1a', fontweight='bold')
    ax.text(3.6 + 2.2/2, y_center + box_height/2 - 0.15, text["prefill_l2"], 
            ha='center', va='center', fontsize=10, color='#1a1a1a', fontweight='bold')
    ax.text(3.6 + 2.2/2, y_center + box_height/2 - 0.45, text["prefill_l3"], 
            ha='center', va='center', fontsize=9, color='#1a1a1a', fontweight='normal')

    # Decode box inside (smaller, green)
    decode_box = FancyBboxPatch(
        (6.0, y_center + 0.15), 0.5, box_height - 0.3,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=green_light,
        edgecolor=green_light,
        linewidth=2
    )
    ax.add_patch(decode_box)

    # Label below decode box
    ax.text(6.25, y_center - 0.15, text["decode"], 
            ha='center', va='top', fontsize=9, color=green_text, fontweight='normal')

    # Arrow 2
    draw_arrow(ax, 8.4, 9.2, y_center + box_height/2, color=red_arrow)

    # Stage 3: De-Tokenization
    draw_box(ax, 9.3, y_center, 2.0, box_height, text["detokenization"], purple)

    # Arrow 3
    draw_arrow(ax, 11.4, 12.2, y_center + box_height/2, color=red_arrow)

    # Stage 4: First Output Token (text only)
    ax.text(13.0, y_center + box_height/2, text["first_output"], 
            ha='center', va='center', fontsize=12, color=green_text, fontweight='bold')

    # Title
    ax.text(7.0, 3.5, text["title"], 
            ha='center', va='center', fontsize=14, color=text_dark, fontweight='bold')

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    localized_figure(draw, "ttft_pipeline", LABELS, __file__)
