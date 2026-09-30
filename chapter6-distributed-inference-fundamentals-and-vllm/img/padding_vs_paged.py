"""
Padding vs PagedAttention FLOPs Comparison

Illustrates how traditional batching wastes FLOPs on padding tokens,
while PagedAttention computes only on actual tokens.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "title_left": "Traditional Batching",
        "title_right": "PagedAttention",
        "pad": "PAD",
        "req": "Req {idx}",
        "compute": "Compute",
        "wasted": "Wasted: {wasted}/{total} positions ({pct:.0f}%)",
        "token_len": "({length} tokens)",
        "actual_tokens": "{actual_compute} actual tokens",
        "no_waste": "No waste: {actual}/{total} positions (100% efficient)",
    },
    "zh": {
        "title_left": "传统批处理 (Padding)",
        "title_right": "PagedAttention (无 Padding)",
        "pad": "PAD",
        "req": "请求 {idx}",
        "compute": "计算负载",
        "wasted": "浪费: {wasted}/{total} 个位置 ({pct:.0f}%)",
        "token_len": "({length} 个 Token)",
        "actual_tokens": "{actual_compute} 个实际有效 Token",
        "no_waste": "零浪费: {actual}/{total} 个位置 (100% 计算效率)",
    }
}


def draw(text: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Colors
    token_colors = ['#4A90D9', '#5CB85C', '#F5A623']  # Blue, Green, Orange for 3 requests
    padding_color = '#E0E0E0'  # Gray for padding
    compute_color = '#FFCDD2'  # Light red for wasted compute
    
    # Request lengths
    req_lengths = [4, 7, 3]  # Request 1: 4 tokens, Request 2: 7 tokens, Request 3: 3 tokens
    max_len = max(req_lengths)
    
    token_width = 0.8
    token_height = 0.6
    row_gap = 0.3
    
    # === Left panel: Traditional Batching with Padding ===
    ax1 = axes[0]
    ax1.set_title(text["title_left"], fontsize=13, fontweight='bold', pad=10)
    
    for req_idx, length in enumerate(req_lengths):
        y = (2 - req_idx) * (token_height + row_gap)
        
        # Draw actual tokens
        for t in range(length):
            x = t * token_width
            rect = patches.FancyBboxPatch((x, y), token_width * 0.9, token_height,
                                          boxstyle="round,pad=0.02,rounding_size=0.05",
                                          facecolor=token_colors[req_idx], 
                                          edgecolor='white', linewidth=1)
            ax1.add_patch(rect)
        
        # Draw padding tokens
        for t in range(length, max_len):
            x = t * token_width
            rect = patches.FancyBboxPatch((x, y), token_width * 0.9, token_height,
                                          boxstyle="round,pad=0.02,rounding_size=0.05",
                                          facecolor=padding_color, 
                                          edgecolor='#BDBDBD', linewidth=1, linestyle='--')
            ax1.add_patch(rect)
            ax1.text(x + token_width * 0.45, y + token_height/2, text["pad"],
                    ha='center', va='center', fontsize=9, color='#757575')
        
        # Request label
        ax1.text(-0.5, y + token_height/2, text["req"].format(idx=req_idx + 1), 
                ha='right', va='center', fontsize=11, fontweight='bold')
    
    # Compute indicator - show all positions are computed
    compute_y = -0.8
    for t in range(max_len):
        x = t * token_width
        # Count actual vs padding
        actual = sum(1 for l in req_lengths if t < l)
        padding = 3 - actual
        
        if padding > 0:
            rect = patches.FancyBboxPatch((x, compute_y), token_width * 0.9, token_height * 0.7,
                                          boxstyle="round,pad=0.02,rounding_size=0.05",
                                          facecolor=compute_color, 
                                          edgecolor='#EF5350', linewidth=1.5)
            ax1.add_patch(rect)
    
    ax1.text(-0.5, compute_y + token_height * 0.35, text["compute"], 
            ha='right', va='center', fontsize=11, fontweight='bold')
    
    # Wasted FLOPs annotation
    total_compute = max_len * 3
    actual_compute = sum(req_lengths)
    wasted = total_compute - actual_compute
    ax1.text(max_len * token_width / 2, compute_y - 0.5, 
            text["wasted"].format(wasted=wasted, total=total_compute, pct=wasted/total_compute*100),
            ha='center', va='top', fontsize=11, color='#D32F2F', fontweight='bold')
    
    ax1.set_xlim(-1.5, max_len * token_width + 0.5)
    ax1.set_ylim(-1.8, 3 * (token_height + row_gap) + 0.3)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # === Right panel: PagedAttention (no padding) ===
    ax2 = axes[1]
    ax2.set_title(text["title_right"], fontsize=13, fontweight='bold', pad=10)
    
    for req_idx, length in enumerate(req_lengths):
        y = (2 - req_idx) * (token_height + row_gap)
        
        # Draw only actual tokens (no padding)
        for t in range(length):
            x = t * token_width
            rect = patches.FancyBboxPatch((x, y), token_width * 0.9, token_height,
                                          boxstyle="round,pad=0.02,rounding_size=0.05",
                                          facecolor=token_colors[req_idx], 
                                          edgecolor='white', linewidth=1)
            ax2.add_patch(rect)
        
        # Request label
        ax2.text(-0.5, y + token_height/2, text["req"].format(idx=req_idx + 1), 
                ha='right', va='center', fontsize=11, fontweight='bold')
        
        # Show length
        ax2.text(length * token_width + 0.2, y + token_height/2, text["token_len"].format(length=length),
                ha='left', va='center', fontsize=10, color='#666666', style='italic')
    
    # Compute indicator - only actual positions
    compute_y = -0.8
    # Show compute matches actual tokens
    ax2.text(-0.5, compute_y + token_height * 0.35, text["compute"], 
            ha='right', va='center', fontsize=11, fontweight='bold')
    
    # Green checkmark area
    rect = patches.FancyBboxPatch((0, compute_y), sum(req_lengths)/3 * token_width, token_height * 0.7,
                                  boxstyle="round,pad=0.02,rounding_size=0.05",
                                  facecolor='#C8E6C9', 
                                  edgecolor='#4CAF50', linewidth=1.5)
    ax2.add_patch(rect)
    ax2.text(sum(req_lengths)/3 * token_width / 2, compute_y + token_height * 0.35, 
            text["actual_tokens"].format(actual_compute=actual_compute),
            ha='center', va='center', fontsize=10, color='#2E7D32')
    
    # No waste annotation
    ax2.text(max_len * token_width / 2, compute_y - 0.5, 
            text["no_waste"].format(actual=actual_compute, total=actual_compute),
            ha='center', va='top', fontsize=11, color='#2E7D32', fontweight='bold')
    
    ax2.set_xlim(-1.5, max_len * token_width + 1.5)
    ax2.set_ylim(-1.8, 3 * (token_height + row_gap) + 0.3)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    plt.tight_layout(pad=0.1)
    return fig


if __name__ == '__main__':
    localized_figure(draw, "padding_vs_paged", LABELS, __file__, pad_inches=0)
