"""
PagedAttention Block-based Memory Management

Illustrates how PagedAttention uses fixed-size blocks from a shared pool,
allowing immediate reuse when requests finish - eliminating fragmentation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from math4ai import save_figure


def draw_paged_attention_blocks():
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Colors
    pool_color = '#E3F2FD'      # Light blue for block pool
    req1_color = '#E8E8E8'      # Gray for finished request
    req2_color = '#4A90D9'      # Blue for active request
    req3_color = '#5CB85C'      # Green for active request
    free_color = '#FFF9C4'      # Light yellow for free blocks
    
    block_width = 1.2
    block_height = 0.7
    block_gap = 0.15
    
    # Block Pool row (top)
    pool_y = 4.0
    pool_labels = ['Block 0', 'Block 1', 'Block 2', 'Block 3', 'Block 4', 'Block 5']
    pool_colors = [free_color, free_color, req2_color, req3_color, req3_color, free_color]
    pool_edge_colors = ['#FBC02D', '#FBC02D', '#357ABD', '#449D44', '#449D44', '#FBC02D']
    
    ax.text(-0.3, pool_y + block_height/2, 'Block Pool', ha='right', va='center', 
            fontsize=11, fontweight='bold')
    
    for i, (label, color, edge) in enumerate(zip(pool_labels, pool_colors, pool_edge_colors)):
        x = i * (block_width + block_gap)
        rect = patches.FancyBboxPatch((x, pool_y), block_width, block_height,
                                       boxstyle="round,pad=0.02,rounding_size=0.08",
                                       facecolor=color, edgecolor=edge, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + block_width/2, pool_y + block_height/2, label,
                ha='center', va='center', fontsize=12)
    
    # Add "..." after blocks
    ax.text(6 * (block_width + block_gap) + 0.3, pool_y + block_height/2, '...',
            ha='left', va='center', fontsize=16, fontweight='bold', color='#666')
    
    # Arrows from pool to requests
    arrow_style = dict(arrowstyle='-|>', color='#666666', lw=1.5)
    
    # Request rows
    req_y_positions = [2.5, 1.5, 0.5]
    
    # Request 1: finished, blocks returned (Block 0, Block 1 - now free)
    req1_y = req_y_positions[0]
    ax.text(-0.3, req1_y + block_height/2, 'Request 1', ha='right', va='center', 
            fontsize=11, fontweight='bold', color='#666666')
    
    # Show returned blocks with dashed outline
    for i, block_idx in enumerate([0, 1]):
        x = i * (block_width + block_gap) + 0.5
        rect = patches.FancyBboxPatch((x, req1_y), block_width, block_height,
                                       boxstyle="round,pad=0.02,rounding_size=0.08",
                                       facecolor=req1_color, edgecolor='#999999', 
                                       linewidth=1.5, linestyle='--')
        ax.add_patch(rect)
        ax.text(x + block_width/2, req1_y + block_height/2, f'Block {block_idx}',
                ha='center', va='center', fontsize=12, color='#666666')
    
    # Status label
    ax.text(2 * (block_width + block_gap) + 0.5, req1_y + block_height/2, 
            '(finished, blocks returned)',
            ha='left', va='center', fontsize=12, color='#666666', style='italic')
    
    # Curved arrows showing blocks returned to pool
    ax.annotate('', xy=(0 * (block_width + block_gap) + block_width/2, pool_y),
                xytext=(0.5 + block_width/2, req1_y + block_height),
                arrowprops=dict(arrowstyle='-|>,head_length=1.2', color='#4CAF50', lw=1.5,
                               connectionstyle='arc3,rad=0.2'))
    ax.annotate('', xy=(1 * (block_width + block_gap) + block_width/2, pool_y),
                xytext=(0.5 + block_width + block_gap + block_width/2, req1_y + block_height),
                arrowprops=dict(arrowstyle='-|>,head_length=1.2', color='#4CAF50', lw=1.5,
                               connectionstyle='arc3,rad=0.2'))
    
    # Request 2: active, using Block 2
    req2_y = req_y_positions[1]
    ax.text(-0.3, req2_y + block_height/2, 'Request 2', ha='right', va='center', 
            fontsize=11, fontweight='bold')
    
    x = 0.5
    rect = patches.FancyBboxPatch((x, req2_y), block_width, block_height,
                                   boxstyle="round,pad=0.02,rounding_size=0.08",
                                   facecolor=req2_color, edgecolor='#357ABD', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + block_width/2, req2_y + block_height/2, 'Block 2',
            ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    
    ax.text(x + block_width + 0.3, req2_y + block_height/2, '(active)',
            ha='left', va='center', fontsize=12, color='#357ABD', style='italic')
    
    # Arrow from pool Block 2 to Request 2
    ax.annotate('', xy=(x + block_width/2, req2_y + block_height),
                xytext=(2 * (block_width + block_gap) + block_width/2, pool_y),
                arrowprops=dict(arrowstyle='-|>,head_length=1.2', color='#357ABD', lw=1.5,
                               connectionstyle='arc3,rad=-0.15'))
    
    # Request 3: active, using Block 3, Block 4
    req3_y = req_y_positions[2]
    ax.text(-0.3, req3_y + block_height/2, 'Request 3', ha='right', va='center', 
            fontsize=11, fontweight='bold')
    
    for i, block_idx in enumerate([3, 4]):
        x = i * (block_width + block_gap) + 0.5
        rect = patches.FancyBboxPatch((x, req3_y), block_width, block_height,
                                       boxstyle="round,pad=0.02,rounding_size=0.08",
                                       facecolor=req3_color, edgecolor='#449D44', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + block_width/2, req3_y + block_height/2, f'Block {block_idx}',
                ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    
    ax.text(2 * (block_width + block_gap) + 1.2, req3_y + block_height/2, '(active)',
            ha='left', va='center', fontsize=12, color='#449D44', style='italic')
    
    # Arrows from pool Block 3, 4 to Request 3
    ax.annotate('', xy=(0.5 + block_width/2, req3_y + block_height),
                xytext=(3 * (block_width + block_gap) + block_width/2, pool_y),
                arrowprops=dict(arrowstyle='-|>,head_length=1.2', color='#449D44', lw=1.5,
                               connectionstyle='arc3,rad=-0.2'))
    ax.annotate('', xy=(0.5 + block_width + block_gap + block_width/2, req3_y + block_height),
                xytext=(4 * (block_width + block_gap) + block_width/2, pool_y),
                arrowprops=dict(arrowstyle='-|>,head_length=1.2', color='#449D44', lw=1.5,
                               connectionstyle='arc3,rad=-0.2'))
    
    # Legend / annotation
    ax.text(6.5, 1.5, 'No fragmentation:\nfreed blocks return\nto pool immediately',
            ha='center', va='center', fontsize=13, color='#4CAF50',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#4CAF50'))
    
    # Set axis properties
    ax.set_xlim(-1.3, 8.8)
    ax.set_ylim(0.4, 4.8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout(pad=0.1)
    save_figure(__file__)


if __name__ == '__main__':
    draw_paged_attention_blocks()
