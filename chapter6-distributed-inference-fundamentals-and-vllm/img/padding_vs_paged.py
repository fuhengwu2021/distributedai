"""
Padding vs PagedAttention FLOPs Comparison

Illustrates how traditional batching wastes FLOPs on padding tokens,
while PagedAttention computes only on actual tokens.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def draw_padding_vs_paged():
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
    ax1.set_title('Traditional Batching', fontsize=13, fontweight='bold', pad=10)
    
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
            ax1.text(x + token_width * 0.45, y + token_height/2, 'PAD',
                    ha='center', va='center', fontsize=9, color='#757575')
        
        # Request label
        ax1.text(-0.5, y + token_height/2, f'Req {req_idx + 1}', 
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
    
    ax1.text(-0.5, compute_y + token_height * 0.35, 'Compute', 
            ha='right', va='center', fontsize=11, fontweight='bold')
    
    # Wasted FLOPs annotation
    total_compute = max_len * 3
    actual_compute = sum(req_lengths)
    wasted = total_compute - actual_compute
    ax1.text(max_len * token_width / 2, compute_y - 0.5, 
            f'Wasted: {wasted}/{total_compute} positions ({wasted/total_compute*100:.0f}%)',
            ha='center', va='top', fontsize=11, color='#D32F2F', fontweight='bold')
    
    ax1.set_xlim(-1.5, max_len * token_width + 0.5)
    ax1.set_ylim(-1.8, 3 * (token_height + row_gap) + 0.3)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # === Right panel: PagedAttention (no padding) ===
    ax2 = axes[1]
    ax2.set_title('PagedAttention', fontsize=13, fontweight='bold', pad=10)
    
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
        ax2.text(-0.5, y + token_height/2, f'Req {req_idx + 1}', 
                ha='right', va='center', fontsize=11, fontweight='bold')
        
        # Show length
        ax2.text(length * token_width + 0.2, y + token_height/2, f'({length} tokens)',
                ha='left', va='center', fontsize=10, color='#666666', style='italic')
    
    # Compute indicator - only actual positions
    compute_y = -0.8
    # Show compute matches actual tokens
    ax2.text(-0.5, compute_y + token_height * 0.35, 'Compute', 
            ha='right', va='center', fontsize=11, fontweight='bold')
    
    # Green checkmark area
    rect = patches.FancyBboxPatch((0, compute_y), sum(req_lengths)/3 * token_width, token_height * 0.7,
                                  boxstyle="round,pad=0.02,rounding_size=0.05",
                                  facecolor='#C8E6C9', 
                                  edgecolor='#4CAF50', linewidth=1.5)
    ax2.add_patch(rect)
    ax2.text(sum(req_lengths)/3 * token_width / 2, compute_y + token_height * 0.35, 
            f'{actual_compute} actual tokens',
            ha='center', va='center', fontsize=10, color='#2E7D32')
    
    # No waste annotation
    ax2.text(max_len * token_width / 2, compute_y - 0.5, 
            f'No waste: {actual_compute}/{actual_compute} positions (100% efficient)',
            ha='center', va='top', fontsize=11, color='#2E7D32', fontweight='bold')
    
    ax2.set_xlim(-1.5, max_len * token_width + 1.5)
    ax2.set_ylim(-1.8, 3 * (token_height + row_gap) + 0.3)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Save figure
    plt.tight_layout(pad=0.1)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_path = os.path.join(script_dir, f'{script_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0)
    print(f"Saved figure to: {output_path}")
    plt.close()


if __name__ == '__main__':
    draw_padding_vs_paged()
