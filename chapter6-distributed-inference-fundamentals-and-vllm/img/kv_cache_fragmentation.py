"""
KV Cache Memory Fragmentation Diagram

Illustrates the memory fragmentation problem with contiguous KV cache allocation.
When Request 1 finishes, its memory cannot be efficiently reused by other active requests.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def draw_kv_cache_fragmentation():
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Colors
    req1_color = '#E8E8E8'  # Light gray for finished request
    req2_color = '#4A90D9'  # Blue for active request
    req3_color = '#5CB85C'  # Green for active request
    empty_color = '#FFFFFF'  # White for empty/wasted space
    
    # Memory layout parameters
    total_width = 20
    block_height = 0.8
    y_positions = [3, 2, 1]  # Y positions for each request row
    
    # Request 1: 8 tokens, finished (grayed out)
    req1_width = 8
    rect1 = patches.FancyBboxPatch((0, y_positions[0]), req1_width, block_height,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    facecolor=req1_color, edgecolor='#999999', linewidth=1.5,
                                    linestyle='--')
    ax.add_patch(rect1)
    ax.text(req1_width/2, y_positions[0] + block_height/2, 'Request 1: 8 tokens (finished)',
            ha='center', va='center', fontsize=11, color='#666666')
    
    # Wasted space after Request 1
    wasted1 = patches.FancyBboxPatch((req1_width, y_positions[0]), total_width - req1_width, block_height,
                                      boxstyle="round,pad=0.02,rounding_size=0.1",
                                      facecolor=empty_color, edgecolor='#CCCCCC', linewidth=1,
                                      linestyle=':')
    ax.add_patch(wasted1)
    ax.text((req1_width + total_width)/2, y_positions[0] + block_height/2, 'Wasted (cannot reuse)',
            ha='center', va='center', fontsize=11, color='#999999', style='italic')
    
    # Request 2: 4 tokens, active
    req2_width = 4
    rect2 = patches.FancyBboxPatch((0, y_positions[1]), req2_width, block_height,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    facecolor=req2_color, edgecolor='#357ABD', linewidth=1.5)
    ax.add_patch(rect2)
    ax.text(req2_width/2, y_positions[1] + block_height/2, 'Request 2: 4 tokens',
            ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    
    # Reserved space for Request 2 (pre-allocated but unused)
    reserved2 = patches.FancyBboxPatch((req2_width, y_positions[1]), 6, block_height,
                                        boxstyle="round,pad=0.02,rounding_size=0.1",
                                        facecolor='#E3F2FD', edgecolor='#90CAF9', linewidth=1,
                                        linestyle=':')
    ax.add_patch(reserved2)
    ax.text(req2_width + 3, y_positions[1] + block_height/2, 'Reserved',
            ha='center', va='center', fontsize=11, color='#64B5F6', style='italic')
    
    # Request 3: 10 tokens, active
    req3_width = 10
    rect3 = patches.FancyBboxPatch((0, y_positions[2]), req3_width, block_height,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    facecolor=req3_color, edgecolor='#449D44', linewidth=1.5)
    ax.add_patch(rect3)
    ax.text(req3_width/2, y_positions[2] + block_height/2, 'Request 3: 10 tokens',
            ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    
    # Reserved space for Request 3
    reserved3 = patches.FancyBboxPatch((req3_width, y_positions[2]), 4, block_height,
                                        boxstyle="round,pad=0.02,rounding_size=0.1",
                                        facecolor='#E8F5E9', edgecolor='#A5D6A7', linewidth=1,
                                        linestyle=':')
    ax.add_patch(reserved3)
    ax.text(req3_width + 2, y_positions[2] + block_height/2, 'Reserved',
            ha='center', va='center', fontsize=11, color='#81C784', style='italic')
    
    # Add fragmentation annotation
    ax.annotate('Memory fragmentation:\nRequest 1\'s space cannot\nbe reused by others',
                xy=(req1_width, y_positions[0] + block_height/2),
                xytext=(16, y_positions[0]-1.5),
                fontsize=11, color='#D32F2F',
                arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5),
                ha='center')
    
    # Add label for GPU Memory
    ax.text(-0.5, y_positions[0] + block_height/2, 'GPU\nMemory', ha='right', va='center', 
            fontsize=11, fontweight='bold')
    
    # Memory address arrow
    ax.annotate('', xy=(total_width + 0.5, 0.5), xytext=(0, 0.5),
                arrowprops=dict(arrowstyle='->', color='#666666', lw=1))
    ax.text(total_width/2, 0.2, 'Memory Address', ha='center', va='top', fontsize=11, color='#666666')
    
    # Set axis properties
    ax.set_xlim(-2, total_width + 0.5)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Save figure following the standard pattern
    plt.tight_layout(pad=0.1)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_path = os.path.join(script_dir, f'{script_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0)
    print(f"Saved figure to: {output_path}")
    plt.close()


if __name__ == '__main__':
    draw_kv_cache_fragmentation()
