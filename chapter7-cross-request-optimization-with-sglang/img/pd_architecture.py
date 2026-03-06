"""
PD Disaggregation Architecture - Publication-Ready Version

Correct semantics:
1. Single Router/Client with ingress/egress paths clearly labeled
2. Control plane (Router selects decode) vs Data plane (KV transfer)
3. Single request flow: one prefill → one decode (not pool-to-pool convergence)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from math4ai import configure_math_fonts, save_figure
configure_math_fonts()


def draw_box(ax, x, y, width, height, label, color, edge_color, fontsize=14, sublabel=None):
    """Draw a rounded box with label."""
    box = patches.FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=2, edgecolor=edge_color, facecolor=color
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + 0.15, label, ha='center', va='center', 
                fontsize=fontsize, fontweight='bold')
        ax.text(x, y - 0.18, sublabel, ha='center', va='center', 
                fontsize=fontsize-1, color='#555', style='italic')
    else:
        ax.text(x, y, label, ha='center', va='center', 
                fontsize=fontsize, fontweight='bold')


def draw_arrow(ax, start, end, color='#546e7a', style='->', lw=2, linestyle='-'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=linestyle))


def draw_curved_arrow(ax, start, end, color='#546e7a', lw=2, rad=0.3):
    """Draw a curved arrow."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               connectionstyle=f'arc3,rad={rad}'))


def main():
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.set_xlim(1.5, 8.5)
    ax.set_ylim(2, 6.75)
    ax.axis('off')
    
    # Colors
    client_color = '#f5f5f5'
    router_color = '#fce4ec'
    prefill_color = '#e3f2fd'
    decode_color = '#e8f5e9'
    
    # === Single Client at top center ===
    draw_box(ax, 5, 6.3, 1.6, 0.6, 'Client', client_color, '#666', fontsize=14)
    
    # === Single Router below Client ===
    draw_box(ax, 5, 5.0, 2.2, 0.7, 'Router', router_color, '#c62828', fontsize=12)
    
    # Phase labels
    ax.text(2.5, 4.0, 'Phase 1', ha='center', va='center', fontsize=14,
            fontweight='bold', color='#1565c0')
    ax.text(2.5, 3.7, 'Prefill', ha='center', va='center', fontsize=12,
            color='#1565c0')
    
    ax.text(7.5, 4.0, 'Phase 2', ha='center', va='center', fontsize=14,
            fontweight='bold', color='#2e7d32')
    ax.text(7.5, 3.7, 'Decode', ha='center', va='center', fontsize=12,
            color='#2e7d32')
    
    # === Prefill worker (left) ===
    draw_box(ax, 2.5, 2.5, 1.8, 0.9, 'Prefill Worker', prefill_color, '#1976d2', fontsize=14,
             sublabel='compute-bound')
    
    # === Decode worker (right) ===
    draw_box(ax, 7.5, 2.5, 1.8, 0.9, 'Decode Worker', decode_color, '#388e3c', fontsize=14,
             sublabel='memory-bound')
    
    # === Data flow arrows ===
    
    # Client → Router (request)
    draw_arrow(ax, (4.5, 6.0), (4.5, 5.4), color='#666', lw=2)
    ax.text(4.2, 5.7, 'request', ha='right', va='center', fontsize=12, color='#666')
    
    # Router → Prefill (dispatch)
    draw_curved_arrow(ax, (4.0, 4.6), (2.5, 3.0), color='#1976d2', lw=2, rad=0.2)
    ax.text(3.2, 3.9, 'dispatch', ha='left', va='center', fontsize=12, color='#1976d2')
    
    # Router → Decode (control plane: select decode worker) - curved like the green arrow
    ax.annotate('', xy=(7.5, 3.0), xytext=(6.0, 4.6),
                arrowprops=dict(arrowstyle='->', color='#888', lw=1.2, linestyle='--',
                               connectionstyle='arc3,rad=0.2'))
    ax.text(5.4, 3.85, 'select worker', ha='left', va='center', fontsize=12, color='#888', style='italic')
    ax.text(5.4, 3.65, '(control plane)', ha='left', va='center', fontsize=12, color='#aaa', style='italic')
    
    # Prefill → Decode (KV cache transfer - data plane, thick arrow)
    # Draw as a thick colored arrow
    kv_start = (3.4, 2.5)
    kv_end = (6.6, 2.5)
    ax.annotate('', xy=kv_end, xytext=kv_start,
                arrowprops=dict(arrowstyle='->', color='#f57c00', lw=4,
                               mutation_scale=20))
    ax.text(5, 2.65, 'KV cache blocks', ha='center', va='center', fontsize=12,
            fontweight='bold', color='#e65100')
    ax.text(5, 2.3, 'RDMA / Mooncake', ha='center', va='center', fontsize=12,
            color='#888', style='italic')
    
    # Decode → Router (streaming tokens)
    draw_curved_arrow(ax, (7.5, 3.0), (6.0, 4.6), color='#388e3c', lw=2, rad=0.2)
    ax.text(7, 4.4, 'tokens', ha='right', va='center', fontsize=12, color='#388e3c')
    
    # Router → Client (streaming response)
    draw_arrow(ax, (5.5, 5.4), (5.5, 6.0), color='#666', lw=2)
    ax.text(5.8, 5.7, 'stream', ha='left', va='center', fontsize=12, color='#666')
    
    # === Vertical phase divider (subtle) ===
    ax.plot([5, 5], [1.2, 4.3], color='#ddd', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.tight_layout(pad=0.1)
    save_figure(__file__)


if __name__ == '__main__':
    main()
