"""
Edge-Cloud Speculative Decoding

This diagram illustrates how speculative decoding coordinates between edge and cloud:
1. Edge device runs a small, fast draft model
2. Draft tokens are sent to cloud for verification
3. Cloud model verifies in parallel (accepts or rejects)
4. Accepted tokens are returned, rejected ones trigger re-generation
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from math4ai import configure_math_fonts, save_figure

configure_math_fonts()

fig, ax = plt.subplots(figsize=(10, 6))

# Colors
edge_color = '#E3F2FD'
edge_border = '#1976D2'
cloud_color = '#E8F5E9'
cloud_border = '#388E3C'
token_accept = '#4CAF50'
token_reject = '#F44336'
token_draft = '#FFC107'
arrow_color = '#546E7A'

# Edge device box
edge_x, edge_y = 1, 2
edge_w, edge_h = 3.5, 4
edge_box = FancyBboxPatch((edge_x, edge_y), edge_w, edge_h,
                           boxstyle="round,pad=0.03,rounding_size=0.2",
                           facecolor=edge_color, edgecolor=edge_border,
                           linewidth=2, zorder=2)
ax.add_patch(edge_box)
ax.text(edge_x + edge_w/2, edge_y + edge_h + 0.2, 'Edge Device',
        fontsize=12, ha='center', va='bottom', fontweight='bold', color=edge_border)

# Draft model inside edge
draft_x, draft_y = 1.5, 4.5
draft_w, draft_h = 2.5, 1
draft_box = FancyBboxPatch((draft_x, draft_y), draft_w, draft_h,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor='white', edgecolor=edge_border,
                            linewidth=1.5, zorder=3)
ax.add_patch(draft_box)
ax.text(draft_x + draft_w/2, draft_y + draft_h/2, 'Draft Model\n(Small, Fast)',
        fontsize=13, ha='center', va='center', color=edge_border)

# Draft tokens
draft_tokens_y = 3
ax.text(edge_x + edge_w/2, draft_tokens_y + 0.6, 'Draft Tokens', fontsize=13, ha='center', va='bottom')
token_labels = ['The', 'cat', 'sat', 'on', 'the']
for i, label in enumerate(token_labels):
    tx = edge_x + 0.5 + i * 0.6
    circle = Circle((tx, draft_tokens_y), 0.22, facecolor=token_draft, edgecolor='black', linewidth=1, zorder=4)
    ax.add_patch(circle)
    ax.text(tx, draft_tokens_y, label, fontsize=11, ha='center', va='center', zorder=5)

# Cloud box
cloud_x, cloud_y = 7.5, 2
cloud_w, cloud_h = 3.5, 4
cloud_box = FancyBboxPatch((cloud_x, cloud_y), cloud_w, cloud_h,
                            boxstyle="round,pad=0.03,rounding_size=0.2",
                            facecolor=cloud_color, edgecolor=cloud_border,
                            linewidth=2, zorder=2)
ax.add_patch(cloud_box)
ax.text(cloud_x + cloud_w/2, cloud_y + cloud_h + 0.2, 'Cloud',
        fontsize=12, ha='center', va='bottom', fontweight='bold', color=cloud_border)

# Verifier model inside cloud
verify_x, verify_y = 8, 4.5
verify_w, verify_h = 2.5, 1
verify_box = FancyBboxPatch((verify_x, verify_y), verify_w, verify_h,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor='white', edgecolor=cloud_border,
                             linewidth=1.5, zorder=3)
ax.add_patch(verify_box)
ax.text(verify_x + verify_w/2, verify_y + verify_h/2, 'Verifier Model\n(Large, Accurate)',
        fontsize=13, ha='center', va='center', color=cloud_border)

# Verified tokens with accept/reject
verify_tokens_y = 3
ax.text(cloud_x + cloud_w/2, verify_tokens_y + 0.6, 'Verified Tokens', fontsize=13, ha='center', va='bottom')
verify_status = [True, True, True, False, True]  # accept/reject
for i, (label, accepted) in enumerate(zip(token_labels, verify_status)):
    tx = cloud_x + 0.5 + i * 0.6
    color = token_accept if accepted else token_reject
    circle = Circle((tx, verify_tokens_y), 0.22, facecolor=color, edgecolor='black', linewidth=1, zorder=4)
    ax.add_patch(circle)
    ax.text(tx, verify_tokens_y, label, fontsize=11, ha='center', va='center', color='white', zorder=5)

# Arrow: Edge to Cloud (send draft)
ax.annotate('', xy=(cloud_x, 4.8), xytext=(edge_x + edge_w, 4.8),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2,
                           connectionstyle='arc3,rad=-0.1'))
ax.text(6, 5.3, 'Send draft tokens', fontsize=13, ha='center', va='bottom', color=arrow_color)

# Arrow: Cloud to Edge (return verified)
ax.annotate('', xy=(edge_x + edge_w, 3), xytext=(cloud_x, 3),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2,
                           connectionstyle='arc3,rad=-0.1'))
ax.text(6, 2.3, 'Return accepted tokens', fontsize=13, ha='center', va='bottom', color=arrow_color)

# Legend
legend_x, legend_y = 6.5, 1.11
ax.add_patch(Circle((legend_x, legend_y + 0.4), 0.15, facecolor=token_draft, edgecolor='black', linewidth=1))
ax.text(legend_x + 0.3, legend_y + 0.4, 'Draft', fontsize=12, va='center')
ax.add_patch(Circle((legend_x + 1.5, legend_y + 0.4), 0.15, facecolor=token_accept, edgecolor='black', linewidth=1))
ax.text(legend_x + 1.8, legend_y + 0.4, 'Accepted', fontsize=12, va='center')
ax.add_patch(Circle((legend_x + 3.2, legend_y + 0.4), 0.15, facecolor=token_reject, edgecolor='black', linewidth=1))
ax.text(legend_x + 3.5, legend_y + 0.4, 'Rejected', fontsize=12, va='center')

# Speedup annotation
'''ax.text(8, 1.5, 'Speedup: 2-3x when drafts are accepted',
        fontsize=14, ha='center', va='center', style='italic', color='#666666',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5', edgecolor='#CCCCCC'))'''

ax.set_xlim(0.9, 11.1)
ax.set_ylim(1.2, 6.5)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout(pad=0.1)
save_figure(__file__)
