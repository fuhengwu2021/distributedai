"""
Federated Learning Architecture (FedAvg)

This diagram illustrates the federated learning paradigm:
1. Central server holds the global model
2. Clients train locally on their private data
3. Only model updates (not data) are sent to server
4. Server aggregates updates to improve global model
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from math4ai import configure_math_fonts, save_figure

configure_math_fonts()

fig, ax = plt.subplots(figsize=(11, 8))

# Colors
server_color = '#E8EAF6'
server_border = '#3F51B5'
client_colors = ['#E3F2FD', '#E8F5E9', '#FFF3E0', '#FCE4EC']
client_borders = ['#1976D2', '#388E3C', '#F57C00', '#C2185B']
arrow_down_color = '#3F51B5'
arrow_up_color = '#4CAF50'

# Server box (top center)
server_x, server_y = 3.5, 6
server_w, server_h = 4, 1.5
server_box = FancyBboxPatch((server_x, server_y), server_w, server_h,
                             boxstyle="round,pad=0.03,rounding_size=0.2",
                             facecolor=server_color, edgecolor=server_border,
                             linewidth=2.5, zorder=2)
ax.add_patch(server_box)
ax.text(server_x + server_w/2, server_y + server_h/2, 'Central Server\n(Global Model)',
        fontsize=11, ha='center', va='center', fontweight='bold', color=server_border)

# Clients (bottom row)
client_labels = ['Hospital A', 'Hospital B', 'Bank C', 'Mobile D']
client_data = ['Patient\nRecords', 'Medical\nImages', 'Transaction\nData', 'User\nBehavior']
n_clients = 4
client_w, client_h = 2, 2.2
client_spacing = 2.5
start_x = 0.5

client_centers = []
for i in range(n_clients):
    cx = start_x + i * client_spacing
    cy = 1
    client_centers.append((cx + client_w/2, cy + client_h))
    
    # Client box
    client_box = FancyBboxPatch((cx, cy), client_w, client_h,
                                 boxstyle="round,pad=0.02,rounding_size=0.15",
                                 facecolor=client_colors[i], edgecolor=client_borders[i],
                                 linewidth=2, zorder=2)
    ax.add_patch(client_box)
    
    # Client label
    ax.text(cx + client_w/2, cy + client_h - 0.3, client_labels[i],
            fontsize=9, ha='center', va='top', fontweight='bold', color=client_borders[i])
    
    # Data label (private)
    ax.text(cx + client_w/2, cy + 0.5, client_data[i],
            fontsize=8, ha='center', va='center', color='#666666')
    
    # Privacy indicator
    ax.text(cx + client_w - 0.3, cy + 0.2, 'Private', fontsize=7, ha='center', va='center',
            color='#666666', style='italic')

# Arrows: Server to Clients (distribute model)
server_center_x = server_x + server_w/2
server_bottom = server_y
for i, (cx, cy) in enumerate(client_centers):
    ax.annotate('', xy=(cx, cy), xytext=(server_center_x, server_bottom),
                arrowprops=dict(arrowstyle='->', color=arrow_down_color, lw=1.5,
                               connectionstyle=f'arc3,rad={0.15 * (i - 1.5)}'))

# Arrows: Clients to Server (send updates)
server_top_y = server_y + server_h
for i, (cx, cy) in enumerate(client_centers):
    ax.annotate('', xy=(server_center_x, server_bottom + 0.1), 
                xytext=(cx, cy + 0.1),
                arrowprops=dict(arrowstyle='->', color=arrow_up_color, lw=1.5,
                               connectionstyle=f'arc3,rad={-0.15 * (i - 1.5)}'))

# Labels for arrows
ax.text(1.2, 4.8, 'Distribute\nModel', fontsize=8, ha='center', va='center', 
        color=arrow_down_color, style='italic')
ax.text(9.5, 4.8, 'Send\nUpdates', fontsize=8, ha='center', va='center', 
        color=arrow_up_color, style='italic')

# Process steps annotation
steps_x = 5.5
ax.text(steps_x, 5.2, '① Server sends global model to clients', fontsize=9, ha='center', color='#333333')
ax.text(steps_x, 4.7, '② Clients train locally on private data', fontsize=9, ha='center', color='#333333')
ax.text(steps_x, 4.2, '③ Clients send model updates (not data)', fontsize=9, ha='center', color='#333333')
ax.text(steps_x, 3.7, '④ Server aggregates: FedAvg', fontsize=9, ha='center', color='#333333')

# Key insight box
insight_text = 'Data never leaves the client'
ax.text(5.5, 0.3, insight_text, fontsize=10, ha='center', va='center',
        fontweight='bold', color='#D32F2F',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE', edgecolor='#D32F2F', linewidth=1.5))

ax.set_xlim(-0.5, 11)
ax.set_ylim(-0.2, 8)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout(pad=0.1)
save_figure(__file__)
