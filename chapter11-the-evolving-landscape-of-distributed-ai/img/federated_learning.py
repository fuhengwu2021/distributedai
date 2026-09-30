#!/usr/bin/env python3
"""
Federated Learning Architecture (FedAvg)

This diagram illustrates the federated learning paradigm:
1. Central server holds the global model
2. Clients train locally on their private data
3. Only model updates (not data) are sent to server
4. Server aggregates updates to improve global model

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
        "server": "Central Server\n(Global Model)",
        "client_labels": ["Hospital A", "Hospital B", "Bank C", "Mobile D"],
        "client_data": ["Patient\nRecords", "Medical\nImages", "Transaction\nData", "User\nBehavior"],
        "privacy": "Private",
        "distribute_model": "Distribute\nModel",
        "send_updates": "Send\nUpdates",
        "insight": "Data never leaves the client",
    },
    "zh": {
        "server": "中心服务器\n（全局模型）",
        "client_labels": ["医院 A", "医院 B", "银行 C", "移动端 D"],
        "client_data": ["患者\n病历记录", "医学\n影像数据", "金融\n交易数据", "用户\n行为日志"],
        "privacy": "隐私数据",
        "distribute_model": "分发\n全局模型",
        "send_updates": "上传\n局部更新",
        "insight": "原始数据严格留存在本地",
    }
}


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))

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
    ax.text(server_x + server_w/2, server_y + server_h/2, text["server"],
            fontsize=11, ha='center', va='center', fontweight='bold', color=server_border)

    # Clients (bottom row)
    client_labels = text["client_labels"]
    client_data = text["client_data"]
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
                fontsize=12, ha='center', va='top', fontweight='bold', color=client_borders[i])

        # Data label (private)
        ax.text(cx + client_w/2, cy + 0.95, client_data[i],
                fontsize=11, ha='center', va='center', color='#666666')

        # Privacy indicator
        ax.text(cx + client_w - 0.55, cy + 0.2, text["privacy"], fontsize=10, ha='center', va='center',
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
                    arrowprops=dict(arrowstyle='->', color=arrow_up_color, lw=1.0005,
                                   connectionstyle=f'arc3,rad={-0.15 * (i - 1.5)}'))

    # Labels for arrows
    ax.text(4.0, 4.8, text["distribute_model"], fontsize=11, ha='center', va='center',
            color=arrow_down_color, style='italic')
    ax.text(7, 4.8, text["send_updates"], fontsize=11, ha='center', va='center',
            color=arrow_up_color, style='italic')

    # Key insight box
    ax.text(8.2, 0.3, text["insight"], fontsize=13, ha='center', va='center',
            fontweight='bold', color='#D32F2F',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE', edgecolor='#D32F2F', linewidth=1.5))

    ax.set_xlim(0.45, 10.05)
    ax.set_ylim(-0., 7.6)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout(pad=0.1)
    return fig


if __name__ == '__main__':
    localized_figure(draw, "federated_learning", LABELS, __file__)
