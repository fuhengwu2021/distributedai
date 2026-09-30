#!/usr/bin/env python3
"""
llm-d Multi-Model Architecture Diagram
Chapter 9: Production LLM Serving Stack

Shows the llm-d architecture for multi-model serving:
- Client applications sending requests
- Inference Gateway with intelligent load balancing
- InferencePool routing layer
- Multiple ModelService instances with vLLM pods
"""

import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Add shared directory to path for figstyle imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "client_app": "Client Applications",
        "request_label": "HTTP Request with 'model' field",
        "gw_title": "Inference Gateway (Kubernetes Gateway API)",
        "gw_features": [
            "Load balancing with prefix-cache awareness",
            "Intelligent request scheduling",
            "Traffic routing to InferencePool"
        ],
        "pool_title": "InferencePool (Routing Layer)",
        "pool_features": [
            "Routes requests to appropriate ModelService",
            "Model-aware request distribution",
            "Health checking and load balancing"
        ],
        "service1_title": "ModelService 1",
        "service2_title": "ModelService 2",
        "service1_model": "(Llama-3.2-1B)",
        "service2_model": "(Qwen2.5-0.5B)",
        "pods_label": "vLLM Pods (2x)",
        "service_f1": "• Intelligent load balancing",
        "service_f2": "• Prefix cache aware routing",
        "legend_client": "Client",
        "legend_gateway": "Gateway",
        "legend_pool": "Pool",
        "legend_service": "Service",
    },
    "zh": {
        "client_app": "客户端应用程序",
        "request_label": "带 'model' 字段的 HTTP 请求",
        "gw_title": "Inference Gateway (Kubernetes Gateway API)",
        "gw_features": [
            "感知 Prefix-Cache 前缀缓存的负载均衡",
            "请求智能调度与队列管理",
            "流量精准路由至 InferencePool"
        ],
        "pool_title": "InferencePool (推理路由层)",
        "pool_features": [
            "将请求路由至目标 ModelService",
            "模型感知的智能请求分发机制",
            "实例健康检查与动态负载均衡"
        ],
        "service1_title": "ModelService 1",
        "service2_title": "ModelService 2",
        "service1_model": "(Llama-3.2-1B)",
        "service2_model": "(Qwen2.5-0.5B)",
        "pods_label": "vLLM Pod 副本 (2x)",
        "service_f1": "• 智能负载均衡",
        "service_f2": "• 前缀缓存感知路由",
        "legend_client": "客户端",
        "legend_gateway": "网关层",
        "legend_pool": "路由池",
        "legend_service": "服务层",
    }
}


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(0.5, 9.5)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')

    # Colors - consistent with other diagrams
    client_color = '#C5CAE9'     # Medium indigo
    gateway_color = '#BBDEFB'    # Medium blue
    pool_color = '#B3E5FC'       # Light cyan
    service_color = '#A5D6A7'    # Medium green
    border_color = '#212121'     # Darker gray

    # Client Applications box
    client_box = FancyBboxPatch(
        (1.5, 8.0), 7.0, 0.5,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=client_color,
        edgecolor=border_color,
        linewidth=1.5
    )
    ax.add_patch(client_box)
    ax.text(5, 8.25, text["client_app"], ha='center', va='center',
            fontsize=12, fontweight='bold')

    # Arrow from Client to Gateway
    ax.annotate('', xy=(5, 7.1), xytext=(5, 8.0),
                arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))
    ax.text(5.6, 7.55, text["request_label"], ha='left', va='center',
            fontsize=11, style='italic', color='black')

    # Inference Gateway box
    gateway_box = FancyBboxPatch(
        (1.0, 5.5), 8.0, 1.6,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=gateway_color,
        edgecolor=border_color,
        linewidth=1.5
    )
    ax.add_patch(gateway_box)
    ax.text(5, 6.8, text["gw_title"], ha='center', va='center',
            fontsize=11, fontweight='bold')

    # Gateway features
    for i, feature in enumerate(text["gw_features"]):
        ax.text(5, 6.35 - i * 0.28, f"• {feature}" if not feature.startswith("•") else feature,
                ha='center', va='center', fontsize=10, color='black')

    # Arrow from Gateway to Pool
    ax.annotate('', xy=(5, 4.6), xytext=(5, 5.5),
                arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))

    # InferencePool box
    pool_box = FancyBboxPatch(
        (1.0, 3.0), 8.0, 1.6,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=pool_color,
        edgecolor='#0277BD',
        linewidth=1.5
    )
    ax.add_patch(pool_box)
    ax.text(5, 4.3, text["pool_title"], ha='center', va='center',
            fontsize=11, fontweight='bold', color='#0277BD')

    # Pool features
    for i, feature in enumerate(text["pool_features"]):
        ax.text(5, 3.85 - i * 0.28, f"• {feature}" if not feature.startswith("•") else feature,
                ha='center', va='center', fontsize=10, color='black')

    # Arrows from Pool to Services
    ax.annotate('', xy=(2.75, 2.1), xytext=(3.5, 3.0),
                arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))
    ax.annotate('', xy=(7.25, 2.1), xytext=(6.5, 3.0),
                arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))

    # ModelService 1 box
    service1_box = FancyBboxPatch(
        (0.6, 0.2), 3.9, 1.9,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=service_color,
        edgecolor='#2E7D32',
        linewidth=1.5
    )
    ax.add_patch(service1_box)
    ax.text(2.55, 1.9, text["service1_title"], ha='center', va='center',
            fontsize=11, fontweight='bold', color='#2E7D32')
    ax.text(2.55, 1.55, text["service1_model"], ha='center', va='center',
            fontsize=10, color='black')
    ax.text(2.55, 1.2, text["pods_label"], ha='center', va='center',
            fontsize=10, color='black')
    ax.text(2.55, 0.85, text["service_f1"], ha='center', va='center',
            fontsize=9, color='black')
    ax.text(2.55, 0.55, text["service_f2"], ha='center', va='center',
            fontsize=9, color='black')

    # ModelService 2 box
    service2_box = FancyBboxPatch(
        (5.5, 0.2), 3.9, 1.9,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=service_color,
        edgecolor='#2E7D32',
        linewidth=1.5
    )
    ax.add_patch(service2_box)
    ax.text(7.45, 1.9, text["service2_title"], ha='center', va='center',
            fontsize=11, fontweight='bold', color='#2E7D32')
    ax.text(7.45, 1.55, text["service2_model"], ha='center', va='center',
            fontsize=10, color='black')
    ax.text(7.45, 1.2, text["pods_label"], ha='center', va='center',
            fontsize=10, color='black')
    ax.text(7.45, 0.85, text["service_f1"], ha='center', va='center',
            fontsize=9, color='black')
    ax.text(7.45, 0.55, text["service_f2"], ha='center', va='center',
            fontsize=9, color='black')

    # Legend
    legend_y = 8.7
    legend_items = [
        (1.5, legend_y, client_color, text["legend_client"]),
        (3.0, legend_y, gateway_color, text["legend_gateway"]),
        (4.8, legend_y, pool_color, text["legend_pool"]),
        (6.3, legend_y, service_color, text["legend_service"]),
    ]

    for x, y, color, label in legend_items:
        box = FancyBboxPatch(
            (x - 0.2, y - 0.012), 0.25, 0.25,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor=color,
            edgecolor=border_color,
            linewidth=1
        )
        ax.add_patch(box)
        ax.text(x + 0.12, y+0.1, label, ha='left', va='center', fontsize=11)

    plt.tight_layout(pad=0.1)
    return fig


if __name__ == '__main__':
    localized_figure(draw, "llmd_multi_model", LABELS, __file__, pad_inches=0, use_math_fonts=True)
