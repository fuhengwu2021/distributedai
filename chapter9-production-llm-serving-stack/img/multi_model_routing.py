#!/usr/bin/env python3
"""
Multi-Model Routing Architecture Diagram
Chapter 9: Production LLM Serving Stack

Shows the architecture of multi-model routing with API Gateway:
- Client applications sending requests
- API Gateway parsing model field and routing
- Multiple vLLM services serving different models
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
        "gw_title": "API Gateway (Unified Entry Point)",
        "gw_features": [
            "Parses 'model' field from request body",
            "Routes to appropriate vLLM service",
            "Returns response to client"
        ],
        "svc1_title": "vLLM Service 1",
        "svc1_model": "(Llama-3.2-1B)",
        "svc1_pod": "Pod: vllm-llama-32-1b",
        "svc1_svc": "Service: vllm-llama-32-1b:8000",
        "svc2_title": "vLLM Service 2",
        "svc2_model": "(Phi-tiny-MoE)",
        "svc2_pod": "Pod: vllm-phi-tiny-moe",
        "svc2_svc": "Service: vllm-phi-tiny-moe:8000",
        "legend_client": "Client",
        "legend_gateway": "Gateway",
        "legend_server": "Model Server",
    },
    "zh": {
        "client_app": "客户端应用程序",
        "request_label": "带 'model' 字段的 HTTP 请求",
        "gw_title": "API Gateway (统一流量接入入口)",
        "gw_features": [
            "解析请求体中的 'model' 目标模型名称",
            "动态路由至对应的 vLLM 后端服务",
            "流式聚合推理响应并返回客户端"
        ],
        "svc1_title": "vLLM 服务 1",
        "svc1_model": "(Llama-3.2-1B)",
        "svc1_pod": "Pod: vllm-llama-32-1b",
        "svc1_svc": "Service: vllm-llama-32-1b:8000",
        "svc2_title": "vLLM 服务 2",
        "svc2_model": "(Phi-tiny-MoE)",
        "svc2_pod": "Pod: vllm-phi-tiny-moe",
        "svc2_svc": "Service: vllm-phi-tiny-moe:8000",
        "legend_client": "客户端",
        "legend_gateway": "网关层",
        "legend_server": "模型服务",
    }
}


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(0.7, 9.3)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')

    # Colors - more contrast
    client_color = '#C5CAE9'     # Medium indigo
    gateway_color = '#BBDEFB'    # Medium blue
    service_color = '#A5D6A7'    # Medium green
    border_color = '#212121'     # Darker gray

    # Client Applications box
    client_box = FancyBboxPatch(
        (1.5, 5.8), 7.0, 0.5,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=client_color,
        edgecolor=border_color,
        linewidth=1.5
    )
    ax.add_patch(client_box)
    ax.text(5, 6.05, text["client_app"], ha='center', va='center',
            fontsize=12, fontweight='bold')

    # Arrow from Client to Gateway
    ax.annotate('', xy=(5, 5.0), xytext=(5, 5.8),
                arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))
    ax.text(5.6, 5.4, text["request_label"], ha='left', va='center',
            fontsize=12, style='italic', color='black')

    # API Gateway box
    gateway_box = FancyBboxPatch(
        (1.0, 3.2), 8.0, 1.8,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=gateway_color,
        edgecolor=border_color,
        linewidth=1.5
    )
    ax.add_patch(gateway_box)
    ax.text(5, 4.7, text["gw_title"], ha='center', va='center',
            fontsize=12, fontweight='bold')

    # Gateway features
    for i, feature in enumerate(text["gw_features"]):
        ax.text(5, 4.2 - i * 0.3, f"• {feature}" if not feature.startswith("•") else feature,
                ha='center', va='center', fontsize=12, color='black')

    # Arrows from Gateway to Services
    ax.annotate('', xy=(3.0, 1.8), xytext=(3.5, 3.2),
                arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))
    ax.annotate('', xy=(7.0, 1.8), xytext=(6.5, 3.2),
                arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))

    # vLLM Service 1 box
    service1_box = FancyBboxPatch(
        (1.0, 0.3), 3.5, 1.5,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=service_color,
        edgecolor='#2E7D32',
        linewidth=1.5
    )
    ax.add_patch(service1_box)
    ax.text(2.75, 1.55, text["svc1_title"], ha='center', va='center',
            fontsize=12, fontweight='bold', color='#2E7D32')
    ax.text(2.75, 1.2, text["svc1_model"], ha='center', va='center',
            fontsize=12, color='black')
    ax.text(2.75, 0.85, text["svc1_pod"], ha='center', va='center',
            fontsize=12, color='black')
    ax.text(2.75, 0.55, text["svc1_svc"], ha='center', va='center',
            fontsize=12, color='black')

    # vLLM Service 2 box
    service2_box = FancyBboxPatch(
        (5.5, 0.3), 3.5, 1.5,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=service_color,
        edgecolor='#2E7D32',
        linewidth=1.5
    )
    ax.add_patch(service2_box)
    ax.text(7.25, 1.55, text["svc2_title"], ha='center', va='center',
            fontsize=12, fontweight='bold', color='#2E7D32')
    ax.text(7.25, 1.2, text["svc2_model"], ha='center', va='center',
            fontsize=12, color='black')
    ax.text(7.25, 0.85, text["svc2_pod"], ha='center', va='center',
            fontsize=12, color='black')
    ax.text(7.25, 0.55, text["svc2_svc"], ha='center', va='center',
            fontsize=12, color='black')

    # Legend
    legend_y = 6.6
    legend_items = [
        (2.0, legend_y, client_color, text["legend_client"]),
        (4.0, legend_y, gateway_color, text["legend_gateway"]),
        (6.5, legend_y, service_color, text["legend_server"]),
    ]

    for x, y, color, label in legend_items:
        box = FancyBboxPatch(
            (x - 0.2, y - 0.12), 0.25, 0.25,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor=color,
            edgecolor=border_color,
            linewidth=1
        )
        ax.add_patch(box)
        ax.text(x + 0.2, y, label, ha='left', va='center', fontsize=12)

    plt.tight_layout(pad=0.1)
    return fig


if __name__ == '__main__':
    localized_figure(draw, "multi_model_routing", LABELS, __file__, pad_inches=0, use_math_fonts=True)
