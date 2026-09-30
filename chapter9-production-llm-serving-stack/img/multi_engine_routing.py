#!/usr/bin/env python3
"""
Multi-Engine Routing Architecture Diagram
Chapter 9: Production LLM Serving Stack

Shows the architecture of routing same model to different inference engines:
- Client applications sending requests with model and owned_by fields
- API Gateway parsing both fields and routing accordingly
- vLLM and SGLang services serving the same model
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
        "req_line1": "HTTP Request with 'model'",
        "req_line2": "and 'owned_by' fields",
        "gw_title": "API Gateway (Unified Entry Point)",
        "gw_features": [
            "Parses 'model' field from request body",
            "Parses 'owned_by' field for engine selection",
            "Routes to appropriate service based on both fields",
            "Returns response to client"
        ],
        "vllm_title": "vLLM Service",
        "sglang_title": "SGLang Service",
        "model_name": "(Llama-3.2-1B)",
        "vllm_pod": "Pod: vllm-llama-32-1b",
        "vllm_svc": "Service: vllm-llama-32-1b:8000",
        "vllm_tag": 'inference_server: "vllm"',
        "vllm_image": "Image: vllm/vllm-openai:v0.12.0",
        "sglang_pod": "Pod: sglang-llama-32-1b",
        "sglang_svc": "Service: sglang-llama-32-1b:8000",
        "sglang_tag": 'inference_server: "sglang"',
        "sglang_image": "Image: lmsysorg/sglang:v0.5.6",
        "legend_client": "Client",
        "legend_gateway": "Gateway",
        "legend_vllm": "vLLM",
        "legend_sglang": "SGLang",
    },
    "zh": {
        "client_app": "客户端应用程序",
        "req_line1": "带 'model' 与",
        "req_line2": "'owned_by' 字段的 HTTP 请求",
        "gw_title": "API Gateway (统一流量接入入口)",
        "gw_features": [
            "解析请求体中的 'model' 目标模型字段",
            "解析 'owned_by' 字段以指定推理后端引擎",
            "基于双重字段联合动态路由至目标服务",
            "聚合后端流式推理响应并回传客户端"
        ],
        "vllm_title": "vLLM 推理服务",
        "sglang_title": "SGLang 推理服务",
        "model_name": "(Llama-3.2-1B)",
        "vllm_pod": "Pod: vllm-llama-32-1b",
        "vllm_svc": "Service: vllm-llama-32-1b:8000",
        "vllm_tag": 'inference_server: "vllm"',
        "vllm_image": "镜像: vllm/vllm-openai:v0.12.0",
        "sglang_pod": "Pod: sglang-llama-32-1b",
        "sglang_svc": "Service: sglang-llama-32-1b:8000",
        "sglang_tag": 'inference_server: "sglang"',
        "sglang_image": "镜像: lmsysorg/sglang:v0.5.6",
        "legend_client": "客户端",
        "legend_gateway": "网关层",
        "legend_vllm": "vLLM 后端",
        "legend_sglang": "SGLang 后端",
    }
}


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(0.7, 9.3)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Colors - more contrast
    client_color = '#C5CAE9'     # Medium indigo
    gateway_color = '#BBDEFB'    # Medium blue
    vllm_color = '#A5D6A7'       # Medium green
    sglang_color = '#FFCC80'     # Medium orange
    border_color = '#212121'     # Darker gray

    # Client Applications box
    client_box = FancyBboxPatch(
        (1.5, 6.8), 7.0, 0.5,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=client_color,
        edgecolor=border_color,
        linewidth=1.5
    )
    ax.add_patch(client_box)
    ax.text(5, 7.05, text["client_app"], ha='center', va='center',
            fontsize=12, fontweight='bold')

    # Arrow from Client to Gateway
    ax.annotate('', xy=(5, 5.9), xytext=(5, 6.8),
                arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))
    ax.text(5.6, 6.35, text["req_line1"], ha='left', va='center',
            fontsize=11, style='italic', color='black')
    ax.text(5.6, 6.05, text["req_line2"], ha='left', va='center',
            fontsize=11, style='italic', color='black')

    # API Gateway box
    gateway_box = FancyBboxPatch(
        (1.0, 4.0), 8.0, 1.9,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=gateway_color,
        edgecolor=border_color,
        linewidth=1.5
    )
    ax.add_patch(gateway_box)
    ax.text(5, 5.6, text["gw_title"], ha='center', va='center',
            fontsize=11, fontweight='bold')

    # Gateway features
    for i, feature in enumerate(text["gw_features"]):
        ax.text(5, 5.15 - i * 0.28, f"• {feature}" if not feature.startswith("•") else feature,
                ha='center', va='center', fontsize=11, color='black')

    # Arrows from Gateway to Services
    ax.annotate('', xy=(3.0, 2.6), xytext=(3.5, 4.0),
                arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))
    ax.annotate('', xy=(7.0, 2.6), xytext=(6.5, 4.0),
                arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))

    # vLLM Service box
    vllm_box = FancyBboxPatch(
        (0.8, 0.3), 3.9, 2.3,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=vllm_color,
        edgecolor='#2E7D32',
        linewidth=1.5
    )
    ax.add_patch(vllm_box)
    ax.text(2.75, 2.35, text["vllm_title"], ha='center', va='center',
            fontsize=11, fontweight='bold', color='#2E7D32')
    ax.text(2.75, 2.0, text["model_name"], ha='center', va='center',
            fontsize=10, color='black')
    ax.text(2.75, 1.65, text["vllm_pod"], ha='center', va='center',
            fontsize=10, color='black')
    ax.text(2.75, 1.35, text["vllm_svc"], ha='center', va='center',
            fontsize=10, color='black')
    ax.text(2.75, 1.0, text["vllm_tag"], ha='center', va='center',
            fontsize=11, fontweight='bold', color='#2E7D32')
    ax.text(2.75, 0.65, text["vllm_image"], ha='center', va='center',
            fontsize=10, style='italic', color='black')

    # SGLang Service box
    sglang_box = FancyBboxPatch(
        (5.3, 0.3), 3.9, 2.3,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=sglang_color,
        edgecolor='#E65100',
        linewidth=1.5
    )
    ax.add_patch(sglang_box)
    ax.text(7.25, 2.35, text["sglang_title"], ha='center', va='center',
            fontsize=11, fontweight='bold', color='#E65100')
    ax.text(7.25, 2.0, text["model_name"], ha='center', va='center',
            fontsize=10, color='black')
    ax.text(7.25, 1.65, text["sglang_pod"], ha='center', va='center',
            fontsize=10, color='black')
    ax.text(7.25, 1.35, text["sglang_svc"], ha='center', va='center',
            fontsize=10, color='black')
    ax.text(7.25, 1.0, text["sglang_tag"], ha='center', va='center',
            fontsize=11, fontweight='bold', color='#E65100')
    ax.text(7.25, 0.65, text["sglang_image"], ha='center', va='center',
            fontsize=10, style='italic', color='black')

    # Legend
    legend_y = 7.6
    legend_items = [
        (1.8, legend_y, client_color, text["legend_client"]),
        (3.5, legend_y, gateway_color, text["legend_gateway"]),
        (5.2, legend_y, vllm_color, text["legend_vllm"]),
        (6.7, legend_y, sglang_color, text["legend_sglang"]),
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
        ax.text(x + 0.2, y, label, ha='left', va='center', fontsize=11)

    plt.tight_layout(pad=0.1)
    return fig


if __name__ == '__main__':
    localized_figure(draw, "multi_engine_routing", LABELS, __file__, pad_inches=0, use_math_fonts=True)
