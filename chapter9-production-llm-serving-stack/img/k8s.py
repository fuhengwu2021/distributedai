#!/usr/bin/env python3
"""
Kubernetes LLM Serving Architecture Diagram
Chapter 9: Production LLM Serving Stack

Shows how Kubernetes primitives map to LLM serving concepts:
- Ingress/Service for routing
- Deployments/Pods for model runners
- HPA for autoscaling
- GPU scheduling
"""

import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "external_traffic": "External Traffic",
        "ingress_label": "Ingress",
        "ingress_sub": "TLS, routing rules",
        "service_label": "Service",
        "service_sub": "Load balancing",
        "deploy_v1": "Deployment: model-v1",
        "deploy_v2": "Deployment: model-v2",
        "pod_label": "Pod",
        "pod_sub": "vLLM",
        "gpu_label": "[GPU]",
        "hpa": "HPA",
        "nodes_title": "GPU Nodes (nvidia.com/gpu scheduling)",
        "node_prefix": "Node",
        "legend_ingress": "Ingress",
        "legend_service": "Service",
        "legend_deploy": "Deployment",
        "legend_pod": "Pod",
        "canary_stable": "90%",
        "canary_split": "     10%",
    },
    "zh": {
        "external_traffic": "外部请求流量",
        "ingress_label": "Ingress",
        "ingress_sub": "TLS 卸载、路由分发规则",
        "service_label": "Service",
        "service_sub": "内部服务负载均衡",
        "deploy_v1": "Deployment: model-v1",
        "deploy_v2": "Deployment: model-v2",
        "pod_label": "Pod",
        "pod_sub": "vLLM",
        "gpu_label": "[GPU]",
        "hpa": "HPA",
        "nodes_title": "GPU 物理/虚拟节点 (nvidia.com/gpu 资源调度)",
        "node_prefix": "节点",
        "legend_ingress": "Ingress 入口",
        "legend_service": "Service 服务",
        "legend_deploy": "Deployment 部署",
        "legend_pod": "Pod 副本",
        "canary_stable": "90%",
        "canary_split": "     10%",
    }
}


def draw_box(ax, x, y, w, h, label, sublabel=None, color='white', fontsize=12, border_color='#424242'):
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=color, edgecolor=border_color, linewidth=1.5
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + 0.12, label, ha='center', va='center', fontsize=fontsize, fontweight='bold')
        ax.text(x, y - 0.15, sublabel, ha='center', va='center', fontsize=11, style='italic')
    else:
        ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, fontweight='bold')


def draw_arrow(ax, start, end, color='#666666', style='-'):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5, linestyle=style))


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0.4, 11.6)
    ax.set_ylim(-0.2, 7.2)
    ax.set_aspect('equal')
    ax.axis('off')

    # Colors
    ingress_color = '#E3F2FD'
    service_color = '#FFF3E0'
    deploy_color = '#E8F5E9'
    pod_color = '#F3E5F5'
    node_color = '#FAFAFA'
    border_color = '#424242'
    k8s_blue = '#326CE5'

    # External traffic (no title - caption in markdown)
    ax.text(6, 7.1, text["external_traffic"], ha='center', va='center', fontsize=11, style='italic')
    draw_arrow(ax, (6, 6.9), (6, 6.5))

    # Ingress
    draw_box(ax, 6, 6.2, 3, 0.5, text["ingress_label"], text["ingress_sub"], color=ingress_color)

    # Arrow to Service
    draw_arrow(ax, (6, 5.95), (6, 5.55))

    # Service
    draw_box(ax, 6, 5.2, 2.5, 0.5, text["service_label"], text["service_sub"], color=service_color)

    # Arrows to Deployments
    draw_arrow(ax, (5.2, 4.95), (3.5, 4.55))
    draw_arrow(ax, (6.8, 4.95), (8.5, 4.55))

    # Deployment boxes (containing pods)
    # Left deployment
    deploy_left_x, deploy_left_y = 3.5, 3.5
    ax.add_patch(FancyBboxPatch(
        (deploy_left_x - 1.8, deploy_left_y - 1.3), 3.6, 2.3,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=deploy_color, edgecolor=border_color, linewidth=1.5, linestyle='--'
    ))
    ax.text(deploy_left_x, deploy_left_y + 0.9, text["deploy_v1"], ha='center', va='center', 
            fontsize=11, fontweight='bold')

    # Pods in left deployment
    draw_box(ax, 2.5, 3.2, 1.2, 0.7, text["pod_label"], text["pod_sub"], color=pod_color, fontsize=11)
    draw_box(ax, 4.5, 3.2, 1.2, 0.7, text["pod_label"], text["pod_sub"], color=pod_color, fontsize=11)
    ax.text(2.5, 2.6, text["gpu_label"], ha='center', va='center', fontsize=11, color='#666')
    ax.text(4.5, 2.6, text["gpu_label"], ha='center', va='center', fontsize=11, color='#666')

    # Right deployment
    deploy_right_x, deploy_right_y = 8.5, 3.5
    ax.add_patch(FancyBboxPatch(
        (deploy_right_x - 1.8, deploy_right_y - 1.3), 3.6, 2.3,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=deploy_color, edgecolor=border_color, linewidth=1.5, linestyle='--'
    ))
    ax.text(deploy_right_x, deploy_right_y + 0.9, text["deploy_v2"], ha='center', va='center', 
            fontsize=11, fontweight='bold')

    # Pods in right deployment
    draw_box(ax, 7.5, 3.2, 1.2, 0.7, text["pod_label"], text["pod_sub"], color=pod_color, fontsize=11)
    draw_box(ax, 9.5, 3.2, 1.2, 0.7, text["pod_label"], text["pod_sub"], color=pod_color, fontsize=11)
    ax.text(7.5, 2.6, text["gpu_label"], ha='center', va='center', fontsize=11, color='#666')
    ax.text(9.5, 2.6, text["gpu_label"], ha='center', va='center', fontsize=11, color='#666')

    # HPA annotations
    ax.text(1.2, 3.5, text["hpa"], ha='center', va='center', fontsize=11, fontweight='bold', color=k8s_blue)
    ax.annotate('', xy=(1.7, 3.5), xytext=(1.5, 3.5),
                arrowprops=dict(arrowstyle='->', color=k8s_blue, lw=1.2))
    ax.text(10.8, 3.5, text["hpa"], ha='center', va='center', fontsize=11, fontweight='bold', color=k8s_blue)
    ax.annotate('', xy=(10.3, 3.5), xytext=(10.5, 3.5),
                arrowprops=dict(arrowstyle='->', color=k8s_blue, lw=1.2))

    # Node layer
    ax.add_patch(FancyBboxPatch(
        (0.5, 0.3), 11, 1.5,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=node_color, edgecolor='#BDBDBD', linewidth=1, linestyle=':'
    ))
    ax.text(6, 1.4, text["nodes_title"], ha='center', va='center', 
            fontsize=11, style='italic', color='#666')

    # Node boxes
    for i, x in enumerate([2, 4.5, 7.5, 10]):
        ax.add_patch(FancyBboxPatch(
            (x - 0.6, 0.5), 1.2, 0.6,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            facecolor='white', edgecolor='#9E9E9E', linewidth=1
        ))
        ax.text(x, 0.8, f"{text['node_prefix']} {i+1}", ha='center', va='center', fontsize=11)

    # Legend
    legend_y = 0.0
    legend_items = [
        (1.5, text["legend_ingress"], ingress_color),
        (3.5, text["legend_service"], service_color),
        (5.5, text["legend_deploy"], deploy_color),
        (7.5, text["legend_pod"], pod_color),
    ]
    for x, label, color in legend_items:
        ax.add_patch(FancyBboxPatch(
            (x - 0.2, legend_y - 0.15), 0.3, 0.3,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor=color, edgecolor=border_color, linewidth=0.8
        ))
        ax.text(x + 0.3, legend_y, label, ha='left', va='center', fontsize=11)

    # Canary annotation
    ax.text(6, 4.7, text["canary_stable"], ha='right', va='center', fontsize=11, color='#666')
    ax.text(6, 4.7, text["canary_split"], ha='left', va='center', fontsize=11, color='#666')

    plt.tight_layout(pad=0.1)
    return fig


if __name__ == '__main__':
    localized_figure(draw, "k8s", LABELS, __file__, pad_inches=0, use_math_fonts=True)
