#!/usr/bin/env python3
"""
Canary Deployment & Automated Rollback Architecture Diagram
Chapter 9: Production LLM Serving Stack

Follows ~/mmb's localized_figure standard:
- Single implementation, multiple outputs (<stem>.png for English, <stem>_zh.png for Chinese)
- High-resolution (300 DPI) PNG and vector PDF exports
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
        "ingress_title": "Production Ingress Traffic",
        "ingress_sub": "HTTP / gRPC Requests (100% Live Workload)",
        "gw_title": "API Gateway / Traffic Shifter (Dynamic Router)",
        "gw_sub": "Gradual Traffic Shifting & Automated Weight Adjustment",
        "gw_d1": "• Traffic Stages: 10% → 25% → 50% → 100%",
        "gw_d2": "• Weighted Round-Robin / Header & Hash Match",
        "stable_title": "Stable Service (v1.0)",
        "stable_sub": "Current Production Baseline",
        "stable_d1": "• vLLM Pod Replicas (90% Traffic)",
        "stable_d2": "• Baseline P99 Latency: ~120 ms",
        "stable_d3": "• Baseline HTTP 5xx: < 0.05%",
        "stable_split": "90% Live Traffic",
        "stable_telem": "Stable Telemetry",
        "canary_title": "Canary Service (v2.0)",
        "canary_sub": "New Weights / Engine Update",
        "canary_d1": "• vLLM Pod Replicas (10% Traffic)",
        "canary_d2": "• Candidate P99 & TPOT Evaluation",
        "canary_d3": "• Active Safety Gate Probing",
        "canary_split": "10% Canary Split",
        "canary_telem": "Canary Telemetry",
        "gate_title": "Real-Time Observability & Safety Gate Controller",
        "gate_sub": "Prometheus, OpenTelemetry & Automated Decision Engine",
        "gate_d1": "• Metric Comparison: ΔP99 Latency & Error Rate Ratio",
        "gate_d2": "• Safety Condition: Error(Canary) < 2.0 × Error(Stable)",
        "gate_d3": "• Promotion Decision: Promote to Next Step if healthy for N min",
        "promote": "Healthy:\nPromote\nTraffic +25%",
        "rollback": "[Circuit Breaker]\nAnomaly Detected:\nInstant Rollback\nCanary → 0%",
    },
    "zh": {
        "ingress_title": "生产环境入站流量",
        "ingress_sub": "HTTP / gRPC 外部请求 (100% 全量生产负载)",
        "gw_title": "API 网关 / 流量切分器（动态路由器）",
        "gw_sub": "渐进式流量切分与权重自动化调谐",
        "gw_d1": "• 流量切分阶梯：10% → 25% → 50% → 100%",
        "gw_d2": "• 加权轮询调度 / 请求头与哈希一致性匹配",
        "stable_title": "稳定版服务集群 (v1.0)",
        "stable_sub": "当前生产基准运行版本",
        "stable_d1": "• vLLM Pod 副本池 (承载 90% 生产流量)",
        "stable_d2": "• 基准 P99 延迟水位：~120 ms",
        "stable_d3": "• 基准 HTTP 5xx 错误率：< 0.05%",
        "stable_split": "90% 稳定版流量",
        "stable_telem": "稳定版遥测指标",
        "canary_title": "金丝雀服务集群 (v2.0)",
        "canary_sub": "新模型权重 / 新推理引擎版本升级",
        "canary_d1": "• vLLM Pod 副本池 (承载 10% 试运行流量)",
        "canary_d2": "• 候选版 P99 延迟与 TPOT 实时评估",
        "canary_d3": "• 活跃安全门禁自动化探针监测",
        "canary_split": "10% 金丝雀分流",
        "canary_telem": "金丝雀遥测指标",
        "gate_title": "全栈实时可观测性体系与安全门禁控制器",
        "gate_sub": "Prometheus、OpenTelemetry 与自动化熔断决策引擎",
        "gate_d1": "• 核心指标实时比对：ΔP99 延迟增量与 5xx 错误率倍数",
        "gate_d2": "• 安全门禁熔断阈值：Error(Canary) < 2.0 × Error(Stable)",
        "gate_d3": "• 自动晋级决策：指标连续 N 分钟符合 SLA 自动进入下一阶段",
        "promote": "指标正常：\n自动晋级\n流量 +25%",
        "rollback": "【安全熔断回滚】\n检测到指标异常：\n秒级自动熔断\n金丝雀流量归零",
    }
}


def draw_box(ax, x, y, width, height, label, sublabel=None, details=None,
             color='white', border='#424242', linestyle='-', linewidth=1.5):
    """Draw a rounded box with title, subtitle, and optional bullet points."""
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=color,
        edgecolor=border,
        linewidth=linewidth,
        linestyle=linestyle
    )
    ax.add_patch(box)

    if sublabel and not details:
        ax.text(x, y + height * 0.20, label, ha='center', va='center',
                fontsize=11.5, fontweight='bold', color=border)
        ax.text(x, y - height * 0.20, sublabel, ha='center', va='center',
                fontsize=9.5, style='italic', color='#374151')
    elif details:
        ax.text(x, y + height * 0.32, label, ha='center', va='center',
                fontsize=11.5, fontweight='bold', color=border)
        if sublabel:
            ax.text(x, y + height * 0.15, sublabel, ha='center', va='center',
                    fontsize=9.5, style='italic', color='#4B5563')
        for i, line in enumerate(details):
            offset_y = y - height * (0.06 + i * 0.17)
            ax.text(x, offset_y, line, ha='center', va='center',
                    fontsize=9.0, color='#1F2937')
    else:
        ax.text(x, y, label, ha='center', va='center',
                fontsize=11.5, fontweight='bold', color=border)


def draw_arrow(ax, start, end, label=None, color='#424242', linestyle='-', lw=1.5,
               label_pos=None, label_color=None, label_size=9.0, rad=0.0):
    """Draw an arrow between two points with optional text label."""
    conn = f'arc3,rad={rad}'
    ax.annotate(
        '', xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle='->',
            color=color,
            lw=lw,
            linestyle=linestyle,
            connectionstyle=conn
        )
    )
    if label and label_pos:
        l_color = label_color or color
        ax.text(label_pos[0], label_pos[1], label, ha='center', va='center',
                fontsize=label_size, fontweight='bold', color=l_color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=l_color, alpha=0.95, lw=0.8))


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(0.2, 8.3)
    ax.set_aspect('equal')
    ax.axis('off')

    # Color definitions
    traffic_color = '#E3F2FD'   # Soft light blue
    gateway_color = '#FFF3E0'   # Soft light orange
    stable_color = '#E8F5E9'    # Soft light green
    canary_color = '#FFF8E1'    # Soft light amber
    metrics_color = '#EDE7F6'   # Soft light purple
    rollback_color = '#FFEBEE'  # Soft light red

    # 1. Top: Production Traffic Ingress
    draw_box(ax, 5.0, 7.6, 4.6, 0.65,
             text["ingress_title"],
             sublabel=text["ingress_sub"],
             color=traffic_color, border='#1976D2')

    # Arrow from Traffic to Gateway
    draw_arrow(ax, (5.0, 7.27), (5.0, 6.65), lw=2, color='#1976D2')

    # 2. Traffic Router / API Gateway
    draw_box(ax, 5.0, 6.05, 6.8, 1.1,
             text["gw_title"],
             sublabel=text["gw_sub"],
             details=[text["gw_d1"], text["gw_d2"]],
             color=gateway_color, border='#E65100')

    # 3. Middle Left: Stable Fleet (v1.0)
    draw_box(ax, 2.8, 4.0, 3.4, 1.5,
             text["stable_title"],
             sublabel=text["stable_sub"],
             details=[text["stable_d1"], text["stable_d2"], text["stable_d3"]],
             color=stable_color, border='#2E7D32')

    # Vertical Arrow from Gateway down to Stable
    draw_arrow(ax, (2.8, 5.5), (2.8, 4.75), color='#2E7D32', lw=2)
    ax.text(2.0, 5.12, text["stable_split"], ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#2E7D32',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='#2E7D32', alpha=0.95, lw=0.8))

    # 4. Middle Right: Canary Fleet (v2.0)
    draw_box(ax, 7.2, 4.0, 3.4, 1.5,
             text["canary_title"],
             sublabel=text["canary_sub"],
             details=[text["canary_d1"], text["canary_d2"], text["canary_d3"]],
             color=canary_color, border='#F57F17')

    # Vertical Arrow from Gateway down to Canary
    draw_arrow(ax, (7.2, 5.5), (7.2, 4.75), color='#F57F17', lw=2)
    ax.text(8.0, 5.12, text["canary_split"], ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#F57F17',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='#F57F17', alpha=0.95, lw=0.8))

    # 5. Bottom: Observability & Automated Safety Gate
    draw_box(ax, 5.0, 1.85, 6.8, 1.4,
             text["gate_title"],
             sublabel=text["gate_sub"],
             details=[text["gate_d1"], text["gate_d2"], text["gate_d3"]],
             color=metrics_color, border='#512DA8')

    # Vertical Telemetry arrows feeding into Safety Gate
    draw_arrow(ax, (2.8, 3.25), (2.8, 2.55),
               color='#2E7D32', linestyle='--', lw=1.5)
    ax.text(2.8, 2.9, text["stable_telem"], ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#2E7D32',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='#2E7D32', alpha=0.95, lw=0.6))

    draw_arrow(ax, (7.2, 3.25), (7.2, 2.55),
               color='#F57F17', linestyle='--', lw=1.5)
    ax.text(7.2, 2.9, text["canary_telem"], ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#F57F17',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='#F57F17', alpha=0.95, lw=0.6))

    # 6. Promotion Loop (Clean outer path on the right side)
    draw_arrow(ax, (8.4, 1.85), (8.4, 6.05), rad=0.45,
               color='#2E7D32', linestyle='-.', lw=1.8)
    ax.text(9.5, 4.0, text["promote"],
            ha='center', va='center', fontsize=8.5, fontweight='bold',
            color='#2E7D32',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='#2E7D32', alpha=0.95, lw=1.0))

    # 7. Rollback Loop (Clean outer path on the left side)
    draw_arrow(ax, (1.6, 1.85), (1.6, 6.05), rad=-0.45,
               color='#C62828', linestyle='--', lw=2.2)
    ax.text(0.5, 4.0, text["rollback"],
            ha='center', va='center', fontsize=8.2, fontweight='bold',
            color='#C62828',
            bbox=dict(boxstyle='round,pad=0.25', facecolor=rollback_color,
                      edgecolor='#C62828', alpha=0.95, lw=1.2))

    plt.tight_layout(pad=0.1)
    return fig


if __name__ == '__main__':
    localized_figure(draw, "canary_deployment", LABELS, __file__)
