#!/usr/bin/env python3
"""
Benchmarking Methodology Flow Diagram
Shows the proper workflow for benchmarking distributed AI systems.

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
        "title": "Benchmarking Methodology Flow",
        "phase_setup": "Setup Phase",
        "setup_items": [
            "Fix random seeds",
            "Document hardware config",
            "Version control scripts",
            "Record system state"
        ],
        "phase_warmup": "Warmup Phase",
        "warmup_items": [
            "Run 10-20 warmup iterations",
            "JIT compilation completes",
            "Memory allocation stabilizes",
            "Caches warm up"
        ],
        "phase_measure": "Measurement Phase",
        "measure_items": [
            "torch.cuda.synchronize()",
            "Multiple iterations (100+)",
            "Record individual timings",
            "Multiple independent runs"
        ],
        "phase_analyze": "Analysis Phase",
        "analyze_items": [
            "Calculate mean, std, min, max",
            "Compute P50, P95, P99",
            "Statistical significance tests",
            "Generate reports"
        ],
        "pitfalls_title": "⚠ Common Pitfalls",
        "pitfalls_items": [
            "No warmup (cold start bias)",
            "Single measurement only",
            "Including data loading time"
        ],
        "best_title": "✓ Best Practices",
        "best_items": [
            "3-5 independent runs",
            "Report confidence intervals",
            "Document everything"
        ],
        "metrics_title": "Key Metrics",
        "metrics_items": [
            "Throughput (samples/s)",
            "Latency (P50/P95/P99)",
            "Scaling efficiency (%)"
        ]
    },
    "zh": {
        "title": "分布式基准评测规范方法流",
        "phase_setup": "环境准备阶段",
        "setup_items": [
            "固定全局随机种子",
            "记录硬件拓扑与配置",
            "评测脚本版本控制",
            "采集系统初始稳态"
        ],
        "phase_warmup": "模型预热阶段",
        "warmup_items": [
            "执行 10-20 次预热迭代",
            "等待 JIT 编译与算子调谐",
            "显存分配器池化水位稳定",
            "各级缓存填充预热"
        ],
        "phase_measure": "正式测量阶段",
        "measure_items": [
            "torch.cuda.synchronize()",
            "多轮次重复迭代（100+）",
            "记录每次独立耗时样本",
            "多轮独立冷启动验证"
        ],
        "phase_analyze": "数据分析阶段",
        "analyze_items": [
            "统计均值、方差、极值",
            "计算 P50、P95、P99 分位数",
            "显著性与置信度检验",
            "自动化生成评测报告"
        ],
        "pitfalls_title": "⚠ 常见评测陷阱",
        "pitfalls_items": [
            "未执行预热（冷启动偏差）",
            "单次测量缺乏统计置信度",
            "混入数据 I/O 阻塞时间"
        ],
        "best_title": "✓ 最佳工程实践",
        "best_items": [
            "进行 3-5 轮独立重复实验",
            "报告置信区间与标准差",
            "全量归档环境与运行元数据"
        ],
        "metrics_title": "核心评测指标",
        "metrics_items": [
            "吞吐量（samples/s）",
            "延迟分位数（P50/P95/P99）",
            "多卡扩展效率（%）"
        ]
    }
}


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, text["title"], fontsize=18, fontweight='bold',
            ha='center', va='center')

    # Colors
    colors = {
        'setup': '#E3F2FD',
        'warmup': '#FFF3E0',
        'measure': '#E8F5E9',
        'analyze': '#F3E5F5',
        'arrow': '#546E7A'
    }

    # Phase boxes - vertical flow
    phases = [
        {'name': text["phase_setup"], 'y': 8.2, 'color': colors['setup'],
         'items': text["setup_items"]},
        {'name': text["phase_warmup"], 'y': 6.2, 'color': colors['warmup'],
         'items': text["warmup_items"]},
        {'name': text["phase_measure"], 'y': 4.2, 'color': colors['measure'],
         'items': text["measure_items"]},
        {'name': text["phase_analyze"], 'y': 2.2, 'color': colors['analyze'],
         'items': text["analyze_items"]},
    ]

    for i, phase in enumerate(phases):
        # Main box
        box = FancyBboxPatch((1, phase['y'] - 0.8), 12, 1.6,
                             boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor=phase['color'], edgecolor='#37474F', linewidth=2)
        ax.add_patch(box)

        # Phase name
        ax.text(2, phase['y'] + 0.4, phase['name'], fontsize=14, fontweight='bold',
                ha='left', va='center', color='#1A237E')

        # Items
        for j, item in enumerate(phase['items']):
            x_pos = 2.5 + (j % 2) * 5.5
            y_pos = phase['y'] - 0.1 - (j // 2) * 0.4
            ax.text(x_pos, y_pos, f'• {item}', fontsize=10, ha='left', va='center', color='#37474F')

        # Arrow to next phase
        if i < len(phases) - 1:
            ax.annotate('', xy=(7, phase['y'] - 1.0), xytext=(7, phase['y'] - 0.8),
                       arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

    # Side annotations - Common Pitfalls
    pitfall_box = FancyBboxPatch((0.2, 0.3), 4.5, 1.5,
                                  boxstyle="round,pad=0.05,rounding_size=0.1",
                                  facecolor='#FFEBEE', edgecolor='#C62828', linewidth=2)
    ax.add_patch(pitfall_box)
    ax.text(2.45, 1.5, text["pitfalls_title"], fontsize=11, fontweight='bold',
            ha='center', va='center', color='#C62828')
    for idx, item in enumerate(text["pitfalls_items"]):
        y_pos = 1.1 - idx * 0.3
        ax.text(0.5, y_pos, f'• {item}', fontsize=9, ha='left', color='#37474F')

    # Best Practices box
    best_box = FancyBboxPatch((5.0, 0.3), 4.5, 1.5,
                               boxstyle="round,pad=0.05,rounding_size=0.1",
                               facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(best_box)
    ax.text(7.25, 1.5, text["best_title"], fontsize=11, fontweight='bold',
            ha='center', va='center', color='#2E7D32')
    for idx, item in enumerate(text["best_items"]):
        y_pos = 1.1 - idx * 0.3
        ax.text(5.3, y_pos, f'• {item}', fontsize=9, ha='left', color='#37474F')

    # Key Metrics box
    metrics_box = FancyBboxPatch((9.8, 0.3), 4.0, 1.5,
                                  boxstyle="round,pad=0.05,rounding_size=0.1",
                                  facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
    ax.add_patch(metrics_box)
    ax.text(11.8, 1.5, text["metrics_title"], fontsize=11, fontweight='bold',
            ha='center', va='center', color='#1565C0')
    for idx, item in enumerate(text["metrics_items"]):
        y_pos = 1.1 - idx * 0.3
        ax.text(10.1, y_pos, f'• {item}', fontsize=9, ha='left', color='#37474F')

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    localized_figure(draw, "benchmarking_methodology", LABELS, __file__)
