"""
SGLang Architecture Diagram

Shows the flow of requests through SGLang's runtime:
- Clients (SGLang Program + HTTP Client)
- API Server (entry point)
- SGLang Runtime (SRT) with Tokenizer, Request Queue, Scheduler, GPU Workers, Detokenizer
- API Server (returns responses)
"""
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Add shared directory to path for figstyle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "clients_header": "Clients",
        "program": "SGLang Program\n+ Interpreter",
        "http_client": "HTTP Client",
        "api_server": "API Server",
        "entry_point": "Entry point",
        "srt_header": "SGLang Runtime (SRT)",
        "tokenizer": "Tokenizer",
        "text_to_tokens": r"Text $\to$ Tokens",
        "request_queue": "Request Queue",
        "batches_reqs": "Batches requests",
        "scheduler": "Scheduler\nRadixAttention",
        "intelligent_batching": "Intelligent batching",
        "gpu_workers": r"GPU Workers" + "\n" + r"W0 $\rightarrow$ W1 $\rightarrow$ W2 $\rightarrow$ W3",
        "model_exec": "Model execution",
        "detokenizer": "Detokenizer",
        "tokens_to_text": r"Tokens $\to$ Text",
        "returns_resp": "Returns responses",
    },
    "zh": {
        "clients_header": "客户端 (Clients)",
        "program": "SGLang 程序\n+ 解释器",
        "http_client": "HTTP 客户端",
        "api_server": "API Server",
        "entry_point": "统一入口",
        "srt_header": "SGLang 运行时 (SRT)",
        "tokenizer": "分词器 (Tokenizer)",
        "text_to_tokens": r"文本 $\to$ Tokens",
        "request_queue": "请求队列 (Request Queue)",
        "batches_reqs": "批量聚合请求",
        "scheduler": "调度器\nRadixAttention",
        "intelligent_batching": "智能批处理",
        "gpu_workers": r"GPU 工作节点 (Workers)" + "\n" + r"W0 $\rightarrow$ W1 $\rightarrow$ W2 $\rightarrow$ W3",
        "model_exec": "模型推理执行",
        "detokenizer": "反分词器 (Detokenizer)",
        "tokens_to_text": r"Tokens $\to$ 文本",
        "returns_resp": "返回流式响应",
    }
}

# Colors
client_color = '#E8F4FD'
server_color = '#FFF3E0'
runtime_color = '#E8F5E9'
component_color = '#FFFFFF'
gpu_color = '#FFECB3'
arrow_color = '#666666'


def draw_box(ax, x, y, width, height, label, facecolor='white', edgecolor='black', fontsize=14, bold=False):
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', fontsize=fontsize, weight=weight)


def draw_arrow(ax, start, end):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))


def draw(text: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.set_aspect('equal')
    ax.axis('off')

    # Clients section (top)
    ax.add_patch(FancyBboxPatch((1, 11.5), 8, 2, boxstyle="round,pad=0.02,rounding_size=0.2",
                                facecolor=client_color, edgecolor='#1976D2', linewidth=2))
    ax.text(5, 13.2, text['clients_header'], ha='center', va='center', fontsize=14, weight='bold', color='#1976D2')

    # Client boxes inside
    draw_box(ax, 1.5, 11.8, 3, 1.2, text['program'], facecolor=component_color, edgecolor='#1976D2', fontsize=14)
    draw_box(ax, 5.5, 11.8, 3, 1.2, text['http_client'], facecolor=component_color, edgecolor='#1976D2', fontsize=14)

    # Arrow from clients to API Server
    draw_arrow(ax, (5, 11.5), (5, 10.7-0.1))

    # API Server (entry)
    draw_box(ax, 2.5, 9.8, 5, 0.8, text['api_server'], facecolor=server_color, edgecolor='#F57C00', fontsize=14, bold=True)
    ax.text(7.8, 10.2, text['entry_point'], ha='left', va='center', fontsize=14, color='#666666', style='italic')

    # Arrow to SRT
    draw_arrow(ax, (5, 9.8), (5, 9.3-0.1))

    # SGLang Runtime (SRT) - main box
    ax.add_patch(FancyBboxPatch((1.5, 2.2), 7, 7, boxstyle="round,pad=0.02,rounding_size=0.2",
                                facecolor=runtime_color, edgecolor='#388E3C', linewidth=2))
    ax.text(5, 9.0-0.1, text['srt_header'], ha='center', va='center', fontsize=14, weight='bold', color='#388E3C')

    # Tokenizer
    draw_box(ax, 3.5, 7.8, 3, 0.7, text['tokenizer'], facecolor=component_color, edgecolor='#388E3C', fontsize=14)
    ax.text(6.8, 8.15, text['text_to_tokens'], ha='left', va='center', fontsize=14, color='#666666', style='italic')

    draw_arrow(ax, (5, 7.8), (5, 7.3-0.1))

    # Request Queue
    draw_box(ax, 3.5, 6.5, 3, 0.7, text['request_queue'], facecolor=component_color, edgecolor='#388E3C', fontsize=14)
    ax.text(6.8, 6.85, text['batches_reqs'], ha='left', va='center', fontsize=14, color='#666666', style='italic')

    draw_arrow(ax, (5, 6.5), (5, 6.0-0.1))

    # Scheduler
    draw_box(ax, 3.5, 5.0, 3, 0.9, text['scheduler'], facecolor=component_color, edgecolor='#388E3C', fontsize=14)
    ax.text(6.8, 5.45, text['intelligent_batching'], ha='left', va='center', fontsize=14, color='#666666', style='italic')

    draw_arrow(ax, (5, 5.0), (5, 4.5-0.1))

    # GPU Workers
    draw_box(ax, 2.5, 3.4, 5, 1.0, text['gpu_workers'], facecolor=gpu_color, edgecolor='#FFA000', fontsize=14)
    ax.text(7.8, 3.9, text['model_exec'], ha='left', va='center', fontsize=14, color='#666666', style='italic')

    draw_arrow(ax, (5, 3.4), (5, 2.9+0.15))

    # Detokenizer
    draw_box(ax, 3.5, 2.4, 3, 0.7, text['detokenizer'], facecolor=component_color, edgecolor='#388E3C', fontsize=14)
    ax.text(6.8, 2.75, text['tokens_to_text'], ha='left', va='center', fontsize=14, color='#666666', style='italic')

    # Arrow from SRT to API Server (bottom)
    draw_arrow(ax, (5, 2.2), (5, 1.7-0.1))

    # API Server (exit)
    draw_box(ax, 2.5, 0.8, 5, 0.8, text['api_server'], facecolor=server_color, edgecolor='#F57C00', fontsize=14, bold=True)
    ax.text(7.8, 1.2, text['returns_resp'], ha='left', va='center', fontsize=14, color='#666666', style='italic')

    plt.tight_layout(pad=0.1)
    return fig


if __name__ == '__main__':
    localized_figure(draw, "sglang_architecture", LABELS, __file__, use_math_fonts=True)
