#!/usr/bin/env python3
"""
Decision framework for choosing distributed training and inference strategies.
Flowchart visualizing when to scale from single-GPU to distributed setups.
"""

import os
import sys
from graphviz import Digraph

# Ensure shared directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "fontname": "Helvetica",
        "A": "Start: What is your use case",
        "B": "Training or Fine tuning",
        "B1": "Model exceeds single GPU memory?",
        "B1Y": "Distributed Training\n(Model/Parameter Parallelism)",
        "B2": "Training time too long?",
        "B2Y": "Distributed Training\n(Data Parallelism)",
        "B3": "Fine tuning with LoRA/QLoRA?",
        "B3Y": "Single GPU",
        "B4": "Large dataset?",
        "B4Y": "Consider Distributed Setup",
        "B4N": "Single GPU",
        "D": "Inference or Serving",
        "D1": "Model exceeds single GPU memory?",
        "D1Y": "Distributed Inference\n(Model Parallelism)",
        "D2": "High throughput required?",
        "D2Y": "Distributed Inference\n(Multiple GPUs)",
        "D2N": "Single GPU",
        "yes": "Yes",
        "no": "No",
    },
    "zh": {
        "fontname": "Noto Sans CJK SC",
        "A": "开始：你的使用场景是什么？",
        "B": "模型训练或微调",
        "B1": "模型超出单卡 GPU 显存？",
        "B1Y": "分布式训练\n（模型/参数并行）",
        "B2": "训练耗时过长？",
        "B2Y": "分布式训练\n（数据并行）",
        "B3": "是否使用 LoRA/QLoRA 微调？",
        "B3Y": "单卡 GPU",
        "B4": "数据集规模巨大？",
        "B4Y": "考虑分布式配置",
        "B4N": "单卡 GPU",
        "D": "模型推理或在线服务",
        "D1": "模型超出单卡 GPU 显存？",
        "D1Y": "分布式推理\n（模型并行）",
        "D2": "需要高吞吐并发服务？",
        "D2Y": "分布式推理\n（多 GPU）",
        "D2N": "单卡 GPU",
        "yes": "是",
        "no": "否",
    }
}


def draw(text: dict) -> Digraph:
    fontname = text["fontname"]
    dot = Digraph(comment='GPU Usage Flowchart')
    dot.attr(rankdir='TD', size='10', ranksep='1.5', nodesep='.1', dpi='300')

    # Define node styles with rounded corners and gradient fills
    dot.attr('node', 
             shape='box', 
             style='rounded,filled', 
             fillcolor='#E3F2FD:#BBDEFB',
             fontname=fontname,
             fontsize='16',
             penwidth='1.5',
             color='#1976D2',
             margin='0.05',
             gradientangle='90')

    # Define edge font
    dot.attr('edge', fontname=fontname, fontsize='14')

    # --- Nodes ---
    dot.node('A', text["A"])

    # Training Branch
    dot.node('B', text["B"])
    dot.node('B1', text["B1"])
    dot.node('B1Y', text["B1Y"], fillcolor='#A5D6A7:#C8E6C9', color='#2E7D32', gradientangle='90')
    dot.node('B2', text["B2"])
    dot.node('B2Y', text["B2Y"], fillcolor='#A5D6A7:#C8E6C9', color='#2E7D32', gradientangle='90')
    dot.node('B3', text["B3"])
    dot.node('B3Y', text["B3Y"], fillcolor='#FFF59D:#FFF9C4', color='#F57F17', gradientangle='90')
    dot.node('B4', text["B4"])
    dot.node('B4Y', text["B4Y"], fillcolor='#A5D6A7:#C8E6C9', color='#2E7D32', gradientangle='90')
    dot.node('B4N', text["B4N"], fillcolor='#FFF59D:#FFF9C4', color='#F57F17', gradientangle='90')

    # Inference Branch
    dot.node('D', text["D"])
    dot.node('D1', text["D1"])
    dot.node('D1Y', text["D1Y"], fillcolor='#A5D6A7:#C8E6C9', color='#2E7D32', gradientangle='90')
    dot.node('D2', text["D2"])
    dot.node('D2Y', text["D2Y"], fillcolor='#A5D6A7:#C8E6C9', color='#2E7D32', gradientangle='90')
    dot.node('D2N', text["D2N"], fillcolor='#FFF59D:#FFF9C4', color='#F57F17', gradientangle='90')

    # --- Edges ---
    dot.edge('A', 'B')
    dot.edge('A', 'D')

    # Training Logic
    dot.edge('B', 'B1')
    dot.edge('B1', 'B1Y', label=text["yes"])
    dot.edge('B1', 'B2', label=text["no"])
    dot.edge('B2', 'B2Y', label=text["yes"])
    dot.edge('B2', 'B3', label=text["no"])
    dot.edge('B3', 'B3Y', label=text["yes"])
    dot.edge('B3', 'B4', label=text["no"])
    dot.edge('B4', 'B4Y', label=text["yes"])
    dot.edge('B4', 'B4N', label=text["no"])

    # Inference Logic
    dot.edge('D', 'D1')
    dot.edge('D1', 'D1Y', label=text["yes"])
    dot.edge('D1', 'D2', label=text["no"])
    dot.edge('D2', 'D2Y', label=text["yes"])
    dot.edge('D2', 'D2N', label=text["no"])

    return dot


if __name__ == '__main__':
    localized_figure(draw, "decision_tree", LABELS, __file__)
