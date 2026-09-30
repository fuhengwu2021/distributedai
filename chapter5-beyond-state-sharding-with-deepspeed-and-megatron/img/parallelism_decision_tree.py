import os
import sys
from graphviz import Digraph

# Ensure shared directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "fontname": "Helvetica",
        "yes": "Yes",
        "no": "No",
        "q1": "Model Size < 10B?",
        "q2": "Model Size < 50B?",
        "q3": "Single layer fits on one GPU?",
        "q4": "Sequence Length >= 8K?",
        "q5": "Model Size < 200B?",
        "q6": "Multiple Nodes?",
        "q7": "MoE Model?",
        "a1": "DDP or ZeRO-1\n(simplest, fastest)",
        "a2": "ZeRO-2 or FSDP2\n(shards gradients)",
        "a3": "ZeRO-3 or FSDP2\n(full state sharding)",
        "a4": "FSDP2 + TP + CP\n(long sequences)",
        "a5": "FSDP2 + TP\n(50B-200B models)",
        "a6": "FSDP2 + TP + PP\n(multi-node scaling)",
        "a7": "FSDP2 + TP + EP\n(MoE models)",
        "a8": "FSDP2 + TP + PP\n(single node, >200B)",
    },
    "zh": {
        "fontname": "Noto Sans CJK SC",
        "yes": "是 (Yes)",
        "no": "否 (No)",
        "q1": "模型参数规模 < 10B？",
        "q2": "模型参数规模 < 50B？",
        "q3": "单个网络层是否能放入单张 GPU？",
        "q4": "序列长度（Seq Len）>= 8K？",
        "q5": "模型参数规模 < 200B？",
        "q6": "是否采用多机多卡拓扑（Multi-Nodes）？",
        "q7": "是否为专家混合模型（MoE）？",
        "a1": "DDP 或 ZeRO-1\n（最简单、训练吞吐最高）",
        "a2": "ZeRO-2 或 FSDP2\n（对梯度进行分片）",
        "a3": "ZeRO-3 或 FSDP2\n（完全状态分片存储）",
        "a4": "FSDP2 + TP + CP\n（超长上下文序列并行）",
        "a5": "FSDP2 + TP\n（50B-200B 中大规模模型）",
        "a6": "FSDP2 + TP + PP\n（多机扩展，压低通信带宽）",
        "a7": "FSDP2 + TP + EP\n（稀疏激活 MoE 模型）",
        "a8": "FSDP2 + TP + PP\n（单节点超大模型，>200B）",
    },
}


def draw(text: dict) -> Digraph:
    # Create a directed graph
    dot = Digraph(comment='Parallelism Strategy Decision Tree')
    dot.attr(rankdir='TD', size='12', ranksep='0.8', nodesep='0.3', dpi='300')
    dot.attr('edge', fontname=text['fontname'], fontsize='12')

    # Define node styles
    # Question nodes - blue gradient
    dot.attr('node', 
             shape='box', 
             style='rounded,filled', 
             fillcolor='#E3F2FD:#BBDEFB',
             fontname=text['fontname'],
             fontsize='14',
             penwidth='1.5',
             color='#1976D2',
             margin='0.15',
             gradientangle='90')

    # --- Question Nodes ---
    dot.node('Q1', text['q1'])
    dot.node('Q2', text['q2'])
    dot.node('Q3', text['q3'])
    dot.node('Q4', text['q4'])
    dot.node('Q5', text['q5'])
    dot.node('Q6', text['q6'])
    dot.node('Q7', text['q7'])

    # --- Answer Nodes (Green - recommended strategies) ---
    green_style = {'fillcolor': '#A5D6A7:#C8E6C9', 'color': '#2E7D32'}

    dot.node('A1', text['a1'], **green_style)
    dot.node('A2', text['a2'], **green_style)
    dot.node('A3', text['a3'], **green_style)
    dot.node('A4', text['a4'], **green_style)
    dot.node('A5', text['a5'], **green_style)
    dot.node('A6', text['a6'], **green_style)
    dot.node('A7', text['a7'], **green_style)
    dot.node('A8', text['a8'], **green_style)

    # --- Edges ---
    # Q1: Model Size < 10B?
    dot.edge('Q1', 'A1', label=text['yes'])
    dot.edge('Q1', 'Q2', label=text['no'])

    # Q2: Model Size < 50B?
    dot.edge('Q2', 'A2', label=text['yes'])
    dot.edge('Q2', 'Q3', label=text['no'])

    # Q3: Single layer fits on one GPU?
    dot.edge('Q3', 'A3', label=text['yes'])
    dot.edge('Q3', 'Q4', label=text['no'])

    # Q4: Sequence Length >= 8K?
    dot.edge('Q4', 'A4', label=text['yes'])
    dot.edge('Q4', 'Q5', label=text['no'])

    # Q5: Model Size < 200B?
    dot.edge('Q5', 'A5', label=text['yes'])
    dot.edge('Q5', 'Q6', label=text['no'])

    # Q6: Multiple Nodes?
    dot.edge('Q6', 'A6', label=text['yes'])
    dot.edge('Q6', 'Q7', label=text['no'])

    # Q7: MoE Model?
    dot.edge('Q7', 'A7', label=text['yes'])
    dot.edge('Q7', 'A8', label=text['no'])

    return dot


if __name__ == '__main__':
    localized_figure(draw, "parallelism_decision_tree", LABELS, __file__)
