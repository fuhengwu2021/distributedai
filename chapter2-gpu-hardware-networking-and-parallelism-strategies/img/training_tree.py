import os
import sys
from graphviz import Digraph

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "fontname": "Helvetica",
        "S1": "Does a full model replica\nfit on one device?",
        "DDP": "Replicated Data Parallelism (DDP)\n(Simple starting point)",
        "S2": "Shard parameters, gradients,\nand optimizer states?",
        "FSDP": "Sharded Data Parallelism\n(FSDP / ZeRO-3)",
        "S3": "Is computation of one\nsample split across devices?",
        "TP": "Tensor Parallelism\n(Heads/Hidden dimensions)",
        "SP": "Sequence Parallelism\n(Long sequences)",
        "PP": "Pipeline Parallelism\n(Layer/Stage splitting)",
        "EP": "Expert Parallelism\n(MoE Models)",
        "S4": "Still need more scale?\n(Hybrid Combinations)",
        "Hybrid": "3D Parallelism\n(DP + TP + PP)",
        "S5": "Memory still insufficient?",
        "Opt": '<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="2"><TR><TD ALIGN="LEFT">System-level Optimizations:</TD></TR><TR><TD ALIGN="LEFT">• Activation Checkpointing</TD></TR><TR><TD ALIGN="LEFT">• CPU/NVMe Offloading</TD></TR></TABLE>>',
        "edge_yes": "Yes",
        "edge_no": "No",
        "edge_fsdp_s3": "Still memory limited",
        "edge_tp": "Large Layers",
        "edge_sp": "Long Sequence",
        "edge_pp": "Deep Models",
        "edge_ep": "Sparse/MoE",
    },
    "zh": {
        "fontname": "Noto Sans CJK SC",
        "edge_fontname": "Noto Sans CJK SC",
        "S1": "单卡显存能否容纳\n完整模型副本？",
        "DDP": "多副本数据并行 (DDP)\n(简单高效的首选起点)",
        "S2": "对参数、梯度及\n优化器状态进行分片？",
        "FSDP": "状态分片数据并行\n(FSDP / ZeRO-3)",
        "S3": "单个样本的计算\n是否切分到多卡？",
        "TP": "张量并行 (TP)\n(注意力头 / 隐藏层切分)",
        "SP": "序列并行 (SP)\n(长上下文序列切分)",
        "PP": "流水线并行 (PP)\n(模型层 / Stage 切分)",
        "EP": "专家并行 (EP)\n(稀疏 MoE 模型)",
        "S4": "仍需更大规模扩展？\n(混合并行组合)",
        "Hybrid": "3D 混合并行\n(DP + TP + PP)",
        "S5": "显存依然不足？",
        "Opt": '<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="2"><TR><TD ALIGN="LEFT">系统级极限优化：</TD></TR><TR><TD ALIGN="LEFT">• 激活值检查点 (Activation Checkpointing)</TD></TR><TR><TD ALIGN="LEFT">• CPU/NVMe 状态卸载 (Offloading)</TD></TR></TABLE>>',
        "edge_yes": "是",
        "edge_no": "否",
        "edge_fsdp_s3": "显存仍受限",
        "edge_tp": "单层超大 (Large Layers)",
        "edge_sp": "长序列 (Long Sequence)",
        "edge_pp": "超深模型 (Deep Models)",
        "edge_ep": "稀疏/MoE",
    }
}


def draw(text: dict) -> Digraph:
    dot = Digraph(comment='Training Strategy Decision Tree')
    dot.attr(rankdir='TD', size='12', ranksep='1.0', nodesep='.5', dpi='300')

    # Global Node Styles
    dot.attr('node', 
             shape='box', 
             style='rounded,filled', 
             fillcolor='#E3F2FD:#BBDEFB',
             fontname=text["fontname"],
             fontsize='16',
             penwidth='1.5',
             color='#1976D2',
             gradientangle='90')

    if "edge_fontname" in text:
        dot.attr('edge', fontname=text["edge_fontname"])

    # Result Node Style (Green)
    result_style = {'fillcolor': '#A5D6A7:#C8E6C9', 'color': '#2E7D32'}

    # --- Nodes ---
    # Step 1: Fitting
    dot.node('S1', text['S1'])

    # Replicated Data Parallelism
    dot.node('DDP', text['DDP'], **result_style)

    # Step 2: Sharding
    dot.node('S2', text['S2'])

    # Sharded Data Parallelism
    dot.node('FSDP', text['FSDP'], **result_style)

    # Step 3: True Model Parallelism
    dot.node('S3', text['S3'])

    # Split types
    dot.node('TP', text['TP'], **result_style)
    dot.node('SP', text['SP'], **result_style)
    dot.node('PP', text['PP'], **result_style)
    dot.node('EP', text['EP'], **result_style)

    # Step 4: Hybrid
    dot.node('S4', text['S4'])
    dot.node('Hybrid', text['Hybrid'], **result_style)

    # Step 5: Optimizations
    dot.node('S5', text['S5'])
    dot.node('Opt', text['Opt'], **result_style)

    # --- Edges ---
    # Step 1 edges
    dot.edge('S1', 'DDP', label=text['edge_yes'])
    dot.edge('S1', 'S2', label=text['edge_no'])

    # Step 2 edges
    dot.edge('S2', 'FSDP')
    dot.edge('FSDP', 'S3', label=text['edge_fsdp_s3'])

    # Step 3 edges - splitting logic
    dot.edge('S3', 'TP', label=text['edge_tp'])
    dot.edge('S3', 'SP', label=text['edge_sp'])
    dot.edge('S3', 'PP', label=text['edge_pp'])
    dot.edge('S3', 'EP', label=text['edge_ep'])

    # Step 4 edges - hybrid combinations
    dot.edge('TP', 'S4')
    dot.edge('PP', 'S4')
    dot.edge('S4', 'Hybrid')
    dot.edge('Hybrid', 'S5')

    # Step 5 edges - optimizations
    dot.edge('S5', 'Opt', label=text['edge_yes'])

    return dot


if __name__ == '__main__':
    localized_figure(draw, "training_tree", LABELS, __file__)
