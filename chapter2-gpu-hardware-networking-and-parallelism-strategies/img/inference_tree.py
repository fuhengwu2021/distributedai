import os
import sys
from graphviz import Digraph

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from figstyle import localized_figure

LABELS = {
    "en": {
        "fontname": "Helvetica",
        "S1": "Are you scaling a Single Request \nor Multiple Requests?",
        "S1_Multi": '<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="2"><TR><TD ALIGN="LEFT">Request-Level Parallelism:</TD></TR><TR><TD ALIGN="LEFT">• Batching</TD></TR><TR><TD ALIGN="LEFT">• Model Replicas</TD></TR><TR><TD ALIGN="LEFT">• Load Balancing</TD></TR></TABLE>>',
        "S2": "Does the model computation \nfit on one device?",
        "S2_Yes": '<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="2"><TR><TD ALIGN="LEFT">Single-GPU Optimized:</TD></TR><TR><TD ALIGN="LEFT">• FlashAttention</TD></TR><TR><TD ALIGN="LEFT">• Quantization (INT8/4)</TD></TR><TR><TD ALIGN="LEFT">• Kernel Fusion</TD></TR></TABLE>>',
        "S3": "How is computation split?",
        "S3_Tensor": "Tensor Parallelism\n(vLLM / TRT-LLM)",
        "S3_Context": "Context Parallelism\n(Long-context / KV split)",
        "S3_Pipe": "Pipeline Parallelism\n(Layer-wise split)",
        "S3_Expert": "Expert Parallelism\n(MoE Models)",
        "S4": "Is Memory or KV Cache \nthe bottleneck?",
        "S4_Paged": "PagedAttention\n(Virtualize KV Cache)",
        "S4_Offload": "Offloading\n(CPU/NVMe/Disaggregation)",
        "edge_s1_single": "Single/Large",
        "edge_s1_multi": "Multiple",
        "edge_s2_yes": "Yes",
        "edge_s2_no": "No",
        "edge_s3_heads": "Heads/Hidden",
        "edge_s3_context": "Long Context",
        "edge_s3_pipe": "Layers",
        "edge_s3_expert": "MoE",
        "edge_s4_cap": "Capacity",
        "edge_s4_strict": "Strict Memory",
    },
    "zh": {
        "fontname": "Noto Sans CJK SC",
        "edge_fontname": "Noto Sans CJK SC",
        "S1": "面向单请求还是\n高并发多请求？",
        "S1_Multi": '<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="2"><TR><TD ALIGN="LEFT">请求级并发并行：</TD></TR><TR><TD ALIGN="LEFT">• 批处理 (Batching)</TD></TR><TR><TD ALIGN="LEFT">• 模型多副本部署</TD></TR><TR><TD ALIGN="LEFT">• 负载均衡 (Load Balancing)</TD></TR></TABLE>>',
        "S2": "模型与上下文计算\n能否装入单卡？",
        "S2_Yes": '<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="2"><TR><TD ALIGN="LEFT">单卡极限优化：</TD></TR><TR><TD ALIGN="LEFT">• FlashAttention</TD></TR><TR><TD ALIGN="LEFT">• 模型量化 (INT8/4 / FP8)</TD></TR><TR><TD ALIGN="LEFT">• 算子融合 (Kernel Fusion)</TD></TR></TABLE>>',
        "S3": "计算如何在卡间切分？",
        "S3_Tensor": "张量并行 (TP)\n(vLLM / TRT-LLM)",
        "S3_Context": "上下文并行 (CP)\n(长上下文 / KV Cache 切分)",
        "S3_Pipe": "流水线并行 (PP)\n(模型层切分)",
        "S3_Expert": "专家并行 (EP)\n(稀疏 MoE 模型)",
        "S4": "显存容量还是 KV Cache\n成为系统瓶颈？",
        "S4_Paged": "PagedAttention\n(KV Cache 分页虚拟化)",
        "S4_Offload": "状态卸载 (Offloading)\n(CPU / NVMe / 存算分离)",
        "edge_s1_single": "单请求/超大模型",
        "edge_s1_multi": "高并发多请求",
        "edge_s2_yes": "是",
        "edge_s2_no": "否",
        "edge_s3_heads": "注意力头 / 隐藏层",
        "edge_s3_context": "长上下文",
        "edge_s3_pipe": "模型层切分",
        "edge_s3_expert": "稀疏 MoE",
        "edge_s4_cap": "KV 显存容量",
        "edge_s4_strict": "物理显存硬上限",
    }
}


def draw(text: dict) -> Digraph:
    dot = Digraph(comment='Inference Strategy Decision Tree')
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
    # Step 1: Scaling Type
    dot.node('S1', text['S1'])

    # Multiple Requests Branch
    dot.node('S1_Multi', text['S1_Multi'], **result_style)

    # Single Request / Large Model Branch
    dot.node('S2', text['S2'])

    # Single GPU Branch
    dot.node('S2_Yes', text['S2_Yes'], **result_style)

    # Model Parallel Branch
    dot.node('S3', text['S3'])
    dot.node('S3_Tensor', text['S3_Tensor'], **result_style)
    dot.node('S3_Context', text['S3_Context'], **result_style)
    dot.node('S3_Pipe', text['S3_Pipe'], **result_style)
    dot.node('S3_Expert', text['S3_Expert'], **result_style)

    # Step 4: Bottlenecks
    dot.node('S4', text['S4'])
    dot.node('S4_Paged', text['S4_Paged'], **result_style)
    dot.node('S4_Offload', text['S4_Offload'], **result_style)

    # --- Edges ---
    # Scaling Edges - S2 on left, S1_Multi on right
    dot.edge('S1', 'S2', label=text['edge_s1_single'])
    dot.edge('S1', 'S1_Multi', label=text['edge_s1_multi'])

    # Fitting Edges
    dot.edge('S2', 'S2_Yes', label=text['edge_s2_yes'])
    dot.edge('S2', 'S3', label=text['edge_s2_no'])

    # Splitting Edges
    dot.edge('S3', 'S3_Tensor', label=text['edge_s3_heads'])
    dot.edge('S3', 'S3_Context', label=text['edge_s3_context'])
    dot.edge('S3', 'S3_Pipe', label=text['edge_s3_pipe'])
    dot.edge('S3', 'S3_Expert', label=text['edge_s3_expert'])

    # Bottleneck connection
    dot.edge('S2_Yes', 'S4')
    dot.edge('S3_Context', 'S4')
    dot.edge('S4', 'S4_Paged', label=text['edge_s4_cap'])
    dot.edge('S4', 'S4_Offload', label=text['edge_s4_strict'])

    return dot


if __name__ == '__main__':
    localized_figure(draw, "inference_tree", LABELS, __file__)
