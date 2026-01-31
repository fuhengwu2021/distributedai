from graphviz import Digraph
import os

# Initialize the graph
dot = Digraph(comment='Inference Strategy Decision Tree')
dot.attr(rankdir='TD', size='12', ranksep='1.0', nodesep='.5', dpi='300')

# Global Node Styles
dot.attr('node', 
         shape='box', 
         style='rounded,filled', 
         fillcolor='#E3F2FD:#BBDEFB',  # Blue gradient for questions
         fontname='Helvetica',
         fontsize='16',
         penwidth='1.5',
         color='#1976D2',
         gradientangle='90')

# Result Node Style (Green)
result_style = {'fillcolor': '#A5D6A7:#C8E6C9', 'color': '#2E7D32'}

# --- Nodes ---
# Step 1: Scaling Type
dot.node('S1', 'Are you scaling a Single Request \nor Multiple Requests?')

# Multiple Requests Branch
dot.node('S1_Multi', '<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="2"><TR><TD ALIGN="LEFT">Request-Level Parallelism:</TD></TR><TR><TD ALIGN="LEFT">• Batching</TD></TR><TR><TD ALIGN="LEFT">• Model Replicas</TD></TR><TR><TD ALIGN="LEFT">• Load Balancing</TD></TR></TABLE>>', **result_style)

# Single Request / Large Model Branch
dot.node('S2', 'Does the model computation \nfit on one device?')

# Single GPU Branch
dot.node('S2_Yes', '<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="2"><TR><TD ALIGN="LEFT">Single-GPU Optimized:</TD></TR><TR><TD ALIGN="LEFT">• FlashAttention</TD></TR><TR><TD ALIGN="LEFT">• Quantization (INT8/4)</TD></TR><TR><TD ALIGN="LEFT">• Kernel Fusion</TD></TR></TABLE>>', **result_style)

# Model Parallel Branch
dot.node('S3', 'How is computation split?')
dot.node('S3_Tensor', 'Tensor Parallelism\n(vLLM / TRT-LLM)', **result_style)
dot.node('S3_Context', 'Context Parallelism\n(Long-context / KV split)', **result_style)
dot.node('S3_Pipe', 'Pipeline Parallelism\n(Layer-wise split)', **result_style)
dot.node('S3_Expert', 'Expert Parallelism\n(MoE Models)', **result_style)

# Step 4: Bottlenecks
dot.node('S4', 'Is Memory or KV Cache \nthe bottleneck?')
dot.node('S4_Paged', 'PagedAttention\n(Virtualize KV Cache)', **result_style)
dot.node('S4_Offload', 'Offloading\n(CPU/NVMe/Disaggregation)', **result_style)

# --- Edges ---
# Scaling Edges - S2 on left, S1_Multi on right
dot.edge('S1', 'S2', label='Single/Large')
dot.edge('S1', 'S1_Multi', label='Multiple')

# Fitting Edges
dot.edge('S2', 'S2_Yes', label='Yes')
dot.edge('S2', 'S3', label='No')

# Splitting Edges
dot.edge('S3', 'S3_Tensor', label='Heads/Hidden')
dot.edge('S3', 'S3_Context', label='Long Context')
dot.edge('S3', 'S3_Pipe', label='Layers')
dot.edge('S3', 'S3_Expert', label='MoE')

# Bottleneck connection
dot.edge('S2_Yes', 'S4')
dot.edge('S3_Context', 'S4')
dot.edge('S4', 'S4_Paged', label='Capacity')
dot.edge('S4', 'S4_Offload', label='Strict Memory')

# --- Render ---
# Save figure (standard pattern: same name as script)
script_dir = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.splitext(os.path.basename(__file__))[0]
output_path_base = os.path.join(script_dir, script_name)
# graphviz automatically appends .png, so don't include extension
dot.render(output_path_base, format='png', cleanup=True)
output_path = os.path.join(script_dir, f'{script_name}.png')
print(f"Saved figure to: {output_path}")

