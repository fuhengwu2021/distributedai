from graphviz import Digraph
import os

# Initialize the graph
dot = Digraph(comment='Training Strategy Decision Tree')
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
# Step 1: Fitting
dot.node('S1', 'Does a full model replica\nfit on one device?')

# Replicated Data Parallelism
dot.node('DDP', 'Replicated Data Parallelism (DDP)\n(Simple starting point)', **result_style)

# Step 2: Sharding
dot.node('S2', 'Shard parameters, gradients,\nand optimizer states?')

# Sharded Data Parallelism
dot.node('FSDP', 'Sharded Data Parallelism\n(FSDP / ZeRO-3)', **result_style)

# Step 3: True Model Parallelism
dot.node('S3', 'Is computation of one\nsample split across devices?')

# Split types
dot.node('TP', 'Tensor Parallelism\n(Heads/Hidden dimensions)', **result_style)
dot.node('SP', 'Sequence Parallelism\n(Long sequences)', **result_style)
dot.node('PP', 'Pipeline Parallelism\n(Layer/Stage splitting)', **result_style)
dot.node('EP', 'Expert Parallelism\n(MoE Models)', **result_style)

# Step 4: Hybrid
dot.node('S4', 'Still need more scale?\n(Hybrid Combinations)')
dot.node('Hybrid', '3D Parallelism\n(DP + TP + PP)', **result_style)

# Step 5: Optimizations
dot.node('S5', 'Memory still insufficient?')
dot.node('Opt', '<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="2"><TR><TD ALIGN="LEFT">System-level Optimizations:</TD></TR><TR><TD ALIGN="LEFT">• Activation Checkpointing</TD></TR><TR><TD ALIGN="LEFT">• CPU/NVMe Offloading</TD></TR></TABLE>>', **result_style)

# --- Edges ---
# Step 1 edges
dot.edge('S1', 'DDP', label='Yes')
dot.edge('S1', 'S2', label='No')

# Step 2 edges
dot.edge('S2', 'FSDP')
dot.edge('FSDP', 'S3', label='Still memory limited')

# Step 3 edges - splitting logic
dot.edge('S3', 'TP', label='Large Layers')
dot.edge('S3', 'SP', label='Long Sequence')
dot.edge('S3', 'PP', label='Deep Models')
dot.edge('S3', 'EP', label='Sparse/MoE')

# Step 4 edges - hybrid combinations
dot.edge('TP', 'S4')
dot.edge('PP', 'S4')
dot.edge('S4', 'Hybrid')
dot.edge('Hybrid', 'S5')

# Step 5 edges - optimizations
dot.edge('S5', 'Opt', label='Yes')

# --- Render ---
# Save figure (standard pattern: same name as script)
script_dir = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.splitext(os.path.basename(__file__))[0]
output_path_base = os.path.join(script_dir, script_name)
# graphviz automatically appends .png, so don't include extension
dot.render(output_path_base, format='png', cleanup=True)
output_path = os.path.join(script_dir, f'{script_name}.png')
print(f"Saved figure to: {output_path}")
