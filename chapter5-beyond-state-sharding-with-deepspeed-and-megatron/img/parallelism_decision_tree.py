from graphviz import Digraph
import os

# Create a directed graph
dot = Digraph(comment='Parallelism Strategy Decision Tree')
dot.attr(rankdir='TD', size='12', ranksep='0.8', nodesep='0.3', dpi='300')

# Define node styles
# Question nodes - blue gradient
dot.attr('node', 
         shape='box', 
         style='rounded,filled', 
         fillcolor='#E3F2FD:#BBDEFB',
         fontname='Helvetica',
         fontsize='14',
         penwidth='1.5',
         color='#1976D2',
         margin='0.15',
         gradientangle='90')

# --- Question Nodes ---
dot.node('Q1', 'Model Size < 10B?')
dot.node('Q2', 'Model Size < 50B?')
dot.node('Q3', 'Single layer fits on one GPU?')
dot.node('Q4', 'Sequence Length >= 8K?')
dot.node('Q5', 'Model Size < 200B?')
dot.node('Q6', 'Multiple Nodes?')
dot.node('Q7', 'MoE Model?')

# --- Answer Nodes (Green - recommended strategies) ---
green_style = {'fillcolor': '#A5D6A7:#C8E6C9', 'color': '#2E7D32'}

dot.node('A1', 'DDP or ZeRO-1\n(simplest, fastest)', **green_style)
dot.node('A2', 'ZeRO-2 or FSDP2\n(shards gradients)', **green_style)
dot.node('A3', 'ZeRO-3 or FSDP2\n(full state sharding)', **green_style)
dot.node('A4', 'FSDP2 + TP + CP\n(long sequences)', **green_style)
dot.node('A5', 'FSDP2 + TP\n(50B-200B models)', **green_style)
dot.node('A6', 'FSDP2 + TP + PP\n(multi-node scaling)', **green_style)
dot.node('A7', 'FSDP2 + TP + EP\n(MoE models)', **green_style)
dot.node('A8', 'FSDP2 + TP + PP\n(single node, >200B)', **green_style)

# --- Edges ---
# Q1: Model Size < 10B?
dot.edge('Q1', 'A1', label='Yes')
dot.edge('Q1', 'Q2', label='No')

# Q2: Model Size < 50B?
dot.edge('Q2', 'A2', label='Yes')
dot.edge('Q2', 'Q3', label='No')

# Q3: Single layer fits on one GPU?
dot.edge('Q3', 'A3', label='Yes')
dot.edge('Q3', 'Q4', label='No')

# Q4: Sequence Length >= 8K?
dot.edge('Q4', 'A4', label='Yes')
dot.edge('Q4', 'Q5', label='No')

# Q5: Model Size < 200B?
dot.edge('Q5', 'A5', label='Yes')
dot.edge('Q5', 'Q6', label='No')

# Q6: Multiple Nodes?
dot.edge('Q6', 'A6', label='Yes')
dot.edge('Q6', 'Q7', label='No')

# Q7: MoE Model?
dot.edge('Q7', 'A7', label='Yes')
dot.edge('Q7', 'A8', label='No')

# Render the graph
script_dir = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.splitext(os.path.basename(__file__))[0]
output_path = os.path.join(script_dir, script_name)
dot.render(output_path, format='png', cleanup=True)
print(f"Saved figure to: {output_path}.png")
