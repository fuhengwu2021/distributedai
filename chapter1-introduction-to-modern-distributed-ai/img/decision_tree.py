from graphviz import Digraph
import os

# Create a directed graph
dot = Digraph(comment='GPU Usage Flowchart')
dot.attr(rankdir='TD', size='10', ranksep='1.5', nodesep='.1', dpi='300')

# Define node styles with rounded corners and gradient fills
dot.attr('node', 
         shape='box', 
         style='rounded,filled', 
         fillcolor='#E3F2FD:#BBDEFB',  # Gradient from light blue to lighter blue
         fontname='Helvetica',
         fontsize='16',
         penwidth='1.5',
         color='#1976D2',
         margin='0.05',
         gradientangle='90')  # Vertical gradient (top to bottom)

# --- Nodes ---
# Start
dot.node('A', 'Start: What is your use case')

# Training Branch
dot.node('B', 'Training or Fine tuning')
dot.node('B1', 'Model exceeds single GPU memory?')
dot.node('B1Y', 'Distributed Training\n(Model/Parameter Parallelism)', fillcolor='#A5D6A7:#C8E6C9', color='#2E7D32', gradientangle='90')
dot.node('B2', 'Training time too long?')
dot.node('B2Y', 'Distributed Training\n(Data Parallelism)', fillcolor='#A5D6A7:#C8E6C9', color='#2E7D32', gradientangle='90')
dot.node('B3', 'Fine tuning with LoRA/QLoRA?')
dot.node('B3Y', 'Single GPU', fillcolor='#FFF59D:#FFF9C4', color='#F57F17', gradientangle='90')
dot.node('B4', 'Large dataset?')
dot.node('B4Y', 'Consider Distributed Setup', fillcolor='#A5D6A7:#C8E6C9', color='#2E7D32', gradientangle='90')
dot.node('B4N', 'Single GPU', fillcolor='#FFF59D:#FFF9C4', color='#F57F17', gradientangle='90')

# Inference Branch
dot.node('D', 'Inference or Serving')
dot.node('D1', 'Model exceeds single GPU memory?')
dot.node('D1Y', 'Distributed Inference\n(Model Parallelism)', fillcolor='#A5D6A7:#C8E6C9', color='#2E7D32', gradientangle='90')
dot.node('D2', 'High throughput required?')
dot.node('D2Y', 'Distributed Inference\n(Multiple GPUs)', fillcolor='#A5D6A7:#C8E6C9', color='#2E7D32', gradientangle='90')
dot.node('D2N', 'Single GPU', fillcolor='#FFF59D:#FFF9C4', color='#F57F17', gradientangle='90')

# --- Edges ---
# Main Split
dot.edge('A', 'B')
dot.edge('A', 'D')

# Training Logic
dot.edge('B', 'B1')
dot.edge('B1', 'B1Y', label='Yes')
dot.edge('B1', 'B2', label='No')
dot.edge('B2', 'B2Y', label='Yes')
dot.edge('B2', 'B3', label='No')
dot.edge('B3', 'B3Y', label='Yes')
dot.edge('B3', 'B4', label='No')
dot.edge('B4', 'B4Y', label='Yes')
dot.edge('B4', 'B4N', label='No')

# Inference Logic
dot.edge('D', 'D1')
dot.edge('D1', 'D1Y', label='Yes')
dot.edge('D1', 'D2', label='No')
dot.edge('D2', 'D2Y', label='Yes')
dot.edge('D2', 'D2N', label='No')

# Render the graph
# Save to the same folder as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.splitext(os.path.basename(__file__))[0]
output_path = os.path.join(script_dir, script_name)
dot.render(output_path, format='png', cleanup=True)
print(f"Saved figure to: {output_path}.png")
