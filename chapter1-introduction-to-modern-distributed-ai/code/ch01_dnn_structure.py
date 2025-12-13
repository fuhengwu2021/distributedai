"""
Visualize a simple DNN structure: x -> Linear -> z -> Sigmoid -> h -> Linear -> y_hat
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Define layer positions
x_pos = 1.5
z_pos = 4.0
h_pos = 6.5
y_pos = 9.0
y_mid = 3.0

# Node sizes and colors
node_radius = 0.3
input_color = '#4A90E2'
hidden_color = '#7ED321'
output_color = '#F5A623'
activation_color = '#BD10E0'

# Draw input layer (x)
x_nodes = 3
x_y_positions = np.linspace(1.5, 4.5, x_nodes)
for i, y_pos_node in enumerate(x_y_positions):
    circle = plt.Circle((x_pos, y_pos_node), node_radius, 
                       color=input_color, ec='black', linewidth=2, zorder=3)
    ax.add_patch(circle)
    if i == x_nodes // 2:
        ax.text(x_pos, y_pos_node, 'x', ha='center', va='center', 
               fontsize=12, fontweight='bold', zorder=4)
    else:
        ax.text(x_pos, y_pos_node, f'x{i+1}', ha='center', va='center', 
               fontsize=10, zorder=4)

# Draw z layer (after first linear)
z_nodes = 4
z_y_positions = np.linspace(1.2, 4.8, z_nodes)
for i, y_pos_node in enumerate(z_y_positions):
    circle = plt.Circle((z_pos, y_pos_node), node_radius, 
                       color=hidden_color, ec='black', linewidth=2, zorder=3)
    ax.add_patch(circle)
    ax.text(z_pos, y_pos_node, f'z{i+1}', ha='center', va='center', 
           fontsize=10, zorder=4)

# Draw h layer (after sigmoid)
h_nodes = 4
h_y_positions = np.linspace(1.2, 4.8, h_nodes)
for i, y_pos_node in enumerate(h_y_positions):
    circle = plt.Circle((h_pos, y_pos_node), node_radius, 
                       color=hidden_color, ec='black', linewidth=2, zorder=3)
    ax.add_patch(circle)
    ax.text(h_pos, y_pos_node, f'h{i+1}', ha='center', va='center', 
           fontsize=10, zorder=4)

# Draw output layer (y_hat)
y_nodes = 2
y_y_positions = np.linspace(2.0, 4.0, y_nodes)
for i, y_pos_node in enumerate(y_y_positions):
    circle = plt.Circle((y_pos, y_pos_node), node_radius, 
                       color=output_color, ec='black', linewidth=2, zorder=3)
    ax.add_patch(circle)
    if i == 0:
        ax.text(y_pos, y_pos_node, 'ŷ', ha='center', va='center', 
               fontsize=12, fontweight='bold', zorder=4)
    else:
        ax.text(y_pos, y_pos_node, f'ŷ{i+1}', ha='center', va='center', 
               fontsize=10, zorder=4)

# Draw connections from x to z (Linear layer 1)
for i, x_y in enumerate(x_y_positions):
    for j, z_y in enumerate(z_y_positions):
        # Create connection with some curvature
        mid_x = (x_pos + z_pos) / 2
        mid_y = (x_y + z_y) / 2
        # Use quadratic bezier for smooth curves
        t = np.linspace(0, 1, 20)
        curve_x = (1-t)**2 * x_pos + 2*(1-t)*t * mid_x + t**2 * z_pos
        curve_y = (1-t)**2 * x_y + 2*(1-t)*t * mid_y + t**2 * z_y
        ax.plot(curve_x, curve_y, 'gray', alpha=0.3, linewidth=0.8, zorder=1)

# Draw connections from z to h (through sigmoid)
for i, z_y in enumerate(z_y_positions):
    h_y = h_y_positions[i]  # One-to-one connection
    ax.plot([z_pos, h_pos], [z_y, h_y], 'gray', alpha=0.3, linewidth=0.8, zorder=1)

# Draw connections from h to y_hat (Linear layer 2)
for i, h_y in enumerate(h_y_positions):
    for j, y_y in enumerate(y_y_positions):
        mid_x = (h_pos + y_pos) / 2
        mid_y = (h_y + y_y) / 2
        t = np.linspace(0, 1, 20)
        curve_x = (1-t)**2 * h_pos + 2*(1-t)*t * mid_x + t**2 * y_pos
        curve_y = (1-t)**2 * h_y + 2*(1-t)*t * mid_y + t**2 * y_y
        ax.plot(curve_x, curve_y, 'gray', alpha=0.3, linewidth=0.8, zorder=1)

# Add layer labels and equations
# Input layer
ax.text(x_pos, 0.5, 'Input\nx', ha='center', va='top', fontsize=11, fontweight='bold')

# Linear layer 1
ax.text((x_pos + z_pos) / 2, 5.5, 'Linear Layer 1\nz = W₁x + b₁', 
       ha='center', va='bottom', fontsize=10, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Sigmoid activation
ax.text((z_pos + h_pos) / 2, 5.5, 'Sigmoid\nh = σ(z)', 
       ha='center', va='bottom', fontsize=10, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor=activation_color, alpha=0.3))

# Linear layer 2
ax.text((h_pos + y_pos) / 2, 5.5, 'Linear Layer 2\nŷ = W₂h + b₂', 
       ha='center', va='bottom', fontsize=10, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Output layer
ax.text(y_pos, 0.5, 'Output\nŷ', ha='center', va='top', fontsize=11, fontweight='bold')

# Add arrows between layers
arrow_props = dict(arrowstyle='->', lw=2, color='black', zorder=2)
ax.annotate('', xy=(z_pos - node_radius - 0.1, y_mid), 
           xytext=(x_pos + node_radius + 0.1, y_mid),
           arrowprops=arrow_props)
ax.annotate('', xy=(h_pos - node_radius - 0.1, y_mid), 
           xytext=(z_pos + node_radius + 0.1, y_mid),
           arrowprops=arrow_props)
ax.annotate('', xy=(y_pos - node_radius - 0.1, y_mid), 
           xytext=(h_pos + node_radius + 0.1, y_mid),
           arrowprops=arrow_props)

# Add title
ax.text(5, 5.8, 'Deep Neural Network Structure', 
       ha='center', va='top', fontsize=14, fontweight='bold')

# Add legend
legend_elements = [
    mpatches.Patch(facecolor=input_color, edgecolor='black', label='Input Layer'),
    mpatches.Patch(facecolor=hidden_color, edgecolor='black', label='Hidden Layer'),
    mpatches.Patch(facecolor=output_color, edgecolor='black', label='Output Layer'),
    mpatches.Patch(facecolor=activation_color, edgecolor='black', alpha=0.3, label='Activation Function')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

plt.tight_layout()

# Save figure
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, 'dnn_structure.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to: {output_path}")

plt.close()
