"""
Visualize memory usage timeline during training.
Shows how weights, activations, gradients, and optimizer states occupy VRAM at different stages.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Memory sizes for a 7B model with BF16 and Adam (in GB)
weights = 14  # Model weights (always in memory)
optimizer_states = 28  # Adam optimizer states (m_{t-1} + v_{t-1}, always in memory)
activations_min = 8  # Minimum activations
activations_max = 16  # Maximum activations
activations_avg = 12  # Average activations
gradients = 14  # Gradients

# Timeline stages
stages = ['Forward\nPass', 'Backward\nPass', 'Optimizer\nStep']
x_pos = np.arange(len(stages))

# Memory breakdown for each stage
# Format: [weights, optimizer_states, activations, gradients]
memory_forward = [weights, optimizer_states, activations_avg, 0]  # No gradients yet
memory_backward = [weights, optimizer_states, activations_avg, gradients]  # All present
memory_optimizer = [weights, optimizer_states, 0, gradients]  # Activations freed

# Colors for each component
colors = {
    'weights': '#2E86AB',
    'optimizer': '#A23B72',
    'activations': '#F18F01',
    'gradients': '#C73E1D'
}

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Stack bars for each stage
width = 0.6
seen_labels = set()  # Track which labels we've already added to legend

# Forward pass
bars_forward = []
labels_forward = ['Weights', 'Optimizer States', 'Activations']
values_forward = [weights, optimizer_states, activations_avg]
colors_forward = [colors['weights'], colors['optimizer'], colors['activations']]

bottom = 0
for i, (label, value, color) in enumerate(zip(labels_forward, values_forward, colors_forward)):
    # Only add label if we haven't seen it before
    label_to_use = label if label not in seen_labels else ''
    if label not in seen_labels:
        seen_labels.add(label)
    bar = ax.bar(x_pos[0], value, width, bottom=bottom, 
                 label=label_to_use, color=color, 
                 edgecolor='white', linewidth=1.5)
    bars_forward.append(bar)
    if value > 2:
        ax.text(x_pos[0], bottom + value/2, f'{int(value)} GB',
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    bottom += value

# Backward pass
bars_backward = []
labels_backward = ['Weights', 'Optimizer States', 'Activations', 'Gradients']
values_backward = [weights, optimizer_states, activations_avg, gradients]
colors_backward = [colors['weights'], colors['optimizer'], colors['activations'], colors['gradients']]

bottom = 0
for i, (label, value, color) in enumerate(zip(labels_backward, values_backward, colors_backward)):
    # Only add label if we haven't seen it before
    label_to_use = label if label not in seen_labels else ''
    if label not in seen_labels:
        seen_labels.add(label)
    bar = ax.bar(x_pos[1], value, width, bottom=bottom,
                 label=label_to_use, color=color,
                 edgecolor='white', linewidth=1.5)
    bars_backward.append(bar)
    if value > 2:
        ax.text(x_pos[1], bottom + value/2, f'{int(value)} GB',
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    bottom += value

# Optimizer step
bars_optimizer = []
labels_optimizer = ['Weights', 'Optimizer States', 'Gradients']
values_optimizer = [weights, optimizer_states, gradients]
colors_optimizer = [colors['weights'], colors['optimizer'], colors['gradients']]

bottom = 0
for i, (label, value, color) in enumerate(zip(labels_optimizer, values_optimizer, colors_optimizer)):
    # Only add label if we haven't seen it before
    label_to_use = label if label not in seen_labels else ''
    if label not in seen_labels:
        seen_labels.add(label)
    bar = ax.bar(x_pos[2], value, width, bottom=bottom,
                 label=label_to_use, color=color,
                 edgecolor='white', linewidth=1.5)
    bars_optimizer.append(bar)
    if value > 2:
        ax.text(x_pos[2], bottom + value/2, f'{int(value)} GB',
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    bottom += value

# Add total labels at top
total_forward = sum(values_forward)
total_backward = sum(values_backward)
total_optimizer = sum(values_optimizer)

ax.text(x_pos[0], total_forward, f'{int(total_forward)} GB',
        ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.text(x_pos[1], total_backward, f'{int(total_backward)} GB',
        ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.text(x_pos[2], total_optimizer, f'{int(total_optimizer)} GB',
        ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add A100 GPU limit line
ax.axhline(y=80, color='red', linestyle='--', linewidth=2, alpha=0.7, label='A100 GPU (80GB)')

# Labels and formatting
ax.set_xlabel('Training Stage', fontsize=12, fontweight='bold')
ax.set_ylabel('Memory (GB)', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(stages, fontsize=11)
ax.set_ylim(0, 85)
ax.set_title('Training Memory Timeline for 7B Model (BF16 + Adam)\nMemory Usage Across Training Stages',
              fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

# Add annotations
ax.annotate('Peak Memory', xy=(x_pos[1], total_backward), xytext=(x_pos[1], total_backward + 5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, fontweight='bold', color='red', ha='center')

plt.tight_layout()

# Save figure
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, 'training_memory_timeline.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to: {output_path}")

plt.close()
