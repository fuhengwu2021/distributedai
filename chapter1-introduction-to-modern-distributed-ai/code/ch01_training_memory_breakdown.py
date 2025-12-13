"""
Visualize training memory requirements breakdown for a 7B parameter model.
Left subplot: FP32 (Float32) precision - single stacked bar showing 120-128GB range
Right subplot: BF16 (Bfloat16) precision - single stacked bar showing 64-72GB range
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Colors for each component
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: FP32 (Float32) - 7B model
# FP32: 4 bytes per parameter
# Model weights: 7B * 4 bytes = 28 GB
# Gradients: 7B * 4 bytes = 28 GB (same precision as weights)
# Optimizer states (Adam): 2× model size = 56 GB
# Activations: 8-16 GB (range)
fp32_components = ['Weights', 'Gradients', 'Optimizer', 'Activations']
fp32_memory = [28, 28, 56, 12]  # using average 12 GB for activations (midpoint of 8-16)
fp32_total_min = 28 + 28 + 56 + 8   # 120 GB
fp32_total_max = 28 + 28 + 56 + 16  # 128 GB

# Single stacked bar for FP32
x_pos = 0
width = 0.6
bottom = 0
bars1 = []
for i, (comp, mem) in enumerate(zip(fp32_components, fp32_memory)):
    bar = ax1.bar(x_pos, mem, width, bottom=bottom, 
                  label=comp, color=colors[i], edgecolor='white', linewidth=1.5)
    bars1.append(bar)
    # Add value labels
    if mem > 3:
        ax1.text(x_pos, bottom + mem/2, f'{int(mem)} GB', 
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    bottom += mem

# Add total range label at top
ax1.text(x_pos, fp32_total_max, f'{fp32_total_min}-{fp32_total_max} GB', 
         ha='center', va='bottom', fontsize=12, fontweight='bold')

ax1.set_xlabel('FP32 Precision', fontsize=12, fontweight='bold')
ax1.set_ylabel('Memory (GB)', fontsize=12, fontweight='bold')
ax1.set_xticks([x_pos])
ax1.set_xticklabels(['FP32'], fontsize=11)
ax1.set_ylim(0, 140)
ax1.set_title('Training Memory Breakdown for 7B Model\n(FP32 + Adam Optimizer)', 
              fontsize=13, fontweight='bold', pad=15)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.axhline(y=80, color='red', linestyle='--', linewidth=2, alpha=0.7, label='A100 GPU (80GB)')
ax1.legend(loc='center left', fontsize=9, framealpha=0.9, bbox_to_anchor=(1.02, 0.5))

# Right plot: BF16 (Bfloat16) - 7B model
# BF16: 2 bytes per parameter
# Model weights: 7B * 2 bytes = 14 GB
# Gradients: 7B * 2 bytes = 14 GB (same precision)
# Optimizer states (Adam): 2× model size = 28 GB
# Activations: 8-16 GB (range)
bf16_components = ['Weights', 'Gradients', 'Optimizer', 'Activations']
bf16_memory = [14, 14, 28, 12]  # using average 12 GB for activations (midpoint of 8-16)
bf16_total_min = 14 + 14 + 28 + 8   # 64 GB
bf16_total_max = 14 + 14 + 28 + 16  # 72 GB

# Single stacked bar for BF16
x_pos2 = 0
width2 = 0.6
bottom2 = 0
bars2 = []
for i, (comp, mem) in enumerate(zip(bf16_components, bf16_memory)):
    bar = ax2.bar(x_pos2, mem, width2, bottom=bottom2, 
                  label=comp, color=colors[i], edgecolor='white', linewidth=1.5)
    bars2.append(bar)
    # Add value labels
    if mem > 3:
        ax2.text(x_pos2, bottom2 + mem/2, f'{int(mem)} GB', 
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    bottom2 += mem

# Add total range label at top
ax2.text(x_pos2, bf16_total_max, f'{bf16_total_min}-{bf16_total_max} GB', 
         ha='center', va='bottom', fontsize=12, fontweight='bold')

ax2.set_xlabel('BF16 Precision', fontsize=12, fontweight='bold')
ax2.set_ylabel('Memory (GB)', fontsize=12, fontweight='bold')
ax2.set_xticks([x_pos2])
ax2.set_xticklabels(['BF16'], fontsize=11)
ax2.set_ylim(0, 80)
ax2.set_title('Training Memory Breakdown for 7B Model\n(BF16 + Adam Optimizer)', 
              fontsize=13, fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.axhline(y=80, color='red', linestyle='--', linewidth=2, alpha=0.7, label='A100 GPU (80GB)')
ax2.legend(loc='center left', fontsize=9, framealpha=0.9, bbox_to_anchor=(1.02, 0.5))

plt.tight_layout(rect=[0, 0, 0.95, 1])

# Save figure
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, 'training_memory_breakdown.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to: {output_path}")

plt.close()
