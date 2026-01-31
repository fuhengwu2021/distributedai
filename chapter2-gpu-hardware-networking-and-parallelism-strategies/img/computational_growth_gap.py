"""
GPU Memory Capacity vs Model Memory Requirements

This plot shows actual GPU memory capacity over time versus actual model memory
requirements (in BF16/FP16), using real data from NVIDIA GPU specifications and
published model sizes. This illustrates why distributed training is essential.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add shared directory to path for math4ai imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from math4ai import configure_math_fonts

# Configure matplotlib for math expressions
configure_math_fonts()

# Real GPU memory capacity data (GB) - from NVIDIA specifications
gpu_years = np.array([2020, 2022, 2023, 2024])
gpu_memory = np.array([80, 80, 141, 192])  # A100, H100, H200, B200

# Real model memory requirements (GB in BF16/FP16) - from published model sizes
# Model sizes from chapter 1, converted to memory: params × 2 bytes (BF16)
model_years = np.array([2020, 2022, 2023, 2024, 2025])
model_memory = np.array([
    350.0,   # GPT-3 (175B params) = 175B × 2 bytes = 350GB
    140.0,   # LLaMA-2 70B = 70B × 2 bytes = 140GB
    472.0,   # DeepSeek-V2 236B = 236B × 2 bytes = 472GB
    1342.0,  # DeepSeek-V3 671B = 671B × 2 bytes = 1342GB
    15000.0  # Gemini-3-Pro ~7.5T = 7.5T × 2 bytes = 15TB = 15,000GB
])

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

# Plotting GPU memory capacity (actual hardware)
ax.plot(gpu_years, gpu_memory, 'b-o', linewidth=2.5, markersize=8,
        label='Single GPU Memory Capacity (A100/H100/H200/B200)', zorder=3)

# Plotting model memory requirements (actual model sizes)
ax.plot(model_years, model_memory, 'r-s', linewidth=2.5, markersize=8,
        label='Model Memory Requirements (BF16, representative models)', zorder=3)

# Fill the gap area where models exceed single GPU capacity
# Interpolate GPU memory for model years
gpu_interp = np.interp(model_years, gpu_years, gpu_memory)
gap_mask = model_memory > gpu_interp
ax.fill_between(model_years, gpu_interp, model_memory, 
                where=gap_mask, alpha=0.3, color='red', 
                label='Memory Gap (requires distributed training)', zorder=1)

# Add annotations for specific models
ax.annotate('GPT-3\n(175B)', xy=(2020, 350), xytext=(2020.5, 600),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'),
            fontsize=9, ha='left')
ax.annotate('LLaMA-2\n(70B)', xy=(2022, 140), xytext=(2022.5, 300),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'),
            fontsize=9, ha='left')
ax.annotate('DeepSeek-V3\n(671B)', xy=(2025, 1342), xytext=(2024.2, 3000),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'),
            fontsize=9, ha='right')
ax.annotate('Gemini-3-Pro\n(~7.5T)', xy=(2025, 15000), xytext=(2024.5, 8000),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'),
            fontsize=9, ha='right')

# Styling (no title - captions are in markdown, font sizes optimized for 7x10 book)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Memory (GB)', fontsize=12)
# No title - figure captions are provided in markdown
ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax.tick_params(labelsize=11)  # Ensure tick labels are readable

# Set y-axis to log scale to show the exponential growth
ax.set_yscale('log')
ax.set_ylim(bottom=50, top=20000)

plt.tight_layout()

# Save figure (standard pattern: same name as script)
script_dir = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.splitext(os.path.basename(__file__))[0]
output_path = os.path.join(script_dir, f'{script_name}.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0.03)
print(f"Saved figure to: {output_path}")
plt.close()  # Close to free memory
