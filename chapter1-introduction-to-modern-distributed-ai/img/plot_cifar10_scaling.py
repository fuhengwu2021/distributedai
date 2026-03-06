"""
Plot training time vs number of GPUs for CIFAR-10 extended training.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add shared directory to path for math4ai imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'shared'))
from math4ai import configure_math_fonts, save_figure

# Configure matplotlib for math expressions
configure_math_fonts()

# Data from CIFAR-10 experiments (20 epochs)
gpus = np.array([1, 2, 4, 6, 8])
times = np.array([73.00, 46.47, 27.72, 21.24, 18.20])  # in seconds

# Calculate speedup
speedup = times[0] / times

# Create figure with single plot
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

# Plot: Training time vs number of GPUs
ax1.plot(gpus, times, 'o-', linewidth=2, markersize=6, color='#2E86AB')
ax1.set_xlabel('Number of GPUs', fontsize=11)
ax1.set_ylabel('Training Time (seconds)', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(gpus)

plt.tight_layout()
save_figure(__file__)
