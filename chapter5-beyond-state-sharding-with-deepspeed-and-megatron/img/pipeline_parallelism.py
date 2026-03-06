import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

from math4ai import save_figure

# Colors
COLORS = {
    'forward': '#4CAF50',      # Green
    'backward': '#F44336',     # Red
    'bubble': '#E0E0E0',       # Gray
    'gpu_bg': '#FAFAFA',       # Light gray
    'border': '#424242',       # Dark gray
}

def draw_pipeline_naive(ax):
    """Draw naive pipeline execution with large bubbles."""
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, 4.5)
    ax.set_title("Naive Pipeline (Large Bubbles)", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("GPU", fontsize=10)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(['GPU 3\n(L24-31)', 'GPU 2\n(L16-23)', 'GPU 1\n(L8-15)', 'GPU 0\n(L0-7)'])
    ax.set_xticks([])
    
    # GPU backgrounds
    for i in range(4):
        ax.add_patch(patches.Rectangle((0, i), 12, 1, facecolor=COLORS['gpu_bg'], 
                                        edgecolor=COLORS['border'], linewidth=0.5))
    
    # Forward passes - sequential, one batch
    # GPU 0: F at t=0-1
    ax.add_patch(patches.FancyBboxPatch((0.5, 3.15), 1.5, 0.7, boxstyle="round,pad=0.02",
                                         facecolor=COLORS['forward'], edgecolor='white', linewidth=1))
    ax.text(1.25, 3.5, 'F', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # GPU 1: F at t=1-2
    ax.add_patch(patches.FancyBboxPatch((2, 2.15), 1.5, 0.7, boxstyle="round,pad=0.02",
                                         facecolor=COLORS['forward'], edgecolor='white', linewidth=1))
    ax.text(2.75, 2.5, 'F', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # GPU 2: F at t=2-3
    ax.add_patch(patches.FancyBboxPatch((3.5, 1.15), 1.5, 0.7, boxstyle="round,pad=0.02",
                                         facecolor=COLORS['forward'], edgecolor='white', linewidth=1))
    ax.text(4.25, 1.5, 'F', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # GPU 3: F at t=3-4
    ax.add_patch(patches.FancyBboxPatch((5, 0.15), 1.5, 0.7, boxstyle="round,pad=0.02",
                                         facecolor=COLORS['forward'], edgecolor='white', linewidth=1))
    ax.text(5.75, 0.5, 'F', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Backward passes
    # GPU 3: B at t=4-5
    ax.add_patch(patches.FancyBboxPatch((6.5, 0.15), 1.5, 0.7, boxstyle="round,pad=0.02",
                                         facecolor=COLORS['backward'], edgecolor='white', linewidth=1))
    ax.text(7.25, 0.5, 'B', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # GPU 2: B at t=5-6
    ax.add_patch(patches.FancyBboxPatch((8, 1.15), 1.5, 0.7, boxstyle="round,pad=0.02",
                                         facecolor=COLORS['backward'], edgecolor='white', linewidth=1))
    ax.text(8.75, 1.5, 'B', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # GPU 1: B at t=6-7
    ax.add_patch(patches.FancyBboxPatch((9.5, 2.15), 1.5, 0.7, boxstyle="round,pad=0.02",
                                         facecolor=COLORS['backward'], edgecolor='white', linewidth=1))
    ax.text(10.25, 2.5, 'B', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # GPU 0: B at t=7-8
    ax.add_patch(patches.FancyBboxPatch((11, 3.15), 1.5, 0.7, boxstyle="round,pad=0.02",
                                         facecolor=COLORS['backward'], edgecolor='white', linewidth=1))
    ax.text(11.75, 3.5, 'B', ha='center', va='center', fontsize=10, fontweight='bold', color='white')


def draw_pipeline_1f1b(ax):
    """Draw 1F1B pipeline schedule with micro-batches."""
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.5, 4.5)
    ax.set_title("1F1B Pipeline (Micro-batches)", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("GPU", fontsize=10)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(['GPU 3\n(L24-31)', 'GPU 2\n(L16-23)', 'GPU 1\n(L8-15)', 'GPU 0\n(L0-7)'])
    ax.set_xticks([])
    
    # GPU backgrounds
    for i in range(4):
        ax.add_patch(patches.Rectangle((0, i), 14, 1, facecolor=COLORS['gpu_bg'], 
                                        edgecolor=COLORS['border'], linewidth=0.5))
    
    w = 0.8  # width of each micro-batch block
    
    # Micro-batch labels
    labels = ['1', '2', '3', '4']
    
    # GPU 0 (top): F1, F2, F3, F4, then B1, B2, B3, B4
    for i, label in enumerate(labels):
        # Forward
        ax.add_patch(patches.FancyBboxPatch((0.3 + i*w, 3.15), w-0.1, 0.7, boxstyle="round,pad=0.02",
                                             facecolor=COLORS['forward'], edgecolor='white', linewidth=1))
        ax.text(0.3 + i*w + (w-0.1)/2, 3.5, f'F{label}', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # GPU 0 backwards (after warmup)
    for i, label in enumerate(labels):
        ax.add_patch(patches.FancyBboxPatch((7.1 + i*w, 3.15), w-0.1, 0.7, boxstyle="round,pad=0.02",
                                             facecolor=COLORS['backward'], edgecolor='white', linewidth=1))
        ax.text(7.1 + i*w + (w-0.1)/2, 3.5, f'B{label}', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # GPU 1: shifted by 1
    for i, label in enumerate(labels):
        ax.add_patch(patches.FancyBboxPatch((1.1 + i*w, 2.15), w-0.1, 0.7, boxstyle="round,pad=0.02",
                                             facecolor=COLORS['forward'], edgecolor='white', linewidth=1))
        ax.text(1.1 + i*w + (w-0.1)/2, 2.5, f'F{label}', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    for i, label in enumerate(labels):
        ax.add_patch(patches.FancyBboxPatch((6.3 + i*w, 2.15), w-0.1, 0.7, boxstyle="round,pad=0.02",
                                             facecolor=COLORS['backward'], edgecolor='white', linewidth=1))
        ax.text(6.3 + i*w + (w-0.1)/2, 2.5, f'B{label}', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # GPU 2: shifted by 2
    for i, label in enumerate(labels):
        ax.add_patch(patches.FancyBboxPatch((1.9 + i*w, 1.15), w-0.1, 0.7, boxstyle="round,pad=0.02",
                                             facecolor=COLORS['forward'], edgecolor='white', linewidth=1))
        ax.text(1.9 + i*w + (w-0.1)/2, 1.5, f'F{label}', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    for i, label in enumerate(labels):
        ax.add_patch(patches.FancyBboxPatch((5.5 + i*w, 1.15), w-0.1, 0.7, boxstyle="round,pad=0.02",
                                             facecolor=COLORS['backward'], edgecolor='white', linewidth=1))
        ax.text(5.5 + i*w + (w-0.1)/2, 1.5, f'B{label}', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # GPU 3: shifted by 3
    for i, label in enumerate(labels):
        ax.add_patch(patches.FancyBboxPatch((2.7 + i*w, 0.15), w-0.1, 0.7, boxstyle="round,pad=0.02",
                                             facecolor=COLORS['forward'], edgecolor='white', linewidth=1))
        ax.text(2.7 + i*w + (w-0.1)/2, 0.5, f'F{label}', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    for i, label in enumerate(labels):
        ax.add_patch(patches.FancyBboxPatch((4.7 + i*w, 0.15), w-0.1, 0.7, boxstyle="round,pad=0.02",
                                             facecolor=COLORS['backward'], edgecolor='white', linewidth=1))
        ax.text(4.7 + i*w + (w-0.1)/2, 0.5, f'B{label}', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Legend
    ax.add_patch(patches.FancyBboxPatch((11.5, 3.6), 0.4, 0.3, boxstyle="round,pad=0.02",
                                         facecolor=COLORS['forward'], edgecolor='white', linewidth=1))
    ax.text(12.1, 3.75, 'Forward', ha='left', va='center', fontsize=8)
    ax.add_patch(patches.FancyBboxPatch((11.5, 3.1), 0.4, 0.3, boxstyle="round,pad=0.02",
                                         facecolor=COLORS['backward'], edgecolor='white', linewidth=1))
    ax.text(12.1, 3.25, 'Backward', ha='left', va='center', fontsize=8)


def main():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    
    draw_pipeline_naive(ax1)
    draw_pipeline_1f1b(ax2)
    
    plt.tight_layout()
    save_figure(__file__)


if __name__ == "__main__":
    main()
