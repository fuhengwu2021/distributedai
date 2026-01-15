#!/usr/bin/env python3
"""
Generate a beautiful artistic background image for book cover.
Uses matplotlib to create a mathematical/geometric design.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Polygon, FancyBboxPatch
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects

def create_cover_background(output_path='img/cover-background.pdf', dpi=300):
    """
    Create a beautiful artistic background for the book cover.
    
    Args:
        output_path: Path to save the background image
        dpi: Resolution (dots per inch)
    """
    # A4 size in inches (for PDF)
    fig_width = 8.27  # A4 width in inches
    fig_height = 11.69  # A4 height in inches
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Create a gradient background
    # Use a light and bright color scheme - light blue to sky blue gradient
    colors = ['#e8f4f8', '#d4e8f0', '#c0dce8', '#b0d0e0', '#a0c4d8', '#90b8d0', '#80acc8', '#70a0c0']
    n_bars = 150
    for i in range(n_bars):
        y_pos = i / n_bars
        height = 1 / n_bars
        # Smooth color interpolation
        color_idx = (i / n_bars) * (len(colors) - 1)
        idx_low = int(color_idx)
        idx_high = min(idx_low + 1, len(colors) - 1)
        t = color_idx - idx_low
        
        # Interpolate between colors
        from matplotlib.colors import hex2color
        c1 = np.array(hex2color(colors[idx_low]))
        c2 = np.array(hex2color(colors[idx_high]))
        color = c1 + t * (c2 - c1)
        
        # Vary alpha for depth - more opaque in center, transparent at edges
        # Use higher alpha for lighter colors to make them more visible
        alpha = 0.6 + 0.4 * (1 - abs(y_pos - 0.5) * 2)
        rect = patches.Rectangle((0, y_pos), 1, height, 
                                 facecolor=color, alpha=alpha, 
                                 edgecolor='none', linewidth=0)
        ax.add_patch(rect)
    
    # Add geometric patterns - circles with varying sizes and colors
    # Create a more organic, flowing pattern
    n_circles_x = 15
    n_circles_y = 20
    
    for i in range(n_circles_x):
        for j in range(n_circles_y):
            x = (i + 0.5) / n_circles_x
            y = (j + 0.5) / n_circles_y
            
            # Vary circle size based on position - create wave pattern
            size = 0.015 + 0.025 * np.sin(i * np.pi / n_circles_x * 2) * np.cos(j * np.pi / n_circles_y * 2)
            
            # Vary opacity - more visible in certain areas
            # Increase opacity for better visibility on light background
            alpha = 0.15 + 0.20 * (1 - abs(x - 0.5) * 2) * (1 - abs(y - 0.5) * 2)
            
            # Use bright white for contrast against light background
            circle_color = (1.0, 1.0, 1.0)
            
            circle = Circle((x, y), size, 
                          facecolor=circle_color, 
                          edgecolor='white', 
                          alpha=alpha,
                          linewidth=0.3)
            ax.add_patch(circle)
    
    # Add mathematical symbols/patterns - sine waves and curves
    x_wave = np.linspace(0, 1, 1000)
    for i in range(6):
        y_wave = 0.15 + 0.7 * i / 5
        # Create more complex wave patterns
        wave = 0.04 * np.sin(2 * np.pi * 2.5 * x_wave + i * np.pi / 3) * np.cos(2 * np.pi * 1.5 * x_wave)
        alpha = 0.20 - 0.025 * abs(i - 2.5)
        ax.plot(x_wave, y_wave + wave, 'w-', linewidth=1.5, alpha=alpha)
    
    # Add some curved lines for elegance
    for i in range(4):
        x_curve = np.linspace(0, 1, 500)
        y_curve = 0.3 + 0.4 * i / 3
        curve = 0.03 * np.sin(2 * np.pi * 4 * x_curve + i * np.pi / 4)
        ax.plot(x_curve, y_curve + curve, 'w-', linewidth=1.0, alpha=0.15)
    
    # Add diagonal lines for depth
    for i in range(8):
        angle = i * np.pi / 8
        x1 = 0.5 + 0.6 * np.cos(angle)
        y1 = 0.5 + 0.6 * np.sin(angle)
        x2 = 0.5 - 0.6 * np.cos(angle)
        y2 = 0.5 - 0.6 * np.sin(angle)
        ax.plot([x1, x2], [y1, y2], 'w-', linewidth=1.0, alpha=0.12)
    
    # Add some geometric shapes - triangles
    for i in range(6):
        x_center = 0.2 + 0.6 * (i % 3) / 2
        y_center = 0.2 + 0.6 * (i // 3) / 1
        size = 0.08
        triangle = Polygon([(x_center, y_center + size),
                           (x_center - size * 0.866, y_center - size * 0.5),
                           (x_center + size * 0.866, y_center - size * 0.5)],
                          facecolor='white', 
                          edgecolor='white',
                          alpha=0.2,
                          linewidth=1.5)
        ax.add_patch(triangle)
    
    # Add subtle grid pattern
    for i in range(20):
        x = i / 20
        ax.axvline(x, color='white', linewidth=0.4, alpha=0.08)
    for i in range(28):
        y = i / 28
        ax.axhline(y, color='white', linewidth=0.4, alpha=0.08)
    
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0, 
                facecolor='white', edgecolor='none', format='pdf')
    plt.close()
    
    print(f"✅ Generated cover background: {output_path}")
    print(f"   Size: {fig_width}\" x {fig_height}\" (A4)")
    print(f"   DPI: {dpi}")

if __name__ == '__main__':
    import os
    import sys
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_path = os.path.join(project_root, 'img', 'cover-background.pdf')
    
    # Create img directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    create_cover_background(output_path)
    print(f"\n✅ Cover background saved to: {output_path}")

