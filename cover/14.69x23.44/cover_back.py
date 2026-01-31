import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib
import numpy as np
from scipy.spatial import Delaunay
import random
from pathlib import Path

def generate_book_back_cover():
    # 1. Setup the Canvas (Full Bleed Fix)
    # Match cover.pdf dimensions: 14.69 x 23.44 inches (1057.44 x 1687.44 points)
    width, height = 14.69, 23.44
    dpi = 300 # Print quality resolution
    
    # Create figure without default subplots to control exact placement
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    
    # Add axes that cover 100% of the figure [left, bottom, width, height]
    # This ensures no white margins appear at the edges
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Turn off axis markings and set strict limits
    ax.set_axis_off()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    # ---------------------------------------------------------
    # 2. Generate Low-Poly Background (same as front)
    # ---------------------------------------------------------
    n_x, n_y = 12, 18
    # Extend grid slightly beyond limits (-1 to width+1) to ensure edges are filled
    x = np.linspace(-1, width+1, n_x)
    y = np.linspace(-1, height+1, n_y)
    grid_x, grid_y = np.meshgrid(x, y)
    points = np.vstack([grid_x.flatten(), grid_y.flatten()]).T

    jitter_strength = 0.6
    np.random.seed(42) # Fixed seed for consistency
    
    for i, point in enumerate(points):
        points[i][0] += np.random.uniform(-jitter_strength, jitter_strength)
        points[i][1] += np.random.uniform(-jitter_strength, jitter_strength)

    tri = Delaunay(points)
    
    # Center point for the radial gradient (slightly above middle)
    center = np.array([width / 2, height / 2 + 1])
    
    # Handle colormap retrieval safely across versions
    try:
        cmap = matplotlib.colormaps['viridis_r'] 
    except:
        cmap = plt.get_cmap('viridis_r') 

    for simplex in tri.simplices:
        triangle_points = points[simplex]
        centroid = np.mean(triangle_points, axis=0)
        
        # Calculate distance for gradient
        dist = np.linalg.norm(centroid - center)
        max_dist = np.linalg.norm(np.array([0, 0]) - center)
        norm_dist = dist / (max_dist * 0.9)
        
        # Clamp values to 0-1 range, then map to lighter range (0.4-1.0)
        # This makes the overall background lighter, especially at edges
        val = np.clip(norm_dist, 0, 1)
        # Map to lighter range: 0.4-1.0 instead of 0-1
        val = 0.4 + val * 0.6
        color = cmap(val)
        
        # Further lighten by mixing with white (reduce intensity)
        # Convert RGBA to RGB, mix with white, then back to RGBA
        color_rgb = np.array(color[:3])
        white = np.array([1.0, 1.0, 1.0])
        # Mix 60% original color with 40% white for lighter appearance
        color_lighter = 0.6 * color_rgb + 0.4 * white
        color_final = tuple(color_lighter) + (color[3],)  # Keep original alpha
        
        poly = patches.Polygon(triangle_points, closed=True, 
                               facecolor=color_final, edgecolor=color_final, alpha=1.0)
        ax.add_patch(poly)

    # ---------------------------------------------------------
    # 3. Add Floating Math Symbols and Formulas (lighter)
    # ---------------------------------------------------------
    # Use a subset of formulas for the back cover
    symbols = [
        r'$A = U\Sigma V^\top$',  # SVD
        r'$\nabla f(\mathbf{x})$',  # Gradient
        r'$H(X) = -\sum p\log p$',  # Entropy
        r'$D_{KL}(p||q)$',  # KL Divergence
        r'$d\mathbf{X}_t = \mathbf{f}dt + \mathbf{G}d\mathbf{W}_t$',  # SDE
        r'$p(x|y) = \frac{p(y|x)p(x)}{p(y)}$',  # Bayes
        r'$A = Q\Lambda Q^\top$',  # Spectral theorem
        r'$\mathbf{x}^\top A\mathbf{x}$',  # Quadratic form
        r'$\nabla^2 f$',  # Hessian
        r'$\mathrm{ELBO}$',  # ELBO
    ]
    
    for i in range(20):  # Fewer symbols on back cover
        sym = symbols[i % len(symbols)]
        sx = random.uniform(0, width)
        sy = random.uniform(0, height)
        
        # Avoid main text area (center region)
        if not (2 * (width / 8) < sx < 6 * (width / 8) and 
                0.25 * height < sy < 0.85 * height):
            size_scale = height / 12.5
            size = random.randint(int(10 * size_scale), int(20 * size_scale))
            rot = random.randint(-45, 45)
            ax.text(sx, sy, sym, fontsize=size, 
                    color='white', alpha=0.08, rotation=rot)

    # ---------------------------------------------------------
    # 4. Add Typography for Back Cover
    # ---------------------------------------------------------
    scale_y = height / 10
    
    # Main title at top
    title_y = 0.82 * height
    ax.text(width/2, title_y, "MATHEMATICS FOR AI/ML", 
            ha='center', fontsize=int(24 * scale_y), fontname='DejaVu Sans', 
            weight='bold', color='black', zorder=21)
    
    # Subtitle
    subtitle_y = 0.78 * height
    ax.text(width/2, subtitle_y, "Second Edition", 
            ha='center', fontsize=int(12 * scale_y), fontname='DejaVu Sans', 
            style='italic', color='black', zorder=21)
    
    # Book description (简介) - with box
    desc_box_padding = 0.3
    desc_box_width = width * 0.75
    desc_box_x = (width - desc_box_width) / 2
    desc_text = (
        "This comprehensive textbook provides a rigorous mathematical\n"
        "foundation for modern artificial intelligence and machine learning.\n"
        "Covering topics from linear algebra and matrix calculus to stochastic\n"
        "differential equations and generative models, this book bridges the\n"
        "gap between mathematical theory and practical applications in\n"
        "deep learning, reinforcement learning, and large language models."
    )
    desc_lines = desc_text.count('\n') + 1
    desc_box_height = desc_lines * int(9 * scale_y) * 1.4 / 72 + desc_box_padding * 2
    desc_box_y = 0.65 * height - desc_box_height
    desc_box = patches.FancyBboxPatch(
        (desc_box_x, desc_box_y), desc_box_width, desc_box_height,
        boxstyle="round,pad=0.02", linewidth=2, edgecolor='black',
        facecolor='none', zorder=20
    )
    ax.add_patch(desc_box)
    
    # Text inside description box (left-aligned)
    desc_text_x = desc_box_x + desc_box_padding
    desc_text_y = desc_box_y + desc_box_height - desc_box_padding
    ax.text(desc_text_x, desc_text_y, desc_text, 
            ha='left', va='top', fontsize=int(9 * scale_y), 
            fontname='DejaVu Sans', color='black', zorder=21,
            linespacing=1.4)
    
    # Key Features - with box
    features_box_padding = 0.3
    features_box_width = width * 0.75
    features_box_x = (width - features_box_width) / 2
    features_text = (
        "Key Features:\n"
        "• Complete coverage of linear algebra, optimization, and probability\n"
        "• Advanced topics: SDEs, Fokker-Planck equations, and score-based\n"
        "   models\n"
        "• Extensive exercises and problem sets for self-study\n"
        "• Mathematical rigor with practical insights for AI/ML practitioners"
    )
    features_lines = features_text.count('\n') + 1
    features_box_height = features_lines * int(9 * scale_y) * 1.4 / 72 + features_box_padding * 2
    features_box_y = 0.40 * height - features_box_height
    features_box = patches.FancyBboxPatch(
        (features_box_x, features_box_y), features_box_width, features_box_height,
        boxstyle="round,pad=0.02", linewidth=2, edgecolor='black',
        facecolor='none', zorder=20
    )
    ax.add_patch(features_box)
    
    # Text inside Key Features box (left-aligned)
    features_text_x = features_box_x + features_box_padding
    features_text_y = features_box_y + features_box_height - features_box_padding
    ax.text(features_text_x, features_text_y, features_text, 
            ha='left', va='top', fontsize=int(9 * scale_y), 
            fontname='DejaVu Sans', color='black', zorder=21,
            linespacing=1.4)
    

    # ---------------------------------------------------------
    # Finalize
    # ---------------------------------------------------------
    # Save to cover/ directory (same directory as this script)
    script_dir = Path(__file__).parent
    output_path = script_dir / "cover_back.pdf"
    plt.savefig(str(output_path), dpi=dpi, format='pdf')
    print(f"Back cover generated: {output_path}")
    plt.close()

if __name__ == "__main__":
    generate_book_back_cover()
