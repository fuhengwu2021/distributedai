import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib
import numpy as np
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d import Axes3D
import random
from pathlib import Path
from matplotlib import font_manager
from matplotlib import patheffects

def draw_text_with_outline(ax, x, y, text, fontsize, fontname='DejaVu Sans', 
                           weight='bold', color='black', outline_color='white', 
                           outline_width=0.15, zorder=21):
    """
    Draw text with an outline/stroke effect for a cool look.
    Creates a rounded, bold appearance by drawing the text multiple times with offsets.
    """
    # Draw outline by drawing text in multiple directions
    offsets = [
        (-outline_width, 0), (outline_width, 0),  # Left, Right
        (0, -outline_width), (0, outline_width),   # Bottom, Top
        (-outline_width*0.7, -outline_width*0.7),  # Bottom-left
        (outline_width*0.7, -outline_width*0.7),   # Bottom-right
        (-outline_width*0.7, outline_width*0.7),  # Top-left
        (outline_width*0.7, outline_width*0.7),    # Top-right
    ]
    
    # Draw outline (behind main text)
    for dx, dy in offsets:
        ax.text(x + dx, y + dy, text, ha='center', va='center',
                fontsize=fontsize, fontname=fontname, weight=weight,
                color=outline_color, zorder=zorder-1)
    
    # Draw main text (on top)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, fontname=fontname, weight=weight,
            color=color, zorder=zorder)

def get_academic_font():
    """Try to find an academic/serious font, fallback to DejaVu Sans if not available."""
    # List of academic fonts to try (in order of preference)
    # These fonts have a more serious, professional, academic appearance
    academic_fonts = ['Inter', 'Source Sans Pro', 'Helvetica Now', 'Helvetica Neue',
                      'Arial', 'Liberation Sans', 'DejaVu Sans']
    
    # Get available fonts
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    
    # Try to find an academic font
    for font in academic_fonts:
        if font in available_fonts:
            return font
    
    # Fallback to DejaVu Sans (which is always available)
    return 'DejaVu Sans'

def generate_book_front_cover():
    # 1. Setup the Canvas (Full Bleed Fix)
    # Match book.pdf dimensions: 7 x 10 inches (Amazon KDP size)
    width, height = 7.0, 10.0
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
    # 2. Generate Low-Poly Background
    # ---------------------------------------------------------
    n_x, n_y = 8, 10
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
    # 3. Add Floating Math Symbols and Formulas
    # ---------------------------------------------------------
    # 35 formulas from the book chapters - mix of symbols and key equations
    symbols = [
        # Linear Algebra (Chapters 1-2, 3-4)
        r'$\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^\top \mathbf{v}$',  # Inner product
        r'$\|\mathbf{v}\|_2 = \sqrt{\mathbf{v}^\top \mathbf{v}}$',  # L2 norm
        r'$A\mathbf{x} = \sum \mathbf{a}_i x_i$',  # Matrix-vector product
        r'$A = U\Sigma V^\top$',  # SVD (Chapter 8)
        r'$A = QR$',  # QR decomposition (Chapter 4)
        r'$A = LU$',  # LU decomposition (Chapter 5)
        r'$\mathrm{proj}_A(\mathbf{x})$',  # Projection (Chapter 3)
        r'$\dim(C(A)) + \dim(N(A)) = N$',  # Rank-nullity (Chapter 3)
        # Eigenvalues & Decomposition (Chapters 6-7)
        r'$A\mathbf{x} = \lambda \mathbf{x}$',  # Eigenvalue equation
        r'$A = Q\Lambda Q^\top$',  # Spectral theorem
        r'$\mathbf{x}^\top A\mathbf{x}$',  # Quadratic form
        r'$\det(A) = \prod \lambda_i$',  # Determinant-eigenvalue
        r'$\mathrm{tr}(A) = \sum \lambda_i$',  # Trace-eigenvalue
        # Matrix Calculus (Chapter 11)
        r'$\nabla f(\mathbf{x})$',  # Gradient
        r'$\mathbf{J}_f$',  # Jacobian
        r'$\nabla^2 f$',  # Hessian
        r"$(g \circ f)' = g' \cdot f'$",  # Chain rule
        r'$\frac{d}{dX}\mathrm{tr}(AX) = A^\top$',  # Trace derivative
        # Optimization (Chapter 12)
        r'$\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\nabla f$',  # Gradient descent
        r'$\nabla^2 f \succeq 0$',  # Convexity
        r'$f(y) \geq f(x) + \nabla f(x)^\top(y-x)$',  # Subgradient
        # Probability (Chapter 13)
        r'$p(x|y) = \frac{p(y|x)p(x)}{p(y)}$',  # Bayes rule
        r'$\mathcal{N}(\mu, \Sigma)$',  # Gaussian
        r'$\mathbb{E}[X]$',  # Expectation
        r'$\mathrm{Var}[X] = \mathbb{E}[(X-\mathbb{E}[X])^2]$',  # Variance
        r'$\mathrm{Cov}[X,Y] = \mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])]$',  # Covariance
        # Information Theory (Chapter 14)
        r'$H(X) = -\sum p\log p$',  # Entropy
        r'$D_{KL}(p||q)$',  # KL Divergence
        r'$I(X;Y) = H(X) - H(X|Y)$',  # Mutual information
        r'$\mathrm{ELBO}$',  # ELBO
        # Variational Inference (Chapter 16)
        r'$q(\mathbf{z}|\mathbf{x})$',  # Variational posterior
        # Score & Energy (Chapter 17)
        r'$\nabla \log p(\mathbf{x})$',  # Score function
        r'$p \propto \exp(-E)$',  # Energy-based model
        # Langevin & Sampling (Chapter 18)
        r'$\mathbf{x}_{k+1} = \mathbf{x}_k + \eta\mathbf{s} + \sqrt{2\eta}\epsilon$',  # Langevin
        # SDE (Chapter 19)
        r'$d\mathbf{X}_t = \mathbf{f}dt + \mathbf{G}d\mathbf{W}_t$',  # SDE
        r'$d\mathbf{X}_t = [\mathbf{f} - \mathbf{a}\nabla\log p_t]dt$',  # Reverse SDE
        # ODE & Continuous Limits (Chapter 20)
        r'$\frac{d\mathbf{x}_t}{dt} = -\nabla f(\mathbf{x}_t)$',  # Gradient flow ODE
        r'$\frac{d\mathbf{X}_t}{dt} = \mathbf{f} - \frac{1}{2}\mathbf{a}\nabla\log p_t$',  # Probability flow ODE
        # Fokker-Planck (Chapter 21)
        r'$\partial_t p_t = -\nabla \cdot (\mathbf{f}p_t) + \frac{1}{2}\nabla^2(\mathbf{D}p_t)$',  # Fokker-Planck
        r'$\mathbf{D} = \mathbf{G}\mathbf{G}^\top$',  # Diffusion matrix
        r'$\partial_t p_t + \nabla \cdot \mathbf{J}_t = 0$',  # Continuity equation
    ]
    
    for i in range(35):
        # Loop through symbols by index, cycling back to start when needed
        sym = symbols[i % len(symbols)]
        sx = random.uniform(0, width)
        sy = random.uniform(0, height)
        
        # Avoid placing symbols over the main title area (updated positions)
        # Title area is now around 0.47-0.70 of height
        if not (1 * (width / 8) < sx < 7 * (width / 8) and 
                0.47 * height < sy < 0.72 * height):
            # Scale font size based on height (for 8.5x11 page)
            size_scale = height / 11.0
            size = random.randint(int(8 * size_scale), int(16 * size_scale))
            rot = random.randint(-45, 45)
            ax.text(sx, sy, sym, fontsize=size, 
                    color='white', alpha=0.1, rotation=rot)

    # ---------------------------------------------------------
    # 4. Add 3D Surface Plot (similar to jacobian.py)
    # ---------------------------------------------------------
    # Create a 3D surface plot in the bottom area
    # Position: moved upward for better balance
    math_plot_x_center = width / 2
    math_plot_y_center = 3.0  # Moved up from 1.5 for 7x10 page
    math_plot_width = 3.5  # Smaller for 7x10 page
    math_plot_height = 2.0  # Smaller for 7x10 page
    
    # Create a new axes for 3D plot positioned at the bottom center
    # Position: [left, bottom, width, height] in figure coordinates (0-1)
    ax3d_left = (math_plot_x_center - math_plot_width/2) / width
    ax3d_bottom = (math_plot_y_center - math_plot_height/2) / height
    ax3d_width_frac = math_plot_width / width
    ax3d_height_frac = math_plot_height / height
    
    ax3d = fig.add_axes([ax3d_left, ax3d_bottom, ax3d_width_frac, ax3d_height_frac], 
                        projection='3d')
    
    # Create data for the mathematical function
    x_range = np.linspace(-1, 1, 40)
    y_range = np.linspace(-1, 1, 40)
    X_math, Y_math = np.meshgrid(x_range, y_range)
    
    # Function: f(x,y) = -5 * x * y * exp(-x^2 - y^2)
    Z_math = -5 * X_math * Y_math * np.exp(-X_math**2 - Y_math**2)
    
    # Plot 3D surface with a blue colormap
    surf = ax3d.plot_surface(X_math, Y_math, Z_math, cmap='Blues', 
                             edgecolor='none', rstride=2, cstride=2, alpha=0.8)
    
    # Style the 3D axes - make background transparent
    ax3d.set_axis_off()  # Hide axes for cleaner look
    ax3d.patch.set_facecolor('none')  # Transparent background
    ax3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Transparent x-axis pane
    ax3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Transparent y-axis pane
    ax3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Transparent z-axis pane
    ax3d.view_init(elev=30, azim=-70)  # Set viewing angle

    # ---------------------------------------------------------
    # 5. Add Typography (from cover_v2.py)
    # ---------------------------------------------------------
    # Scale factor: now using 8.5x11 directly
    # Convert relative positions (0-1) to absolute positions
    scale_x = 1.0
    scale_y = height / 10
    
    # Get academic font for title
    academic_font = get_academic_font()
    
    # 1. MAIN TITLE - Academic font, bold, clear structure
    title1_y = 0.70 * height
    title2_y = 0.63 * height
    title_fontsize = int(32 * scale_y)
    
    ax.text(width/2, title1_y, "DISTRIBUTED", 
            ha='center', fontsize=title_fontsize, fontname=academic_font, 
            weight='bold', color='black', zorder=21)
    
    ax.text(width/2, title2_y, "AI SYSTEMS", 
            ha='center', fontsize=title_fontsize, fontname=academic_font, 
            weight='bold', color='black', zorder=21)

    # 2. Edition - below title
    edition_y = 0.55 * height
    ax.text(width/2, edition_y, "First Edition", 
            ha='center', fontsize=int(11 * scale_y), fontname=academic_font,
            style='italic', color='black', zorder=21)

    # 3. Subtitle - single line, graduate-level reference
    subtitle_y = 0.47 * height
    subtitle_text = "A practical guide to building scalable \ntraining, inference, and serving systems for production AI"
    ax.text(width/2, subtitle_y, subtitle_text, 
            ha='center', va='top', fontsize=int(10 * scale_y), 
            fontname='DejaVu Sans', color='black', zorder=21,
            linespacing=1.3)

    # 4. Author - subtle, at bottom
    author_y = 0.15 * height
    ax.text(width/2, author_y, "By Henry (Fuheng) Wu",
            ha='center', fontsize=int(9 * scale_y), fontname='DejaVu Sans', 
            color='black', zorder=21)

    # 5. Year - subtle, at very bottom
    footer_y = 0.10 * height
    ax.text(width/2, footer_y, "2026", 
            ha='center', fontsize=int(10 * scale_y), fontname='DejaVu Sans', 
            color='black', zorder=21)

    # ---------------------------------------------------------
    # Finalize
    # ---------------------------------------------------------
    # Note: We do NOT use bbox_inches='tight' here because 
    # we manually set axes to [0,0,1,1].
    # Save as PDF with same dimensions as cover.pdf
    # Save to cover/ directory (same directory as this script)
    script_dir = Path(__file__).parent
    output_path = script_dir / "cover_front.pdf"
    plt.savefig(str(output_path), dpi=dpi, format='pdf')
    print(f"Front cover generated: {output_path}")
    # Optionally also save PNG
    # plt.savefig("math_book_cover.png", dpi=dpi)
    # print("Cover generated: math_book_cover.png")
    # plt.show() # Uncomment if running in a notebook/IDE with display

if __name__ == "__main__":
    generate_book_front_cover()