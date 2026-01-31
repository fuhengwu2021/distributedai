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
    # 2. Generate Low-Poly Background (same as front)
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
    # 3. Add Floating Technical Symbols and Formulas (lighter)
    # ---------------------------------------------------------
    # Use distributed systems and AI-related symbols for the back cover
    symbols = [
        r'$\sum_{i=1}^{N} \nabla_i$',  # Distributed gradients
        r'$\mathbf{W} = \mathrm{Shard}(\mathbf{W}_1, \ldots, \mathbf{W}_N)$',  # Model sharding
        r'$\mathrm{AllReduce}(\mathbf{g})$',  # Collective communication
        r'$\mathrm{KV}$',  # KV Cache
        r'$\mathrm{FSDP}$',  # FSDP
        r'$\mathrm{DDP}$',  # DDP
        r'$\mathrm{TP} \times \mathrm{PP} \times \mathrm{DP}$',  # Parallelism
        r'$\mathrm{Throughput} = \frac{N}{T}$',  # Throughput
        r'$\mathrm{Latency} = f(b, s)$',  # Latency
        r'$\mathrm{GPU} \times N$',  # Multi-GPU
    ]
    
    for i in range(20):  # Fewer symbols on back cover
        sym = symbols[i % len(symbols)]
        sx = random.uniform(0, width)
        sy = random.uniform(0, height)
        
        # Avoid main text area (center region)
        if not (2 * (width / 8) < sx < 6 * (width / 8) and 
                0.25 * height < sy < 0.85 * height):
            size_scale = height / 10.0
            size = random.randint(int(7 * size_scale), int(14 * size_scale))
            rot = random.randint(-45, 45)
            ax.text(sx, sy, sym, fontsize=size, 
                    color='white', alpha=0.08, rotation=rot)

    # ---------------------------------------------------------
    # 4. Add Typography for Back Cover
    # ---------------------------------------------------------
    scale_y = height / 10
    
    # Main title at top
    title_y = 0.82 * height
    ax.text(width/2, title_y, "DISTRIBUTED AI SYSTEMS", 
            ha='center', fontsize=int(24 * scale_y), fontname='DejaVu Sans', 
            weight='bold', color='black', zorder=21)
    
    # Subtitle
    subtitle_y = 0.78 * height
    ax.text(width/2, subtitle_y, "Build scalable training, inference and serving systems", 
            ha='center', fontsize=int(11 * scale_y), fontname='DejaVu Sans', 
            style='italic', color='black', zorder=21)
    
    # Book description - with box
    desc_box_padding = 0.165
    desc_box_width = width * 0.70  # Wider box
    desc_box_x = (width - desc_box_width) / 2
    desc_text = (
        "With large AI models now reaching billions or even trillions of parameters,\n"
        "distributed systems have become essential for training and serving AI.\n"
        "This comprehensive guide bridges the gap between theory and practice\n"
        "with hands-on code examples demonstrating production-grade techniques\n"
        "used in real-world systems, from distributed training through inference\n"
        "to production serving."
    )
    desc_lines = desc_text.count('\n') + 1
    desc_box_height = desc_lines * int(9 * scale_y) * 1.4 / 72 + desc_box_padding * 2
    desc_box_y = 0.75 * height - desc_box_height
    # Corner radius in inches (adjust as needed)
    corner_radius = 0.15
    desc_box = patches.FancyBboxPatch(
        (desc_box_x, desc_box_y), desc_box_width, desc_box_height,
        boxstyle=f"round,pad=0.02,rounding_size={corner_radius}", 
        linewidth=1, edgecolor='grey',
        facecolor='white', alpha=0.5, zorder=20
    )
    ax.add_patch(desc_box)
    
    # Text inside description box (left-aligned)
    desc_text_x = desc_box_x + desc_box_padding
    desc_text_y = desc_box_y + desc_box_height - desc_box_padding
    ax.text(desc_text_x, desc_text_y, desc_text, 
            ha='left', va='top', fontsize=int(9 * scale_y), 
            fontname='DejaVu Sans', color='black', zorder=21,
            linespacing=1.4)
    
    # Key Features and Intended Audience - merged into one box
    combined_box_padding = 0.165
    combined_box_width = width * 0.70  # Wider box
    combined_box_x = (width - combined_box_width) / 2
    # Split content for rendering headers in bold separately
    features_header = "Key Features"
    features_content = (
        "• Master distributed training with DDP, FSDP, DeepSpeed, and Megatron\n"
        "• Build high-performance inference systems with vLLM and SGLang\n"
        "• Understand GPU hardware, interconnects, and parallelism strategies\n"
        "• Deploy production serving stacks with orchestration and observability\n"
        "• Hands-on code examples tested on real infrastructure throughout"
    )
    audience_header = "Intended Audience"
    audience_content = (
        "Designed for ML engineers, AI researchers, and DevOps engineers who\n"
        "need to train or serve large AI models at scale. Platform engineers,\n"
        "HP Cluster administrators, and cloud architects will advance their\n"
        "skills with this guide."
    )
    # Calculate total lines for box height
    combined_lines = features_content.count('\n') + audience_content.count('\n') + 6  # +6 for headers and spacing
    combined_text = (
        features_header + "\n\n" + features_content + "\n\n" + 
        audience_header + "\n\n" + audience_content
    )
    combined_lines = combined_text.count('\n') + 1
    combined_box_height = combined_lines * int(9 * scale_y) * 1.4 / 72 + combined_box_padding * 2
    combined_box_y = 0.50 * height - combined_box_height
    # Use same corner radius for consistency
    combined_box = patches.FancyBboxPatch(
        (combined_box_x, combined_box_y), combined_box_width, combined_box_height,
        boxstyle=f"round,pad=0.02,rounding_size={corner_radius}", 
        linewidth=1, edgecolor='grey',
        facecolor='white', alpha=0.5, zorder=20
    )
    ax.add_patch(combined_box)
    
    # Text inside combined box - render with bold headers
    combined_text_x = combined_box_x + combined_box_padding
    combined_text_y = combined_box_y + combined_box_height - combined_box_padding
    
    # Render text with bold headers using weight parameter
    # Split the text and render headers separately
    y_pos = combined_text_y
    line_height = int(9 * scale_y) * 1.4 / 72
    
    # "Key Features" header in bold
    ax.text(combined_text_x, y_pos, features_header, 
            ha='left', va='top', fontsize=int(9 * scale_y), 
            fontname='DejaVu Sans', color='black', zorder=21,
            weight='bold')
    y_pos -= line_height * 2  # Skip header line and one blank line
    
    # Features content
    ax.text(combined_text_x, y_pos, features_content, 
            ha='left', va='top', fontsize=int(9 * scale_y), 
            fontname='DejaVu Sans', color='black', zorder=21,
            linespacing=1.4)
    y_pos -= (features_content.count('\n') + 1) * line_height
    y_pos -= line_height * 2  # Skip two blank lines
    
    # "Intended Audience" header in bold
    ax.text(combined_text_x, y_pos, audience_header, 
            ha='left', va='top', fontsize=int(9 * scale_y), 
            fontname='DejaVu Sans', color='black', zorder=21,
            weight='bold')
    y_pos -= line_height * 2  # Skip header line and one blank line
    
    # Audience content
    ax.text(combined_text_x, y_pos, audience_content, 
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
