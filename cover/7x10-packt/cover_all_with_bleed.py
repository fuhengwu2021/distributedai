#!/usr/bin/env python3
"""
Create a wrap-around book cover PDF with KDP bleed support:
- Front cover (cover_front.pdf)
- Spine (generated)
- Back cover (cover_back.pdf)

KDP Requirements:
- Bleed: 0.125 inch on all sides
- Final canvas: (width + 0.25) x (height + 0.25) inches
- All content stays in same position (offset by 0.125" from edges)
- Background extends to cover bleed areas

Output: cover_all_with_bleed.pdf
"""

import sys
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib
import numpy as np
from scipy.spatial import Delaunay

# KDP bleed requirement: 0.125 inch on all sides
BLEED_INCHES = 0.125
BLEED_POINTS = BLEED_INCHES * 72

def get_pdf_dimensions(pdf_path):
    """Get the dimensions of a PDF page in points."""
    reader = PdfReader(str(pdf_path))
    page = reader.pages[0]
    mediabox = page.mediabox
    width = float(mediabox.width)
    height = float(mediabox.height)
    return width, height

def create_spine_pdf_with_bleed(spine_width_points, height_points, output_path):
    """Create a PDF for the book spine with matching background style and bleed."""
    # Convert points to inches
    spine_width_inches = spine_width_points / 72.0
    height_inches = height_points / 72.0
    width = spine_width_inches
    height = height_inches
    
    # Add bleed to canvas size
    canvas_width = width + 2 * BLEED_INCHES
    canvas_height = height + 2 * BLEED_INCHES
    
    dpi = 300
    fig = plt.figure(figsize=(canvas_width, canvas_height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(-BLEED_INCHES, width + BLEED_INCHES)
    ax.set_ylim(-BLEED_INCHES, height + BLEED_INCHES)
    
    # Generate Low-Poly Background extended to cover bleed area
    n_x, n_y = 4, 18  # Fewer x points since spine is narrow
    # Extend grid to cover bleed area
    x = np.linspace(-BLEED_INCHES - 0.5, width + BLEED_INCHES + 0.5, n_x)
    y = np.linspace(-BLEED_INCHES - 1, height + BLEED_INCHES + 1, n_y)
    grid_x, grid_y = np.meshgrid(x, y)
    points = np.vstack([grid_x.flatten(), grid_y.flatten()]).T
    
    jitter_strength = 0.4
    np.random.seed(42)  # Same seed for consistency
    
    for i, point in enumerate(points):
        points[i][0] += np.random.uniform(-jitter_strength, jitter_strength)
        points[i][1] += np.random.uniform(-jitter_strength, jitter_strength)
    
    tri = Delaunay(points)
    
    # Center point for the radial gradient (in content area)
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
        max_dist = np.linalg.norm(np.array([-BLEED_INCHES, -BLEED_INCHES]) - center)
        norm_dist = dist / (max_dist * 0.9)
        
        # Clamp values to 0-1 range, then map to lighter range (0.4-1.0)
        val = np.clip(norm_dist, 0, 1)
        val = 0.4 + val * 0.6
        color = cmap(val)
        
        # Further lighten by mixing with white
        color_rgb = np.array(color[:3])
        white = np.array([1.0, 1.0, 1.0])
        color_lighter = 0.6 * color_rgb + 0.4 * white
        color_final = tuple(color_lighter) + (color[3],)
        
        poly = patches.Polygon(triangle_points, closed=True,
                               facecolor=color_final, edgecolor=color_final, alpha=1.0)
        ax.add_patch(poly)
    
    # Add title text on spine (rotated 90 degrees)
    # Text position is in content area (not in bleed)
    width_points = width * 72  # Convert inches to points
    title_font_size = max(8, int(width_points * 0.28))
    author_font_size = max(6, int(width_points * 0.2))
    
    # Title at top (in content area, not bleed)
    top_padding = 0.07 * height
    title_y = height - top_padding
    ax.text(width/2, title_y, "MATHEMATICS FOR AI/ML",
            ha='center', va='top', fontsize=title_font_size,
            fontname='DejaVu Sans', weight='bold', color='black', zorder=21,
            rotation=-90)
    
    # Author at bottom (in content area, not bleed)
    bottom_padding = 0.05 * height
    author_y = bottom_padding
    ax.text(width/2, author_y, "Henry Fuheng Wu",
            ha='center', va='bottom', fontsize=author_font_size,
            fontname='DejaVu Sans', weight='bold', color='black', zorder=21,
            rotation=-90)
    
    # Save with exact figure dimensions (including bleed)
    plt.savefig(str(output_path), dpi=dpi, format='pdf',
                facecolor='white', edgecolor='none',
                bbox_inches='tight', pad_inches=0)
    plt.close()

def merge_covers_with_bleed(front_pdf, spine_pdf, back_pdf, output_pdf, spine_content_width_points):
    """
    Merge covers with KDP bleed support.
    Creates a canvas with bleed and places covers with 0.125" offset.
    
    Args:
        spine_content_width_points: The content width of the spine (without bleed) in points
    """
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    import tempfile
    import shutil
    
    # Get dimensions from PDFs
    # Front and back: content dimensions (no bleed)
    front_width, front_height = get_pdf_dimensions(front_pdf)
    back_width, back_height = get_pdf_dimensions(back_pdf)
    
    # Spine PDF has bleed, but we use content width for positioning
    spine_pdf_width, spine_pdf_height = get_pdf_dimensions(spine_pdf)
    spine_content_width = spine_content_width_points  # Use provided content width
    
    # Calculate content dimensions (without bleed)
    content_width = front_width + spine_content_width + back_width
    content_height = front_height
    
    # Calculate final canvas dimensions with bleed
    final_width_inches = (content_width / 72.0) + 2 * BLEED_INCHES
    final_height_inches = (content_height / 72.0) + 2 * BLEED_INCHES
    
    print(f"   Content size: {content_width/72:.2f}\" x {content_height/72:.2f}\"")
    print(f"   Final canvas with bleed: {final_width_inches:.2f}\" x {final_height_inches:.2f}\"")
    
    # Create canvas with bleed
    c = canvas.Canvas(str(output_pdf), pagesize=(final_width_inches * inch, final_height_inches * inch))
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Try pdftoppm first (usually more reliable)
        try:
            import subprocess
            
            # Convert PDFs to PNG
            subprocess.run(['pdftoppm', '-png', '-r', '300', '-singlefile',
                          str(front_pdf), str(temp_dir / 'front')], check=True, 
                          capture_output=True)
            subprocess.run(['pdftoppm', '-png', '-r', '300', '-singlefile',
                          str(spine_pdf), str(temp_dir / 'spine')], check=True,
                          capture_output=True)
            subprocess.run(['pdftoppm', '-png', '-r', '300', '-singlefile',
                          str(back_pdf), str(temp_dir / 'back')], check=True,
                          capture_output=True)
            
            front_img = temp_dir / 'front.png'
            spine_img = temp_dir / 'spine.png'
            back_img = temp_dir / 'back.png'
            
            if front_img.exists() and spine_img.exists() and back_img.exists():
                # Order: back (left) + spine (middle) + front (right)
                # Place with BLEED_INCHES offset from left and bottom
                x_offset = BLEED_INCHES * inch
                y_offset = BLEED_INCHES * inch
                
                c.drawImage(str(back_img), x_offset, y_offset, 
                           width=back_width, height=back_height)
                # Spine: use content width for positioning, but draw full PDF (with bleed)
                c.drawImage(str(spine_img), x_offset + back_width, y_offset, 
                           width=spine_content_width, height=back_height)
                c.drawImage(str(front_img), x_offset + back_width + spine_content_width, y_offset,
                           width=front_width, height=back_height)
                c.save()
                shutil.rmtree(temp_dir)
                return True
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Try pdf2image as fallback
        try:
            from pdf2image import convert_from_path
            
            front_img = convert_from_path(str(front_pdf), dpi=300)[0]
            front_img_path = temp_dir / "front.png"
            front_img.save(front_img_path, 'PNG')
            
            spine_img = convert_from_path(str(spine_pdf), dpi=300)[0]
            spine_img_path = temp_dir / "spine.png"
            spine_img.save(spine_img_path, 'PNG')
            
            back_img = convert_from_path(str(back_pdf), dpi=300)[0]
            back_img_path = temp_dir / "back.png"
            back_img.save(back_img_path, 'PNG')
            
            # Order: back (left) + spine (middle) + front (right)
            # Place with BLEED_INCHES offset from left and bottom
            x_offset = BLEED_INCHES * inch
            y_offset = BLEED_INCHES * inch
            
            c.drawImage(str(back_img_path), x_offset, y_offset,
                       width=back_width, height=back_height, preserveAspectRatio=True)
            # Spine: use content width for positioning, but draw full PDF (with bleed)
            c.drawImage(str(spine_img_path), x_offset + back_width, y_offset,
                       width=spine_content_width, height=back_height, preserveAspectRatio=False)
            c.drawImage(str(front_img_path), x_offset + back_width + spine_content_width, y_offset,
                       width=front_width, height=back_height, preserveAspectRatio=True)
            c.save()
            shutil.rmtree(temp_dir)
            return True
            
        except ImportError:
            print("  Error: Need pdftoppm (poppler-utils) or pdf2image")
            print("    Install: sudo apt-get install poppler-utils")
            print("    Or: pip install pdf2image pillow")
            shutil.rmtree(temp_dir)
            return False
            
    except Exception as e:
        print(f"  Error: {e}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return False


def calculate_spine_width(pages):
    """
    Calculate spine width in inches based on page count.
    
    Based on recommended spine widths:
    - 500 pages: 1.25 in
    - 520 pages: 1.30 in
    - 550 pages: 1.38-1.40 in (default)
    - 580 pages: 1.45 in
    - 600 pages: 1.50 in (maximum)
    """
    page_widths = {
        500: 1.25,
        520: 1.30,
        550: 1.39,
        580: 1.45,
        600: 1.50
    }
    
    if pages in page_widths:
        return page_widths[pages]
    
    sorted_pages = sorted(page_widths.keys())
    
    if pages < sorted_pages[0]:
        return page_widths[sorted_pages[0]]
    
    if pages > sorted_pages[-1]:
        return page_widths[sorted_pages[-1]]
    
    for i in range(len(sorted_pages) - 1):
        if sorted_pages[i] <= pages <= sorted_pages[i + 1]:
            p1, p2 = sorted_pages[i], sorted_pages[i + 1]
            w1, w2 = page_widths[p1], page_widths[p2]
            ratio = (pages - p1) / (p2 - p1)
            width = w1 + ratio * (w2 - w1)
            return round(width, 2)
    
    return 1.39

def main(pages=550):
    """Main function to create wrap-around cover with KDP bleed."""
    script_dir = Path(__file__).parent
    
    front_pdf = script_dir / "cover_front.pdf"
    back_pdf = script_dir / "cover_back.pdf"
    output_pdf = script_dir / "cover_all_with_bleed.pdf"
    
    # Check if front and back covers exist
    if not front_pdf.exists():
        print(f"❌ Error: {front_pdf} not found")
        print("   Please run cover/7x10/cover_front.py first")
        sys.exit(1)
    
    if not back_pdf.exists():
        print(f"❌ Error: {back_pdf} not found")
        print("   Please run cover/7x10/cover_back.py first")
        sys.exit(1)
    
    print("📚 Creating wrap-around book cover with KDP bleed...")
    print(f"   Front: {front_pdf}")
    print(f"   Back: {back_pdf}")
    print(f"   Bleed: {BLEED_INCHES:.3f}\" on all sides")
    
    # Get dimensions from front cover
    front_width, front_height = get_pdf_dimensions(front_pdf)
    back_width, back_height = get_pdf_dimensions(back_pdf)
    
    # Ensure heights match
    if abs(front_height - back_height) > 1:
        print(f"⚠️  Warning: Height mismatch (front: {front_height:.1f}, back: {back_height:.1f})")
        print("   Using front cover height for all")
    
    height = front_height
    
    # Calculate spine width based on page count
    spine_width_inches = calculate_spine_width(pages)
    spine_width_points = spine_width_inches * 72
    
    print(f"   Pages: {pages}")
    print(f"   Spine width: {spine_width_inches:.2f} inches ({spine_width_points:.1f} points)")
    
    # Create spine PDF with bleed
    print("   Creating spine with bleed...")
    temp_spine = script_dir / "cover_spine_temp_bleed.pdf"
    create_spine_pdf_with_bleed(spine_width_points, height, temp_spine)
    
    # Merge all three PDFs with bleed
    print("   Merging front + spine + back with bleed...")
    success = merge_covers_with_bleed(front_pdf, temp_spine, back_pdf, output_pdf, spine_width_points)
    
    # Cleanup temp spine
    if temp_spine.exists():
        temp_spine.unlink()
    
    if success and output_pdf.exists():
        total_width, total_height = get_pdf_dimensions(output_pdf)
        total_width_inches = total_width / 72.0
        total_height_inches = total_height / 72.0
        
        # Calculate content dimensions (without bleed)
        content_width = (front_width + spine_width_points + back_width) / 72.0
        content_height = front_height / 72.0
        
        print(f"\n✅ Successfully created {output_pdf}")
        print(f"   Final canvas: {total_width_inches:.2f}\" x {total_height_inches:.2f}\"")
        print(f"   Content area: {content_width:.2f}\" x {content_height:.2f}\"")
        print(f"   Layout: Back ({back_width/72:.2f}\") + Spine ({spine_width_inches:.2f}\") + Front ({front_width/72:.2f}\")")
        print(f"   ✅ KDP-ready with {BLEED_INCHES:.3f}\" bleed on all sides")
    else:
        print(f"\n❌ Failed to create {output_pdf}")
        sys.exit(1)

if __name__ == "__main__":
    import sys
    # Support command-line argument for page count
    if len(sys.argv) > 1:
        try:
            pages = int(sys.argv[1])
        except ValueError:
            print(f"⚠️  Invalid page count: {sys.argv[1]}. Using default: 550 pages")
            pages = 550
    else:
        pages = 550  # Default: 550 pages
    
    main(pages)
