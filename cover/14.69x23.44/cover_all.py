#!/usr/bin/env python3
"""
Create a wrap-around book cover PDF combining:
- Front cover (cover_front.pdf)
- Spine (generated)
- Back cover (cover_back.pdf)

Output: cover_all.pdf
"""

import sys
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib
import numpy as np
from scipy.spatial import Delaunay

def get_pdf_dimensions(pdf_path):
    """Get the dimensions of a PDF page in points."""
    reader = PdfReader(str(pdf_path))
    page = reader.pages[0]
    mediabox = page.mediabox
    width = float(mediabox.width)
    height = float(mediabox.height)
    return width, height

def create_spine_pdf(spine_width_points, height_points, output_path):
    """Create a PDF for the book spine with matching background style."""
    # Convert points to inches
    spine_width_inches = spine_width_points / 72.0
    height_inches = height_points / 72.0
    width = spine_width_inches
    height = height_inches
    
    dpi = 300
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    
    # Generate Low-Poly Background (same as front/back covers)
    n_x, n_y = 4, 18  # Fewer x points since spine is narrow
    x = np.linspace(-0.5, width + 0.5, n_x)
    y = np.linspace(-1, height + 1, n_y)
    grid_x, grid_y = np.meshgrid(x, y)
    points = np.vstack([grid_x.flatten(), grid_y.flatten()]).T
    
    jitter_strength = 0.4
    np.random.seed(42)  # Same seed for consistency
    
    for i, point in enumerate(points):
        points[i][0] += np.random.uniform(-jitter_strength, jitter_strength)
        points[i][1] += np.random.uniform(-jitter_strength, jitter_strength)
    
    tri = Delaunay(points)
    
    # Center point for the radial gradient
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
    # Calculate font size based on spine width (since text is rotated 90 degrees)
    # Use a large font size that will occupy most of the spine width
    # Convert width from inches to points for font size calculation
    width_points = width * 72  # Convert inches to points
    # Title font size: use about 60% of spine width in points
    title_font_size = int(width_points * 0.6)
    # Author font size: use about 40% of spine width in points
    author_font_size = int(width_points * 0.4)
    
    # Title at top
    title_y = 0.78 * height
    ax.text(width/2, title_y, "MATHEMATICS FOR AI/ML",
            ha='center', va='center', fontsize=title_font_size,
            fontname='DejaVu Sans', weight='bold', color='black', zorder=21,
            rotation=-90)
    
    # Author at bottom
    author_y = 0.15 * height
    ax.text(width/2, author_y, "Henry Fuheng Wu",
            ha='center', va='center', fontsize=author_font_size,
            fontname='DejaVu Sans', weight='bold', color='black', zorder=21,
            rotation=-90)
    
    plt.savefig(str(output_path), dpi=dpi, format='pdf',
                bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()

def merge_covers_horizontally_simple(front_pdf, spine_pdf, back_pdf, output_pdf):
    """
    Simple merge using reportlab and pdf2image or pdftoppm.
    Falls back to PyPDF2 if image conversion is not available.
    """
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    import tempfile
    import shutil
    
    # Get dimensions
    front_width, front_height = get_pdf_dimensions(front_pdf)
    spine_width, spine_height = get_pdf_dimensions(spine_pdf)
    back_width, back_height = get_pdf_dimensions(back_pdf)
    
    total_width = front_width + spine_width + back_width
    total_width_inches = total_width / 72.0
    total_height_inches = front_height / 72.0
    
    # Create canvas
    c = canvas.Canvas(str(output_pdf), pagesize=(total_width_inches * inch, total_height_inches * inch))
    
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
                c.drawImage(str(back_img), 0, 0, width=back_width, height=back_height)
                c.drawImage(str(spine_img), back_width, 0, width=spine_width, height=spine_height)
                c.drawImage(str(front_img), back_width + spine_width, 0, 
                           width=front_width, height=front_height)
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
            c.drawImage(str(back_img_path), 0, 0, width=back_width, height=back_height)
            c.drawImage(str(spine_img_path), back_width, 0, width=spine_width, height=spine_height)
            c.drawImage(str(front_img_path), back_width + spine_width, 0,
                       width=front_width, height=front_height)
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


def main():
    """Main function to create wrap-around cover."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # All cover files are now in the cover/ directory
    front_pdf = script_dir / "cover_front.pdf"
    back_pdf = script_dir / "cover_back.pdf"
    output_pdf = script_dir / "cover_all.pdf"
    
    # Check if front and back covers exist
    if not front_pdf.exists():
        print(f"❌ Error: {front_pdf} not found")
        print("   Please run cover/cover_front.py first")
        sys.exit(1)
    
    if not back_pdf.exists():
        print(f"❌ Error: {back_pdf} not found")
        print("   Please run cover/cover_back.py first")
        sys.exit(1)
    
    print("📚 Creating wrap-around book cover...")
    print(f"   Front: {front_pdf}")
    print(f"   Back: {back_pdf}")
    
    # Get dimensions from front cover
    front_width, front_height = get_pdf_dimensions(front_pdf)
    back_width, back_height = get_pdf_dimensions(back_pdf)
    
    # Ensure heights match
    if abs(front_height - back_height) > 1:
        print(f"⚠️  Warning: Height mismatch (front: {front_height:.1f}, back: {back_height:.1f})")
        print("   Using front cover height for all")
    
    height = front_height
    
    # Calculate spine width (adjustable - typical: 0.5-2 inches depending on page count)
    # For a typical 500-page book: ~1 inch spine
    # You can adjust this value
    spine_width_inches = 1.0  # Adjust this based on your book's page count
    spine_width_points = spine_width_inches * 72
    
    print(f"   Spine width: {spine_width_inches:.2f} inches ({spine_width_points:.1f} points)")
    
    # Create spine PDF
    print("   Creating spine...")
    temp_spine = script_dir / "cover_spine_temp.pdf"
    create_spine_pdf(spine_width_points, height, temp_spine)
    
    # Merge all three PDFs
    print("   Merging front + spine + back...")
    success = merge_covers_horizontally_simple(front_pdf, temp_spine, back_pdf, output_pdf)
    
    # Cleanup temp spine
    if temp_spine.exists():
        temp_spine.unlink()
    
    if success and output_pdf.exists():
        total_width, _ = get_pdf_dimensions(output_pdf)
        total_width_inches = total_width / 72.0
        print(f"\n✅ Successfully created {output_pdf}")
        print(f"   Total width: {total_width_inches:.2f} inches")
        print(f"   Layout: Front ({front_width/72:.2f}\") + Spine ({spine_width_inches:.2f}\") + Back ({back_width/72:.2f}\")")
    else:
        print(f"\n❌ Failed to create {output_pdf}")
        sys.exit(1)

if __name__ == "__main__":
    main()
