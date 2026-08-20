#!/usr/bin/env python3
"""
Create a wrap-around Chinese book cover PDF combining:
- Front cover (cover_front_zh.pdf)
- Spine (generated in Chinese)
- Back cover (cover_back_zh.pdf)

Output: cover_all_zh.pdf
"""

import sys
from pathlib import Path
import tempfile
import shutil
import subprocess
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from PyPDF2 import PdfReader
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
import numpy as np
from scipy.spatial import Delaunay


def get_chinese_font():
    """Find available Chinese font on the system."""
    cjk_fonts = [
        'Noto Sans CJK SC',
        'Noto Serif CJK SC',
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'Droid Sans Fallback',
        'SimHei',
        'Microsoft YaHei',
        'PingFang SC',
        'Source Han Sans CN'
    ]
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    for font in cjk_fonts:
        if font in available_fonts:
            return font
    return 'DejaVu Sans'


def get_pdf_dimensions(pdf_path):
    """Get dimensions of a PDF page in points."""
    reader = PdfReader(str(pdf_path))
    page = reader.pages[0]
    mediabox = page.mediabox
    return float(mediabox.width), float(mediabox.height)


def create_spine_pdf_zh(spine_width_points, height_points, output_path):
    """Create a PDF for the book spine with Chinese typography."""
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
    
    # Low-Poly Background
    n_x, n_y = 4, 18
    x = np.linspace(-0.5, width + 0.5, n_x)
    y = np.linspace(-1, height + 1, n_y)
    grid_x, grid_y = np.meshgrid(x, y)
    points = np.vstack([grid_x.flatten(), grid_y.flatten()]).T
    
    jitter_strength = 0.4
    np.random.seed(42)
    
    for i, point in enumerate(points):
        points[i][0] += np.random.uniform(-jitter_strength, jitter_strength)
        points[i][1] += np.random.uniform(-jitter_strength, jitter_strength)
    
    tri = Delaunay(points)
    center = np.array([width / 2, height / 2 + 1])
    
    try:
        cmap = matplotlib.colormaps['viridis_r']
    except Exception:
        cmap = plt.get_cmap('viridis_r')
    
    for simplex in tri.simplices:
        triangle_points = points[simplex]
        centroid = np.mean(triangle_points, axis=0)
        dist = np.linalg.norm(centroid - center)
        max_dist = np.linalg.norm(np.array([0, 0]) - center)
        norm_dist = dist / (max_dist * 0.9)
        val = np.clip(norm_dist, 0, 1)
        val = 0.4 + val * 0.6
        color = cmap(val)
        color_rgb = np.array(color[:3])
        white = np.array([1.0, 1.0, 1.0])
        color_lighter = 0.6 * color_rgb + 0.4 * white
        color_final = tuple(color_lighter) + (color[3],)
        
        poly = patches.Polygon(triangle_points, closed=True,
                               facecolor=color_final, edgecolor=color_final, alpha=1.0)
        ax.add_patch(poly)
    
    zh_font = get_chinese_font()
    width_points = width * 72
    title_font_size = max(8, int(width_points * 0.28))
    author_font_size = max(6, int(width_points * 0.20))
    
    top_padding = 0.07 * height
    title_y = height - top_padding
    ax.text(width / 2, title_y, "现代分布式 AI 系统实战",
            ha='center', va='top', fontsize=title_font_size,
            fontname=zh_font, weight='bold', color='#0f172a', zorder=21,
            rotation=-90)
    
    bottom_padding = 0.05 * height
    author_y = bottom_padding
    ax.text(width / 2, author_y, "吴富恒 (Henry Wu) 著",
            ha='center', va='bottom', fontsize=author_font_size,
            fontname=zh_font, weight='bold', color='#1e293b', zorder=21,
            rotation=-90)
    
    plt.savefig(str(output_path), dpi=dpi, format='pdf',
                facecolor='white', edgecolor='none',
                bbox_inches='tight', pad_inches=0)
    plt.close()


def merge_covers_zh(front_pdf, spine_pdf, back_pdf, output_pdf):
    """Merge front, spine, back into wrap-around cover."""
    front_width, front_height = get_pdf_dimensions(front_pdf)
    spine_width, spine_height = get_pdf_dimensions(spine_pdf)
    back_width, back_height = get_pdf_dimensions(back_pdf)
    
    total_width = front_width + spine_width + back_width
    total_width_inches = total_width / 72.0
    total_height_inches = front_height / 72.0
    
    c = canvas.Canvas(str(output_pdf), pagesize=(total_width_inches * inch, total_height_inches * inch))
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        subprocess.run(['pdftoppm', '-png', '-r', '300', '-singlefile',
                       str(front_pdf), str(temp_dir / 'front')], check=True, capture_output=True)
        subprocess.run(['pdftoppm', '-png', '-r', '300', '-singlefile',
                       str(spine_pdf), str(temp_dir / 'spine')], check=True, capture_output=True)
        subprocess.run(['pdftoppm', '-png', '-r', '300', '-singlefile',
                       str(back_pdf), str(temp_dir / 'back')], check=True, capture_output=True)
        
        front_img = temp_dir / 'front.png'
        spine_img = temp_dir / 'spine.png'
        back_img = temp_dir / 'back.png'
        
        c.drawImage(str(back_img), 0, 0, width=back_width, height=back_height)
        c.drawImage(str(spine_img), back_width, 0, width=spine_width, height=back_height)
        c.drawImage(str(front_img), back_width + spine_width, 0, width=front_width, height=back_height)
        c.save()
        shutil.rmtree(temp_dir)
        return True
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise e


def main():
    script_dir = Path(__file__).parent
    front_pdf = script_dir / "cover_front_zh.pdf"
    back_pdf = script_dir / "cover_back_zh.pdf"
    output_pdf = script_dir / "cover_all_zh.pdf"
    spine_pdf = script_dir / "cover_spine_zh.pdf"
    
    if not front_pdf.exists():
        print("Creating front cover first...")
        subprocess.run([sys.executable, str(script_dir / "cover_front_zh.py")], check=True)
        
    if not back_pdf.exists():
        print("Creating back cover first...")
        subprocess.run([sys.executable, str(script_dir / "cover_back_zh.py")], check=True)
    
    _, height_points = get_pdf_dimensions(front_pdf)
    page_count = 550
    spine_width_inches = page_count * 0.002252 + 0.15
    spine_width_points = spine_width_inches * 72.0
    
    print(f"📚 正在生成中文全景展开封面（Wrap-around Cover）...")
    print(f"   前封面: {front_pdf}")
    print(f"   后封面: {back_pdf}")
    print(f"   书脊宽度: {spine_width_inches:.2f} 英寸 ({spine_width_points:.1f} points)")
    
    create_spine_pdf_zh(spine_width_points, height_points, spine_pdf)
    merge_covers_zh(front_pdf, spine_pdf, back_pdf, output_pdf)
    print(f"✅ 成功生成全书中文全景展开封面: {output_pdf}")


if __name__ == "__main__":
    main()
