#!/usr/bin/env python3
"""
Generate Chinese Front Cover for Distributed AI Systems (7x10 inch for Peanutbook/Amazon KDP).
Output: cover/7x10/cover_front_zh.pdf
"""

import random
from pathlib import Path
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


def generate_book_front_cover_zh():
    width, height = 7.0, 10.0
    dpi = 300
    
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    # 1. Generate Low-Poly Background
    n_x, n_y = 8, 10
    x = np.linspace(-1, width + 1, n_x)
    y = np.linspace(-1, height + 1, n_y)
    grid_x, grid_y = np.meshgrid(x, y)
    points = np.vstack([grid_x.flatten(), grid_y.flatten()]).T

    jitter_strength = 0.6
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
        
        poly = patches.Polygon(
            triangle_points, closed=True, 
            facecolor=color_final, edgecolor=color_final, alpha=1.0
        )
        ax.add_patch(poly)

    # 2. Add Floating Distributed AI Math & System Symbols
    symbols = [
        r'$\mathrm{AllReduce}(\mathbf{g})$',
        r'$\mathbf{W} = \mathrm{Shard}(\mathbf{W}_1, \dots, \mathbf{W}_N)$',
        r'$\mathrm{TP} \times \mathrm{PP} \times \mathrm{DP}$',
        r'$\mathrm{Attention}(Q, K, V)$',
        r'$\mathrm{KV\text{-}Cache}$',
        r'$\mathrm{ReduceScatter}(\nabla)$',
        r'$\mathrm{AllGather}(\mathbf{w})$',
        r'$\text{FLOPs} = 6ND$',
        r'$\mathrm{TopK}(g_1, \dots, g_E)$',
        r'$\mathrm{RadixTree}$',
        r'$\mathrm{PagedAttention}$',
        r'$\mathrm{RingAttention}$',
        r'$\mathrm{Zero\text{-}3\text{ }Offload}$',
        r'$\mathrm{TTFT} + N \times \mathrm{ITL}$',
        r'$\mathrm{NCCL\text{-}Direct}$',
        r'$\nabla L(\mathbf{\theta})$',
        r'$\mathbb{E}[\mathcal{L}(\theta)]$',
        r'$\frac{d\mathbf{x}}{dt} = \mathbf{f}(\mathbf{x}, t)$',
        r'$\mathrm{Softmax}(\frac{QK^\top}{\sqrt{d_k}})$',
        r'$\mathrm{FSDP2\text{ }DTensor}$',
    ]
    
    for i in range(35):
        sym = symbols[i % len(symbols)]
        sx = random.uniform(0, width)
        sy = random.uniform(0, height)
        
        if not (1 * (width / 8) < sx < 7 * (width / 8) and 0.45 * height < sy < 0.75 * height):
            size_scale = height / 10.0
            size = random.randint(int(8 * size_scale), int(15 * size_scale))
            rot = random.randint(-40, 40)
            ax.text(sx, sy, sym, fontsize=size, color='white', alpha=0.12, rotation=rot)

    # 3. Add 3D Surface Plot
    math_plot_x_center = width / 2
    math_plot_y_center = 2.9
    math_plot_width = 3.6
    math_plot_height = 2.0
    
    ax3d_left = (math_plot_x_center - math_plot_width / 2) / width
    ax3d_bottom = (math_plot_y_center - math_plot_height / 2) / height
    ax3d_width_frac = math_plot_width / width
    ax3d_height_frac = math_plot_height / height
    
    ax3d = fig.add_axes([ax3d_left, ax3d_bottom, ax3d_width_frac, ax3d_height_frac], projection='3d')
    
    x_range = np.linspace(-1, 1, 40)
    y_range = np.linspace(-1, 1, 40)
    X_math, Y_math = np.meshgrid(x_range, y_range)
    Z_math = -5 * X_math * Y_math * np.exp(-X_math**2 - Y_math**2)
    
    surf = ax3d.plot_surface(X_math, Y_math, Z_math, cmap='Blues', edgecolor='none', rstride=2, cstride=2, alpha=0.85)
    
    ax3d.set_axis_off()
    ax3d.patch.set_facecolor('none')
    ax3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax3d.view_init(elev=30, azim=-70)

    # 4. Chinese Typography
    zh_font = get_chinese_font()
    scale_y = height / 10
    
    # 英文小标
    ax.text(width / 2, 0.74 * height, "DISTRIBUTED AI SYSTEMS", 
            ha='center', fontsize=int(14 * scale_y), fontname=zh_font, 
            weight='bold', color='#1a365d', alpha=0.85, zorder=21)
    
    # 中文主标题
    title1_y = 0.67 * height
    title2_y = 0.60 * height
    
    ax.text(width / 2, title1_y, "现代分布式", 
            ha='center', fontsize=int(32 * scale_y), fontname=zh_font, 
            weight='bold', color='#0f172a', zorder=21)
    
    ax.text(width / 2, title2_y, "AI 系统实战", 
            ha='center', fontsize=int(32 * scale_y), fontname=zh_font, 
            weight='bold', color='#0f172a', zorder=21)

    # 版本标识
    edition_y = 0.54 * height
    ax.text(width / 2, edition_y, "第一版 · 全中文版 (First Edition)", 
            ha='center', fontsize=int(10.5 * scale_y), fontname=zh_font,
            style='italic', color='#334155', zorder=21)

    # 中文副标题
    subtitle_y = 0.46 * height
    subtitle_text = "大模型分布式训练、高性能推理与云原生集群架构实战指南\n涵盖 DDP / FSDP / DeepSpeed / Megatron-LM / vLLM / SGLang / SLURM"
    ax.text(width / 2, subtitle_y, subtitle_text, 
            ha='center', va='top', fontsize=int(9.5 * scale_y), 
            fontname=zh_font, color='#1e293b', zorder=21,
            linespacing=1.4)

    # 作者
    author_y = 0.14 * height
    ax.text(width / 2, author_y, "吴富恒 (Henry Wu) 著",
            ha='center', fontsize=int(10.5 * scale_y), fontname=zh_font, 
            weight='bold', color='#0f172a', zorder=21)

    # 年份
    footer_y = 0.09 * height
    ax.text(width / 2, footer_y, "2026", 
            ha='center', fontsize=int(10 * scale_y), fontname=zh_font, 
            color='#64748b', zorder=21)

    # Save
    script_dir = Path(__file__).parent
    output_path = script_dir / "cover_front_zh.pdf"
    plt.savefig(str(output_path), dpi=dpi, format='pdf')
    print(f"✅ 中文前封面已生成: {output_path}")
    
    # Also save PNG preview
    png_path = script_dir / "cover_front_zh.png"
    plt.savefig(str(png_path), dpi=dpi, format='png')
    print(f"✅ 中文前封面预览图已生成: {png_path}")
    plt.close()


if __name__ == "__main__":
    generate_book_front_cover_zh()
