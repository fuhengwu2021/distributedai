#!/usr/bin/env python3
"""
Generate Chinese Back Cover for Distributed AI Systems (7x10 inch for Peanutbook/Amazon KDP).
Output: cover/7x10/cover_back_zh.pdf
"""

import argparse
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


def _draw_low_poly_background(ax, width, height):
    """Low-poly viridis background (matches front cover style)."""
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
        cmap = matplotlib.colormaps["viridis_r"]
    except Exception:
        cmap = plt.get_cmap("viridis_r")

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
        color_lighter = 0.6 * color_rgb + 0.4 * np.array([1.0, 1.0, 1.0])
        color_final = tuple(color_lighter) + (color[3],)
        ax.add_patch(
            patches.Polygon(
                triangle_points,
                closed=True,
                facecolor=color_final,
                edgecolor=color_final,
                alpha=1.0,
            )
        )


def generate_book_back_cover_zh(background="white"):
    width, height = 7.0, 10.0
    dpi = 300
    
    use_poly = background == "poly"
    fig = plt.figure(figsize=(width, height), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="white")
    ax.set_axis_off()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    if use_poly:
        _draw_low_poly_background(ax, width, height)
    else:
        ax.add_patch(
            patches.Rectangle((0, 0), width, height, facecolor="white", edgecolor="none", zorder=0)
        )

    # 浮动数学/系统符号
    symbol_color = "white" if use_poly else "#94a3b8"
    symbols = [
        r'$\sum_{i=1}^{N} \nabla_i$',
        r'$\mathbf{W} = \mathrm{Shard}(\mathbf{W}_1, \dots, \mathbf{W}_N)$',
        r'$\mathrm{AllReduce}(\mathbf{g})$',
        r'$\mathrm{KV\text{-}Cache}$',
        r'$\mathrm{FSDP2}$',
        r'$\mathrm{DDP}$',
        r'$\mathrm{TP} \times \mathrm{PP} \times \mathrm{DP}$',
        r'$\mathrm{Throughput} = \frac{N}{T}$',
        r'$\mathrm{RadixTree}$',
        r'$\mathrm{GPU} \times N$',
    ]
    
    for i in range(20):
        sym = symbols[i % len(symbols)]
        sx = random.uniform(0, width)
        sy = random.uniform(0, height)
        if not (1.5 * (width / 8) < sx < 6.5 * (width / 8) and 0.20 * height < sy < 0.88 * height):
            size_scale = height / 10.0
            size = random.randint(int(7 * size_scale), int(13 * size_scale))
            rot = random.randint(-40, 40)
            ax.text(sx, sy, sym, fontsize=size, color=symbol_color, alpha=0.12, rotation=rot)

    zh_font = get_chinese_font()
    scale_y = height / 10
    
    # 1. 顶部大标题
    title_y = 0.84 * height
    ax.text(width / 2, title_y, "现代分布式 AI 系统实战", 
            ha='center', fontsize=int(22 * scale_y), fontname=zh_font, 
            weight='bold', color='#0f172a', zorder=21)
    
    # 顶部副标题
    subtitle_y = 0.80 * height
    ax.text(width / 2, subtitle_y, "DISTRIBUTED AI SYSTEMS · PRODUCTION REFERENCE", 
            ha='center', fontsize=int(9.5 * scale_y), fontname=zh_font, 
            weight='bold', color='#475569', zorder=21)
    
    corner_radius = 0.12
    box_width = width * 0.76
    box_x = (width - box_width) / 2
    box_padding = 0.16

    # 2. 核心简介卡片 (Description Box)
    desc_box_height = 1.35
    desc_box_y = 0.63 * height
    
    desc_box = patches.FancyBboxPatch(
        (box_x, desc_box_y), box_width, desc_box_height,
        boxstyle=f"round,pad=0.02,rounding_size={corner_radius}", 
        linewidth=1, edgecolor='#cbd5e1',
        facecolor='#f8fafc' if not use_poly else 'white', alpha=0.92, zorder=20
    )
    ax.add_patch(desc_box)
    
    desc_text = (
        "随着大语言模型迈向千亿与万亿参数时代，单芯片物理显存与算力早已触及极限。\n"
        "分布式技术已成为大模型训练、推理加速与工业级部署的核心生命线。\n"
        "本书系统打通底层原理与工程实战，深入剖析 DDP、FSDP、DeepSpeed ZeRO、\n"
        "Megatron 3D 并行、vLLM、SGLang、SLURM 集群调度及云原生推理服务栈。"
    )
    ax.text(box_x + box_padding, desc_box_y + desc_box_height - box_padding, desc_text, 
            ha='left', va='top', fontsize=int(8.5 * scale_y), 
            fontname=zh_font, color='#1e293b', zorder=21,
            linespacing=1.45)
    
    # 3. 核心特色与目标读者卡片 (Features & Audience Box)
    feat_box_height = 4.25
    feat_box_y = 0.17 * height
    
    feat_box = patches.FancyBboxPatch(
        (box_x, feat_box_y), box_width, feat_box_height,
        boxstyle=f"round,pad=0.02,rounding_size={corner_radius}", 
        linewidth=1, edgecolor='#cbd5e1',
        facecolor='#f8fafc' if not use_poly else 'white', alpha=0.92, zorder=20
    )
    ax.add_patch(feat_box)
    
    # 特色内容
    cur_y = feat_box_y + feat_box_height - box_padding
    
    ax.text(box_x + box_padding, cur_y, "核心技术特色 / Key Features", 
            ha='left', va='top', fontsize=int(9.5 * scale_y), 
            fontname=zh_font, weight='bold', color='#0369a1', zorder=21)
    cur_y -= 0.32
    
    features_text = (
        "• 深度解析 DDP、FSDP、DeepSpeed ZeRO-1/2/3 与 Megatron 混合并行\n"
        "• 构建基于 vLLM PagedAttention 与 SGLang RadixAttention 的极致低延迟推理\n"
        "• 剖析 GPU 物理硬件、NVLink/NVSwitch 拓扑与 InfiniBand/RoCE 集群网络\n"
        "• 基于 SLURM 与 Kubernetes (llm-d) 编排千万级高并发在线服务栈\n"
        "• 配套经过万卡集群真实硬件验证的端到端生产级可运行代码"
    )
    ax.text(box_x + box_padding, cur_y, features_text, 
            ha='left', va='top', fontsize=int(8.2 * scale_y), 
            fontname=zh_font, color='#1e293b', zorder=21,
            linespacing=1.45)
    cur_y -= 1.60
    
    # 读者群体
    ax.text(box_x + box_padding, cur_y, "目标读者群体 / Target Audience", 
            ha='left', va='top', fontsize=int(9.5 * scale_y), 
            fontname=zh_font, weight='bold', color='#0369a1', zorder=21)
    cur_y -= 0.32
    
    audience_text = (
        "本书专为大模型算法工程师（MLE）、AI 基础设施（AI Infra）研发专家、\n"
        "分布式系统架构师、HPC 超算中心运维工程师以及云计算技术专家量身打造。\n"
        "帮助读者系统跨越单机算法模型向大规模物理集群算力落地的工程鸿沟。"
    )
    ax.text(box_x + box_padding, cur_y, audience_text, 
            ha='left', va='top', fontsize=int(8.2 * scale_y), 
            fontname=zh_font, color='#1e293b', zorder=21,
            linespacing=1.45)

    # 底部出版社/元数据
    footer_y = 0.08 * height
    ax.text(width / 2, footer_y, "Peanutbook Open Engineering Series · 2026", 
            ha='center', fontsize=int(8.5 * scale_y), fontname=zh_font, 
            color='#64748b', zorder=21)

    # 保存
    script_dir = Path(__file__).parent
    output_path = script_dir / "cover_back_zh.pdf"
    plt.savefig(str(output_path), dpi=dpi, format="pdf", facecolor="white")
    print(f"✅ 中文后封面已生成: {output_path}")
    
    png_path = script_dir / "cover_back_zh.png"
    plt.savefig(str(png_path), dpi=dpi, format='png')
    print(f"✅ 中文后封面预览图已生成: {png_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate 7×10 Chinese back cover PDF.")
    parser.add_argument(
        "--background",
        choices=("white", "poly"),
        default="white",
        help="Background style: pure white (default) or low-poly gradient (poly).",
    )
    args = parser.parse_args()
    generate_book_back_cover_zh(background=args.background)


if __name__ == "__main__":
    main()
