#!/usr/bin/env python3
"""
Generate back cover PDF using back.png as the background.

Loads cover/7x10/back.png, scales it to fill the 7×10 inch trim (300 DPI),
then draws the back-cover typography on top (same content as cover_back.py).

Input:  cover/7x10/back.png (default)
Output: cover/7x10/cover_back.pdf
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# KDP trim size: 7 × 10 inches at print resolution
WIDTH_INCHES = 7.0
HEIGHT_INCHES = 10.0
DPI = 300
TARGET_WIDTH_PX = int(WIDTH_INCHES * DPI)
TARGET_HEIGHT_PX = int(HEIGHT_INCHES * DPI)

# Background image opacity (0 = white only, 1 = full-strength back.png)
BACKGROUND_ALPHA = 0.55
# Semi-opaque panels behind body text
TEXT_BOX_ALPHA = 0.82


def cover_on_canvas(img: Image.Image) -> Image.Image:
    """Scale image to cover trim size (fill canvas; center-crop if needed)."""
    src_w, src_h = img.size
    scale = max(TARGET_WIDTH_PX / src_w, TARGET_HEIGHT_PX / src_h)
    fit_w = int(round(src_w * scale))
    fit_h = int(round(src_h * scale))
    fitted = img.resize((fit_w, fit_h), Image.LANCZOS)

    canvas = Image.new("RGB", (TARGET_WIDTH_PX, TARGET_HEIGHT_PX))
    offset_x = (TARGET_WIDTH_PX - fit_w) // 2
    offset_y = (TARGET_HEIGHT_PX - fit_h) // 2
    canvas.paste(fitted, (offset_x, offset_y))
    return canvas


def _draw_back_cover_content(ax, width: float, height: float) -> None:
    """Typography and boxes for the back cover (overlay on background image)."""
    random.seed(42)

    symbols = [
        r"$\sum_{i=1}^{N} \nabla_i$",
        r"$\mathbf{W} = \mathrm{Shard}(\mathbf{W}_1, \ldots, \mathbf{W}_N)$",
        r"$\mathrm{AllReduce}(\mathbf{g})$",
        r"$\mathrm{KV}$",
        r"$\mathrm{FSDP}$",
        r"$\mathrm{DDP}$",
        r"$\mathrm{TP} \times \mathrm{PP} \times \mathrm{DP}$",
        r"$\mathrm{Throughput} = \frac{N}{T}$",
        r"$\mathrm{Latency} = f(b, s)$",
        r"$\mathrm{GPU} \times N$",
    ]

    for i in range(20):
        sym = symbols[i % len(symbols)]
        sx = random.uniform(0, width)
        sy = random.uniform(0, height)
        if not (
            2 * (width / 8) < sx < 6 * (width / 8)
            and 0.25 * height < sy < 0.85 * height
        ):
            size_scale = height / 10.0
            size = random.randint(int(7 * size_scale), int(14 * size_scale))
            rot = random.randint(-45, 45)
            ax.text(
                sx,
                sy,
                sym,
                fontsize=size,
                color="white",
                alpha=0.08,
                rotation=rot,
                zorder=10,
            )

    scale_y = height / 10
    corner_radius = 0.15

    ax.text(
        width / 2,
        0.82 * height,
        "DISTRIBUTED AI SYSTEMS",
        ha="center",
        fontsize=int(24 * scale_y),
        fontname="DejaVu Sans",
        weight="bold",
        color="black",
        zorder=21,
    )
    ax.text(
        width / 2,
        0.78 * height,
        "Build scalable training, inference and serving systems",
        ha="center",
        fontsize=int(11 * scale_y),
        fontname="DejaVu Sans",
        style="italic",
        color="black",
        zorder=21,
    )

    desc_box_padding = 0.165
    desc_box_width = width * 0.70
    desc_box_x = (width - desc_box_width) / 2
    desc_text = (
        "With large AI models now reaching billions or even trillions of parameters,\n"
        "distributed systems have become essential for training and serving AI.\n"
        "This comprehensive guide bridges the gap between theory and practice\n"
        "with hands-on code examples demonstrating production-grade techniques\n"
        "used in real-world systems, from distributed training through inference\n"
        "to production serving."
    )
    desc_box_height = (
        (desc_text.count("\n") + 1) * int(9 * scale_y) * 1.4 / 72
        + desc_box_padding * 2
    )
    desc_box_y = 0.75 * height - desc_box_height
    ax.add_patch(
        patches.FancyBboxPatch(
            (desc_box_x, desc_box_y),
            desc_box_width,
            desc_box_height,
            boxstyle=f"round,pad=0.02,rounding_size={corner_radius}",
            linewidth=1,
            edgecolor="grey",
            facecolor="white",
            alpha=TEXT_BOX_ALPHA,
            zorder=20,
        )
    )
    ax.text(
        desc_box_x + desc_box_padding,
        desc_box_y + desc_box_height - desc_box_padding,
        desc_text,
        ha="left",
        va="top",
        fontsize=int(9 * scale_y),
        fontname="DejaVu Sans",
        color="black",
        zorder=21,
        linespacing=1.4,
    )

    combined_box_padding = 0.165
    combined_box_width = width * 0.70
    combined_box_x = (width - combined_box_width) / 2
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
    line_height = int(9 * scale_y) * 1.4 / 72
    features_lines = features_content.count("\n") + 1
    audience_lines = audience_content.count("\n") + 1
    # Between the old 13-line estimate (too short) and the full layout sum (too tall)
    combined_box_height = combined_box_padding * 2 + (
        (features_lines + audience_lines + 6) * line_height
    )
    combined_box_y = 0.50 * height - combined_box_height
    ax.add_patch(
        patches.FancyBboxPatch(
            (combined_box_x, combined_box_y),
            combined_box_width,
            combined_box_height,
            boxstyle=f"round,pad=0.02,rounding_size={corner_radius}",
            linewidth=1,
            edgecolor="grey",
            facecolor="white",
            alpha=TEXT_BOX_ALPHA,
            zorder=20,
        )
    )

    combined_text_x = combined_box_x + combined_box_padding
    combined_text_y = combined_box_y + combined_box_height - combined_box_padding
    y_pos = combined_text_y

    ax.text(
        combined_text_x,
        y_pos,
        features_header,
        ha="left",
        va="top",
        fontsize=int(9 * scale_y),
        fontname="DejaVu Sans",
        color="black",
        zorder=21,
        weight="bold",
    )
    y_pos -= line_height * 2
    ax.text(
        combined_text_x,
        y_pos,
        features_content,
        ha="left",
        va="top",
        fontsize=int(9 * scale_y),
        fontname="DejaVu Sans",
        color="black",
        zorder=21,
        linespacing=1.4,
    )
    y_pos -= (features_content.count("\n") + 1) * line_height
    y_pos -= line_height * 2

    ax.text(
        combined_text_x,
        y_pos,
        audience_header,
        ha="left",
        va="top",
        fontsize=int(9 * scale_y),
        fontname="DejaVu Sans",
        color="black",
        zorder=21,
        weight="bold",
    )
    y_pos -= line_height * 2
    ax.text(
        combined_text_x,
        y_pos,
        audience_content,
        ha="left",
        va="top",
        fontsize=int(9 * scale_y),
        fontname="DejaVu Sans",
        color="black",
        zorder=21,
        linespacing=1.4,
    )


def generate_book_back_cover(
    background_path: Path,
    output_path: Path,
    dpi: int = DPI,
    background_alpha: float = BACKGROUND_ALPHA,
) -> None:
    """Compose back.png background + typography; write PDF."""
    bg_img = cover_on_canvas(Image.open(background_path).convert("RGB"))
    bg_array = np.asarray(bg_img) / 255.0

    width, height = WIDTH_INCHES, HEIGHT_INCHES
    fig = plt.figure(figsize=(width, height), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    # White base so faded background keeps text readable
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            width,
            height,
            facecolor="white",
            edgecolor="none",
            zorder=0,
        )
    )
    ax.imshow(
        bg_array,
        extent=[0, width, 0, height],
        origin="lower",
        aspect="auto",
        alpha=background_alpha,
        zorder=1,
        interpolation="lanczos",
    )
    _draw_back_cover_content(ax, width, height)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, format="pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 7×10 back cover PDF from back.png.")
    parser.add_argument(
        "--background-alpha",
        type=float,
        default=BACKGROUND_ALPHA,
        help=f"Opacity of back.png over white (default: {BACKGROUND_ALPHA}). Lower = clearer text.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    background_path = script_dir / "back.png"
    output_path = script_dir / "cover_back.pdf"

    if not background_path.exists():
        raise FileNotFoundError(f"Back cover background PNG not found: {background_path}")

    src = Image.open(background_path)
    generate_book_back_cover(
        background_path,
        output_path,
        background_alpha=args.background_alpha,
    )
    scale = max(TARGET_WIDTH_PX / src.width, TARGET_HEIGHT_PX / src.height)
    fit_w = int(round(src.width * scale))
    fit_h = int(round(src.height * scale))
    print(f"Back cover generated: {output_path}")
    print(f"  Background: {background_path} ({src.width}×{src.height} px)")
    print(
        f"  Scaled: {fit_w}×{fit_h} px (cover fill on "
        f"{TARGET_WIDTH_PX}×{TARGET_HEIGHT_PX} canvas)"
    )
    print(f"  Trim: {WIDTH_INCHES}×{HEIGHT_INCHES} in @ {DPI} DPI")
    print(f"  Background alpha: {args.background_alpha}")


if __name__ == "__main__":
    main()
