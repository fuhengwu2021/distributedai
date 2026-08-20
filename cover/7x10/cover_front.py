#!/usr/bin/env python3
"""
Convert front.png (front_final_v3.png) to cover_front.pdf for Amazon KDP 7×10 inch trim size.
Preserves aspect ratio (no stretch).
Output: cover/7x10/cover_front.pdf at 300 DPI (2100×3000 px)
"""

from pathlib import Path
from PIL import Image

WIDTH_INCHES = 7.0
HEIGHT_INCHES = 10.0
DPI = 300
TARGET_WIDTH_PX = int(WIDTH_INCHES * DPI)
TARGET_HEIGHT_PX = int(HEIGHT_INCHES * DPI)


def fit_on_canvas(img: Image.Image, background: str = "white") -> Image.Image:
    src_w, src_h = img.size
    scale = min(TARGET_WIDTH_PX / src_w, TARGET_HEIGHT_PX / src_h)
    fit_w = int(round(src_w * scale))
    fit_h = int(round(src_h * scale))
    fitted = img.resize((fit_w, fit_h), Image.LANCZOS)

    canvas = Image.new("RGB", (TARGET_WIDTH_PX, TARGET_HEIGHT_PX), background)
    offset_x = (TARGET_WIDTH_PX - fit_w) // 2
    offset_y = (TARGET_HEIGHT_PX - fit_h) // 2
    canvas.paste(fitted, (offset_x, offset_y))
    return canvas


def png_to_cover_pdf(input_path: Path, output_path: Path, dpi: int = DPI) -> None:
    img = Image.open(input_path).convert("RGB")
    canvas = fit_on_canvas(img)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "PDF", resolution=float(dpi), save_all=True)


def main() -> None:
    script_dir = Path(__file__).parent
    input_path = script_dir / "front.png"
    output_path = script_dir / "cover_front.pdf"

    if not input_path.exists():
        raise FileNotFoundError(f"Front cover PNG not found: {input_path}")

    src = Image.open(input_path)
    png_to_cover_pdf(input_path, output_path)
    scale = min(TARGET_WIDTH_PX / src.width, TARGET_HEIGHT_PX / src.height)
    fit_w = int(round(src.width * scale))
    fit_h = int(round(src.height * scale))
    print(f"Front cover generated: {output_path}")
    print(f"  Source: {input_path} ({src.width}×{src.height} px)")
    print(f"  Fitted: {fit_w}×{fit_h} px (centered on white {TARGET_WIDTH_PX}×{TARGET_HEIGHT_PX} canvas)")
    print(f"  Trim: {WIDTH_INCHES}×{HEIGHT_INCHES} in @ {DPI} DPI")


if __name__ == "__main__":
    main()