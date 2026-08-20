#!/usr/bin/env python3
"""
Generate Chinese Front Cover for Distributed AI Systems (7x10 inch for Peanutbook/Amazon KDP).
Packt-style with 3D Sphere Network artwork, author photo, and Chinese typography.
Output:
- cover/7x10/cover_front_zh.pdf
- cover/7x10/cover_front_zh.png
- cover/7x10/front_final_zh.png
- cover/7x10/front_zh.png -> front_final_zh.png
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).parent
BASE_IMG_PATH = SCRIPT_DIR / "front_final_v3.png"
OUTPUT_PNG_PATH = SCRIPT_DIR / "front_final_zh.png"
OUTPUT_PDF_PATH = SCRIPT_DIR / "cover_front_zh.pdf"
COVER_FRONT_ZH_PNG = SCRIPT_DIR / "cover_front_zh.png"

# KDP / Peanutbook 7x10 @ 300 DPI canvas
TARGET_W = 2100
TARGET_H = 3000


def get_font(font_path_candidates, size):
    for p in font_path_candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def generate_book_front_cover_zh():
    if not BASE_IMG_PATH.exists():
        raise FileNotFoundError(f"Base image not found: {BASE_IMG_PATH}")
    
    # Load base Packt image (1127x1396)
    base_img = Image.open(BASE_IMG_PATH).convert("RGBA")
    w, h = base_img.size
    
    draw = ImageDraw.Draw(base_img)
    bg_color = (254, 254, 254, 255)
    
    # 1. Clean top text area (0 <= y <= 475, safely above 3D sphere starting at y=485)
    draw.rectangle([0, 0, w, 475], fill=bg_color)
    
    # 2. Clean bottom-left foreword area (x: 40 to 480, y: 1100 to 1250)
    draw.rectangle([40, 1100, 480, 1250], fill=bg_color)
    
    # 3. Clean bottom-left author area (x: 40 to 500, y: 1260 to 1380)
    draw.rectangle([40, 1260, 500, 1380], fill=bg_color)
    
    bold_fonts = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    medium_fonts = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]

    font_title_zh = get_font(bold_fonts, 130)
    font_subtitle_zh = get_font(bold_fonts, 28)
    font_subtitle_en = get_font(medium_fonts, 21)
    font_foreword_label = get_font(bold_fonts, 24)
    font_foreword_name = get_font(bold_fonts, 22)
    font_foreword_title = get_font(medium_fonts, 20)
    font_author_zh = get_font(bold_fonts, 46)

    left_margin = 65
    
    # Title (Large Single Line): 分布式 AI 系统
    draw.text((left_margin, 95), "分布式 AI 系统", font=font_title_zh, fill="#000000")
    
    # Subtitle
    draw.text((left_margin, 255), "大模型分布式训练、高性能推理与服务化实战指南", font=font_subtitle_zh, fill="#111827")
    draw.text((left_margin, 300), "A practical guide to building scalable training, inference, and serving systems for production AI", font=font_subtitle_en, fill="#4b5563")

    # Foreword (above orange line)
    draw.text((left_margin, 1120), "推荐序", font=font_foreword_label, fill="#f05a28")
    draw.text((left_margin, 1158), "赵刚 (Gang Zhao)", font=font_foreword_name, fill="#000000")
    draw.text((left_margin, 1192), "英伟达高级工程师 (Staff Engineer, Nvidia)", font=font_foreword_title, fill="#374151")

    # Author Name (below orange line)
    draw.text((left_margin, 1285), "巫富珩 (Fuheng Wu)", font=font_author_zh, fill="#000000")

    final_cover_rgb = base_img.convert("RGB")
    
    # Fit onto 2100x3000 canvas with white letterboxing
    scale = min(TARGET_W / w, TARGET_H / h)
    fit_w = int(round(w * scale))
    fit_h = int(round(h * scale))
    fitted = final_cover_rgb.resize((fit_w, fit_h), Image.LANCZOS)
    
    canvas_300dpi = Image.new("RGB", (TARGET_W, TARGET_H), "white")
    offset_x = (TARGET_W - fit_w) // 2
    offset_y = (TARGET_H - fit_h) // 2
    canvas_300dpi.paste(fitted, (offset_x, offset_y))
    
    # 1. Save front_final_zh.png & cover_front_zh.png
    canvas_300dpi.save(OUTPUT_PNG_PATH, "PNG", quality=95)
    canvas_300dpi.save(COVER_FRONT_ZH_PNG, "PNG", quality=95)
    print(f"✅ 生成高清位图: {OUTPUT_PNG_PATH}")
    
    # 2. Update symlink front_zh.png -> front_final_zh.png
    front_zh_link = SCRIPT_DIR / "front_zh.png"
    if front_zh_link.exists() or front_zh_link.is_symlink():
        front_zh_link.unlink()
    front_zh_link.symlink_to("front_final_zh.png")
    
    # 3. Save PDF cover_front_zh.pdf
    canvas_300dpi.save(OUTPUT_PDF_PATH, "PDF", resolution=300.0, save_all=True)
    print(f"✅ 生成印刷级 PDF: {OUTPUT_PDF_PATH}")
    
    # 4. Save Amazon thumbnails
    amz_full = canvas_300dpi.resize((1216, 1500), Image.LANCZOS)
    amz_full.save(SCRIPT_DIR / "amazon_cover_zh.jpg", "JPEG", quality=90)
    amz_260 = canvas_300dpi.resize((260, 321), Image.LANCZOS)
    amz_260.save(SCRIPT_DIR / "amazon_cover_zh_260w.jpg", "JPEG", quality=88)
    amz_90 = canvas_300dpi.resize((90, 111), Image.LANCZOS)
    amz_90.save(SCRIPT_DIR / "amazon_cover_zh_90w.jpg", "JPEG", quality=85)


if __name__ == "__main__":
    generate_book_front_cover_zh()
