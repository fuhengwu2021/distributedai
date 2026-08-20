#!/usr/bin/env python3
"""
Generate high-fidelity Chinese Front Cover based on front_final_v3.png (Packt Style 3D Spheres).
Preserves the exact 3D sphere network artwork, author photo, orange dividing line, and layout.
Outputs:
- cover/7x10/front_final_zh.png (2100x3000 px @ 300 DPI)
- cover/7x10/front_zh.png (symlink to front_final_zh.png)
- cover/7x10/cover_front_zh.pdf (300 DPI PDF)
- cover/7x10/amazon_cover_zh.jpg (1216x1500 px)
- cover/7x10/amazon_cover_zh_260w.jpg (260x321 px)
- cover/7x10/amazon_cover_zh_90w.jpg (90x111 px)
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).parent
BASE_IMG_PATH = SCRIPT_DIR / "front_final_v3.png"
OUTPUT_PNG_PATH = SCRIPT_DIR / "front_final_zh.png"

# Target print dimensions for 7x10 @ 300 DPI
TARGET_W = 2100
TARGET_H = 3000


def get_font(font_path_candidates, size):
    for p in font_path_candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def build_chinese_front_cover():
    if not BASE_IMG_PATH.exists():
        raise FileNotFoundError(f"Base image not found: {BASE_IMG_PATH}")
    
    # Load base image (1127x1396)
    base_img = Image.open(BASE_IMG_PATH).convert("RGBA")
    w, h = base_img.size
    
    # Clean top title & subtitle area (0 <= y <= 475, completely removing old title & subtitle before spheres start at y=485)
    draw = ImageDraw.Draw(base_img)
    bg_color = (254, 254, 254, 255)
    
    # Clear top text area completely
    draw.rectangle([0, 0, w, 475], fill=bg_color)
    
    # Clear bottom-left foreword area (x: 40 to 480, y: 1100 to 1250)
    draw.rectangle([40, 1100, 480, 1250], fill=bg_color)
    
    # Clear bottom-left author area (x: 40 to 500, y: 1260 to 1380)
    draw.rectangle([40, 1260, 500, 1380], fill=bg_color)
    
    # Fonts
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
    regular_fonts = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]

    font_title_zh = get_font(bold_fonts, 96)
    font_subtitle_zh = get_font(bold_fonts, 27)
    font_subtitle_en = get_font(medium_fonts, 20)
    font_foreword_label = get_font(bold_fonts, 24)
    font_foreword_name = get_font(bold_fonts, 22)
    font_foreword_title = get_font(medium_fonts, 20)
    font_author_zh = get_font(bold_fonts, 46)

    # 1. Draw Top Title & Subtitle (Left aligned at x = 65)
    left_margin = 65
    
    # Title Line 1: 分布式
    # Title Line 2: AI 系统
    draw.text((left_margin, 70), "分布式", font=font_title_zh, fill="#000000")
    draw.text((left_margin, 185), "AI 系统", font=font_title_zh, fill="#000000")
    
    # Subtitle
    draw.text((left_margin, 325), "大模型分布式训练、高性能推理与服务化实战指南", font=font_subtitle_zh, fill="#111827")
    draw.text((left_margin, 370), "A practical guide to building scalable training, inference, and serving systems for production AI", font=font_subtitle_en, fill="#4b5563")

    # 2. Draw Bottom-Left Foreword (above orange line)
    draw.text((left_margin, 1120), "推荐序", font=font_foreword_label, fill="#f05a28")
    draw.text((left_margin, 1158), "赵刚 (Gang Zhao)", font=font_foreword_name, fill="#000000")
    draw.text((left_margin, 1192), "英伟达主任工程师 (Staff Engineer, Nvidia)", font=font_foreword_title, fill="#374151")

    # 3. Draw Bottom-Left Author Name (below orange line)
    draw.text((left_margin, 1285), "吴富恒 (Fuheng Wu)", font=font_author_zh, fill="#000000")

    # Convert back to RGB
    final_cover_rgb = base_img.convert("RGB")
    
    # Scale to high-res 7x10 @ 300 DPI (2100x3000)
    # Fit onto 2100x3000 canvas with white padding
    scale = min(TARGET_W / w, TARGET_H / h)
    fit_w = int(round(w * scale))
    fit_h = int(round(h * scale))
    fitted = final_cover_rgb.resize((fit_w, fit_h), Image.LANCZOS)
    
    canvas_300dpi = Image.new("RGB", (TARGET_W, TARGET_H), "white")
    offset_x = (TARGET_W - fit_w) // 2
    offset_y = (TARGET_H - fit_h) // 2
    canvas_300dpi.paste(fitted, (offset_x, offset_y))
    
    # Save high-res PNG
    canvas_300dpi.save(OUTPUT_PNG_PATH, "PNG", quality=95)
    print(f"✅ 生成与 front_final_v3.png 同款 3D 球形网络的中文封面: {OUTPUT_PNG_PATH} (2100x3000)")
    
    # Update symlink front_zh.png
    front_zh_link = SCRIPT_DIR / "front_zh.png"
    if front_zh_link.exists() or front_zh_link.is_symlink():
        front_zh_link.unlink()
    front_zh_link.symlink_to("front_final_zh.png")
    
    # Save PDF
    pdf_path = SCRIPT_DIR / "cover_front_zh.pdf"
    canvas_300dpi.save(pdf_path, "PDF", resolution=300.0, save_all=True)
    print(f"✅ 生成印刷级 PDF: {pdf_path}")
    
    # Save Amazon Cover Thumbnails
    amz_full = canvas_300dpi.resize((1216, 1500), Image.LANCZOS)
    amz_path = SCRIPT_DIR / "amazon_cover_zh.jpg"
    amz_full.save(amz_path, "JPEG", quality=90)
    
    amz_260 = canvas_300dpi.resize((260, 321), Image.LANCZOS)
    amz_260.save(SCRIPT_DIR / "amazon_cover_zh_260w.jpg", "JPEG", quality=88)
    
    amz_90 = canvas_300dpi.resize((90, 111), Image.LANCZOS)
    amz_90.save(SCRIPT_DIR / "amazon_cover_zh_90w.jpg", "JPEG", quality=85)
    print(f"✅ 生成 Amazon 封面各规格缩略图")


if __name__ == "__main__":
    build_chinese_front_cover()
