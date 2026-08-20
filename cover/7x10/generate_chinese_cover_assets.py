#!/usr/bin/env python3
"""
Generate complete bitmap assets (PNG, JPG, Amazon Cover thumbnails) for Chinese 7x10 Cover.
Outputs:
- cover/7x10/front_final_zh.png (2100x3000 px @ 300 DPI)
- cover/7x10/front_zh.png (symlink to front_final_zh.png)
- cover/7x10/front_zh.jpg
- cover/7x10/back_zh.png
- cover/7x10/amazon_cover_zh.jpg (1216x1500)
- cover/7x10/amazon_cover_zh_260w.jpg (260x321)
- cover/7x10/amazon_cover_zh_90w.jpg (90x111)
"""

from pathlib import Path
import subprocess
import sys
from PIL import Image

SCRIPT_DIR = Path(__file__).parent


def generate_bitmaps():
    print("🎨 正在生成中文封面高清位图资源...")
    
    # 1. Ensure front and back PDF/PNG generators run
    subprocess.run([sys.executable, str(SCRIPT_DIR / "cover_front_zh.py")], check=True)
    subprocess.run([sys.executable, str(SCRIPT_DIR / "cover_back_zh.py")], check=True)
    
    front_png = SCRIPT_DIR / "cover_front_zh.png"
    back_png = SCRIPT_DIR / "cover_back_zh.png"
    
    if not front_png.exists():
        raise FileNotFoundError(f"Missing {front_png}")
    if not back_png.exists():
        raise FileNotFoundError(f"Missing {back_png}")
    
    # 2. Generate front_final_zh.png and front_zh.png
    front_final_zh = SCRIPT_DIR / "front_final_zh.png"
    img_front = Image.open(front_png).convert("RGB")
    img_front.save(front_final_zh, "PNG", quality=95)
    print(f"✅ 生成 {front_final_zh} ({img_front.width}x{img_front.height})")
    
    # Symlink front_zh.png -> front_final_zh.png
    front_zh_link = SCRIPT_DIR / "front_zh.png"
    if front_zh_link.exists() or front_zh_link.is_symlink():
        front_zh_link.unlink()
    front_zh_link.symlink_to("front_final_zh.png")
    print(f"✅ 创建软链接 {front_zh_link} -> front_final_zh.png")
    
    # 3. Generate front_zh.jpg
    front_zh_jpg = SCRIPT_DIR / "front_zh.jpg"
    img_front.save(front_zh_jpg, "JPEG", quality=92)
    print(f"✅ 生成 {front_zh_jpg}")
    
    # 4. Generate back_zh.png
    back_zh_dest = SCRIPT_DIR / "back_zh.png"
    img_back = Image.open(back_png).convert("RGBA")
    img_back.save(back_zh_dest, "PNG", quality=95)
    print(f"✅ 生成 {back_zh_dest} ({img_back.width}x{img_back.height})")
    
    # 5. Generate Amazon Cover Thumbnails
    # amazon_cover_zh.jpg (1216x1500)
    amz_full = img_front.resize((1216, 1500), Image.LANCZOS)
    amz_path = SCRIPT_DIR / "amazon_cover_zh.jpg"
    amz_full.save(amz_path, "JPEG", quality=90)
    print(f"✅ 生成 Amazon 封面标准图: {amz_path} (1216x1500)")
    
    # amazon_cover_zh_260w.jpg (260x321)
    amz_260 = img_front.resize((260, 321), Image.LANCZOS)
    amz_260_path = SCRIPT_DIR / "amazon_cover_zh_260w.jpg"
    amz_260.save(amz_260_path, "JPEG", quality=88)
    print(f"✅ 生成 Amazon 封面中缩略图: {amz_260_path} (260x321)")
    
    # amazon_cover_zh_90w.jpg (90x111)
    amz_90 = img_front.resize((90, 111), Image.LANCZOS)
    amz_90_path = SCRIPT_DIR / "amazon_cover_zh_90w.jpg"
    amz_90.save(amz_90_path, "JPEG", quality=85)
    print(f"✅ 生成 Amazon 封面小缩略图: {amz_90_path} (90x111)")
    
    print("\n🎉 全部中文封面位图资产生成完毕！")


if __name__ == "__main__":
    generate_bitmaps()
