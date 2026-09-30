#!/usr/bin/env python3
"""
compare_figure_diff.py -- Pixel-level visual regression tool for figures.

Compares an original baseline PNG against a newly rendered PNG.
Checks:
- Canvas dimensions (width, height)
- Bounding box difference (ImageChops.difference)
- Pixel mismatch count and percentage

Usage:
    python scripts/compare_figure_diff.py <orig_img.png> <new_img.png>
"""

import sys
from pathlib import Path
from PIL import Image, ImageChops
import numpy as np


def compare_images(orig_path: str, new_path: str) -> bool:
    orig = Path(orig_path)
    new = Path(new_path)

    if not orig.is_file():
        print(f"❌ Original file not found: {orig}")
        return False
    if not new.is_file():
        print(f"❌ New file not found: {new}")
        return False

    im_orig = Image.open(orig).convert("RGBA")
    im_new = Image.open(new).convert("RGBA")

    print(f"🔍 Comparing:")
    print(f"   Original: {orig} {im_orig.size}")
    print(f"   New:      {new} {im_new.size}")

    if im_orig.size != im_new.size:
        print(f"❌ DIMENSION MISMATCH: {im_orig.size} vs {im_new.size}")
        return False

    diff = ImageChops.difference(im_orig, im_new)
    bbox = diff.getbbox()

    if bbox is None:
        print("✅ 100% IDENTICAL: Every pixel matches exactly!")
        return True

    diff_arr = np.array(diff)
    diff_pixels = np.count_nonzero(diff_arr)
    total_pixels = im_orig.size[0] * im_orig.size[1] * 4
    diff_pct = (diff_pixels / total_pixels) * 100

    print(f"⚠️ PIXEL DIFFERENCE DETECTED:")
    print(f"   Diff Bounding Box: {bbox}")
    print(f"   Diff Pixel Count:  {diff_pixels} / {total_pixels} ({diff_pct:.4f}%)")
    return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/compare_figure_diff.py <orig.png> <new.png>")
        sys.exit(1)

    matched = compare_images(sys.argv[1], sys.argv[2])
    sys.exit(0 if matched else 1)
