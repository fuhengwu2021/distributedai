#!/usr/bin/env python3
"""
Set DPI for PNG images.

Usage:
    python3 set_image_dpi.py <image_path> [dpi]
    python3 set_image_dpi.py <image_path>          # Default: 100 DPI

Examples:
    python3 set_image_dpi.py img/decoder_only.png
    python3 set_image_dpi.py img/decoder_only.png 100
    python3 set_image_dpi.py img/decoder_only.png 300
"""

import sys
import os
from PIL import Image


def set_image_dpi(image_path, dpi=100):
    """
    Set DPI for a PNG image.
    
    Args:
        image_path: Path to the PNG image file
        dpi: DPI value to set (default: 100)
    
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return False
    
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Save with specified DPI
        img.save(image_path, dpi=(dpi, dpi))
        
        # Verify
        img2 = Image.open(image_path)
        actual_dpi = img2.info.get('dpi', 'not set')
        
        print(f"✅ DPI set to {dpi} for {image_path}")
        print(f"   Verified DPI: {actual_dpi}")
        
        return True
    except Exception as e:
        print(f"Error: Failed to set DPI: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    image_path = sys.argv[1]
    dpi = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    success = set_image_dpi(image_path, dpi)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
