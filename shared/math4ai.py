#!/usr/bin/env python3
"""
math4ai.py -- Bridge module providing configure_math_fonts and save_figure.
Re-exports from mathicon / figstyle for compatibility with chapter scripts.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

try:
    from mathicon import configure_math_fonts, save_figure, save_figure_to_script_dir
except ImportError:
    from figstyle import configure_book_style as configure_math_fonts
    from figstyle import save_figure_to_script_dir

    def save_figure(caller_file=None, *, dpi=300, bbox_inches='tight', facecolor='white',
                    edgecolor='none', pad_inches=0.03, close=True):
        if caller_file:
            script_dir = Path(caller_file).resolve().parent
            stem = Path(caller_file).stem
        else:
            script_dir = Path.cwd()
            stem = "figure"
        out_path = script_dir / f"{stem}.png"
        plt.savefig(out_path, dpi=dpi, bbox_inches=bbox_inches, facecolor=facecolor,
                    edgecolor=edgecolor, pad_inches=pad_inches)
        if close:
            plt.close()
        return str(out_path)

from figstyle import localized_figure, configure_book_style
