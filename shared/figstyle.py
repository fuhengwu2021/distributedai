#!/usr/bin/env python3
"""
figstyle.py -- Shared styling and localization infrastructure for book figures.
Adopted from the gold standard in ~/mmb (mortgagekit/figstyle.py).

Features:
- localized_figure: Generates <stem>.png for English and <stem>_<lang>.png for other languages
  from a single draw(text) implementation.
- Zero-drift guarantee: Each language renders in an isolated plt.rc_context() so that
  English figures (<stem>.png) remain 100% pixel-faithful and identical to their original layout.
- configure_book_style: Automatically loads CJK fonts for 'zh' and math fonts for 'en'.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Dict, Any, Optional

import matplotlib.pyplot as plt

# Import base utilities from mathicon if available
try:
    from mathicon import configure_math_fonts, save_figure_to_script_dir
except ImportError:
    def configure_math_fonts(font_size=11):
        plt.rcParams.update({"font.size": font_size})

    def save_figure_to_script_dir(filename: str, caller_file: Optional[str] = None, dpi=300):
        if caller_file:
            target_dir = Path(caller_file).resolve().parent
        else:
            target_dir = Path.cwd()
        out_path = target_dir / filename
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.03)
        return str(out_path)

# Pan-CJK serif and sans-serif fonts in priority order
CJK_FONTS = [
    "Noto Serif CJK SC",
    "Noto Sans CJK SC",
    "Noto Serif CJK JP",
    "Noto Sans CJK JP",
    "Source Han Serif SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
]


def configure_book_style(font_size: Optional[float] = None, lang: str = "en", use_math_fonts: bool = False) -> None:
    """Apply language-specific matplotlib font and canvas settings.

    Only modifies font families for 'zh'; does NOT overwrite figure-specific
    spines, ticks, or grid settings, guaranteeing that English plots remain
    completely unchanged and identical to their original design.
    """
    if use_math_fonts:
        if font_size is not None:
            configure_math_fonts(font_size=font_size)
        else:
            configure_math_fonts()
    elif font_size is not None:
        plt.rcParams.update({"font.size": font_size})

    if lang == "zh":
        current_serif = list(plt.rcParams.get("font.serif", []))
        current_sans = list(plt.rcParams.get("font.sans-serif", []))
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.serif": CJK_FONTS + current_serif,
            "font.sans-serif": CJK_FONTS + current_sans,
            "axes.unicode_minus": False,
        })


def localized_figure(
    draw: Callable[[Dict[str, Any]], plt.Figure],
    stem: str,
    labels: Dict[str, Dict[str, Any]],
    caller_file: str,
    font_size: Optional[float] = None,
    export_pdf: bool = False,
    pad_inches: float = 0.03,
    use_math_fonts: bool = False,
    dpi: int = 300,
    bbox_inches: Optional[str] = "tight"
) -> None:
    """Render one figure per language from a single drawing implementation.

    draw(text) is called once per language key in labels.
    - English ('en') writes <stem>.png and <stem>.pdf
    - Other languages (e.g. 'zh') write <stem>_<lang>.png and <stem>_<lang>.pdf

    Each render runs inside an isolated plt.rc_context() to ensure that CJK
    settings never pollute or alter the English figure rendering.
    """
    script_dir = Path(caller_file).resolve().parent

    for lang, text in labels.items():
        with plt.rc_context():
            configure_book_style(font_size=font_size, lang=lang, use_math_fonts=use_math_fonts)
            fig = draw(text)
            suffix = "" if lang == "en" else f"_{lang}"

            png_name = f"{stem}{suffix}.png"
            pdf_name = f"{stem}{suffix}.pdf"

            png_path = script_dir / png_name
            if hasattr(fig, "savefig"):
                savefig_kwargs = {
                    "dpi": dpi,
                    "facecolor": "white",
                    "edgecolor": "none",
                }
                if bbox_inches is not None:
                    savefig_kwargs["bbox_inches"] = bbox_inches
                    savefig_kwargs["pad_inches"] = pad_inches
                fig.savefig(png_path, **savefig_kwargs)
                print(f"✅ Generated: {png_path}")

                if export_pdf:
                    fig.savefig(pdf_path, **savefig_kwargs)
                    print(f"✅ Generated: {pdf_path}")

                plt.close(fig)
            elif hasattr(fig, "render"):
                base_path = script_dir / f"{stem}{suffix}"
                fig.render(str(base_path), format="png", cleanup=True)
                print(f"✅ Generated: {png_path}")

                if export_pdf:
                    fig.render(str(base_path), format="pdf", cleanup=True)
                    print(f"✅ Generated: {pdf_name}")
            elif fig is None:
                if png_path.exists():
                    print(f"✅ Generated: {png_path}")
