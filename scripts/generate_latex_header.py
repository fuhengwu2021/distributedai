#!/usr/bin/env python3
"""
Generate LaTeX header from template file.
Usage:
    python3 generate_latex_header.py <template_file> <header_title> <chapter_style>
"""

import sys
import os

def main():
    if len(sys.argv) < 4:
        sys.stderr.write("Usage: generate_latex_header.py <template_file> <header_title> <chapter_style> [chapter_number]\n")
        sys.exit(1)
    
    template_file = sys.argv[1]
    header_title = sys.argv[2]
    chapter_style = sys.argv[3]
    chapter_number = sys.argv[4] if len(sys.argv) > 4 else ""
    
    if not template_file:
        sys.stderr.write("Error: Template file path is required\n")
        sys.exit(1)
    
    if not os.path.exists(template_file):
        sys.stderr.write(f"Error: Template file not found: {template_file}\n")
        sys.exit(1)
    
    # Read template file
    with open(template_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace placeholders
    content = content.replace('{{header_title}}', header_title)
    # Replace chapter number placeholder if provided
    if chapter_number and chapter_number.strip():
        content = content.replace('{{chapter_number}}', chapter_number.strip())
    else:
        # If no chapter number provided, use a special marker that LaTeX can detect
        content = content.replace('{{chapter_number}}', 'NOSUCHCHAPTERNUMBER')
    
    # Generate chapter number style code based on style
    if chapter_style == 'square':
        style_code = r"""    % Square style
    \coordinate (square-center) at ($(current page.north east) + (-2cm,-2cm)$);
    \coordinate (chapter-pos) at (square-center);
    % Blue background and border (same color)
    \fill[chapterblue] ($(square-center) + (-2cm,-2cm)$) rectangle 
      ($(square-center) + (2cm,2cm)$);
    % Note: Chapter number will be white (set in template)"""
    else:
        style_code = r"""    % Circle style (default) - quarter circle
    \coordinate (circle-center) at (current page.north east);
    \coordinate (chapter-pos) at ($(circle-center) + (-1.5cm,-1.5cm)$);
    % White background - quarter circle area (no border on top/right edges)
    \fill[white] ($(circle-center) + (-4cm,0)$) arc (180:270:4cm) -- (circle-center) -- cycle;
    % Blue border only on the curved arc (not on top/right edges)
    \draw[chapterblue,line width=2pt] ($(circle-center) + (-4cm,0)$) arc (180:270:4cm);"""
    
    # Replace the placeholder in template
    if 'CHAPTER_NUMBER_STYLE_PLACEHOLDER' in content:
        content = content.replace('CHAPTER_NUMBER_STYLE_PLACEHOLDER', style_code)
    
    # Set chapter number color based on style (white for square, black for circle)
    chapter_number_color = 'white' if chapter_style == 'square' else 'black'
    if 'CHAPTER_NUMBER_COLOR_PLACEHOLDER' in content:
        content = content.replace('CHAPTER_NUMBER_COLOR_PLACEHOLDER', chapter_number_color)
    
    # Output rendered content
    print(content, end='')

if __name__ == '__main__':
    main()
