#!/usr/bin/env python3
"""
Convert foreignObject elements to native SVG text elements.

This script converts SVG files that use foreignObject (HTML inside SVG) to use
native SVG text elements instead. This is necessary because many PDF generators
(pandoc, LaTeX, etc.) don't support foreignObject elements.

Usage:
    python3 convert_svg_foreignobject_to_text.py input.svg output.svg
    python3 convert_svg_foreignobject_to_text.py input.svg  # overwrites input.svg
"""

import sys
import re
from xml.dom import minidom
from pathlib import Path


def get_text_from_node(node):
    """Recursively extract text content from a node."""
    text = ""
    if node.nodeType == node.TEXT_NODE:
        return node.nodeValue or ""
    for child in node.childNodes:
        text += get_text_from_node(child)
    return text


def convert_foreignobject_to_text(svg_file, output_file=None):
    """
    Convert foreignObject elements to native SVG text elements.
    
    Args:
        svg_file: Path to input SVG file
        output_file: Path to output SVG file (if None, overwrites input)
    """
    svg_path = Path(svg_file)
    if not svg_path.exists():
        raise FileNotFoundError(f"SVG file not found: {svg_file}")
    
    if output_file is None:
        output_file = svg_file
    
    # Read SVG as string first to preserve structure
    with open(svg_path, 'r', encoding='utf-8') as f:
        svg_content = f.read()
    
    # Parse with minidom to handle namespaces better
    dom = minidom.parseString(svg_content)
    
    # Find all foreignObject elements
    foreign_objects = dom.getElementsByTagName('foreignObject')
    
    print(f"Found {len(foreign_objects)} foreignObject elements in {svg_file}")
    
    # Collect replacements (can't modify while iterating)
    replacements = []
    
    # For each foreignObject, extract text and convert to SVG text
    for fo in foreign_objects:
        # Get position and size
        x = float(fo.getAttribute('x') or 0)
        y = float(fo.getAttribute('y') or 0)
        width = float(fo.getAttribute('width') or 0)
        height = float(fo.getAttribute('height') or 0)
        
        # Skip if width/height is 0 (invisible elements)
        if width == 0 or height == 0:
            continue
        
        # Extract text content recursively
        text_content = get_text_from_node(fo).strip()
        
        if not text_content:
            continue
        
        # Get style from foreignObject
        style = fo.getAttribute('style') or ""
        # Also check for class to get font styling
        class_attr = fo.getAttribute('class') or ""
        
        # Create SVG text element
        text_elem = dom.createElement('text')
        text_elem.setAttribute('x', str(x + width/2))  # Center horizontally
        text_elem.setAttribute('y', str(y + height/2))  # Center vertically
        text_elem.setAttribute('text-anchor', 'middle')
        text_elem.setAttribute('dominant-baseline', 'middle')
        if style:
            text_elem.setAttribute('style', style)
        if class_attr:
            text_elem.setAttribute('class', class_attr)
        text_elem.appendChild(dom.createTextNode(text_content))
        
        replacements.append((fo, text_elem))
    
    # Apply replacements
    for fo, text_elem in replacements:
        fo.parentNode.replaceChild(text_elem, fo)
    
    # Write output
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(dom.toxml())
    
    print(f"Converted {len(replacements)} foreignObject elements to text")
    print(f"Converted SVG written to {output_file}")
    
    return len(replacements)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        convert_foreignobject_to_text(input_file, output_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
