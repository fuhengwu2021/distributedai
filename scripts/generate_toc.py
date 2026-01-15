#!/usr/bin/env python3
"""
Script to automatically generate toc.md from chapter files.

This script:
1. Scans all chapter directories (chapter*-*)
2. Extracts chapter number, title, and subtitle from each chapter.md file
3. Organizes chapters by part (based on chapter number ranges)
4. Generates toc.md file

Usage:
    python3 scripts/generate_toc.py
    python3 scripts/generate_toc.py --output toc.md
"""

import re
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional


def extract_chapter_info(chapter_file: Path) -> Optional[Tuple[int, str, str]]:
    """
    Extract chapter number, title, and subtitle from a chapter markdown file.
    
    Returns:
        Tuple of (chapter_number, title, subtitle) or None if not found
    """
    try:
        with open(chapter_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find chapter title (first line starting with # Chapter)
        chapter_title = None
        subtitle = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Match chapter title: "# Chapter N: Title {-}"
            if line.startswith('# Chapter'):
                # Extract chapter number and title
                match = re.match(r'^# Chapter\s+(\d+):\s*(.+?)\s*\{-?\}?$', line)
                if match:
                    chapter_num = int(match.group(1))
                    title = match.group(2).strip()
                    chapter_title = (chapter_num, title)
                    break
        
        # Find subtitle (first italic line after chapter title)
        if chapter_title:
            for i in range(len(lines)):
                line = lines[i].strip()
                # Match italic subtitle: "*subtitle text*"
                if line.startswith('*') and line.endswith('*') and len(line) > 2:
                    subtitle = line[1:-1].strip()  # Remove asterisks
                    break
        
        if chapter_title:
            return (chapter_title[0], chapter_title[1], subtitle or "")
        
    except Exception as e:
        print(f"Warning: Could not read {chapter_file}: {e}", file=sys.stderr)
    
    return None


def find_chapter_directories(root_dir: Path) -> List[Path]:
    """Find all chapter directories matching pattern chapter*-*"""
    chapters = []
    for item in root_dir.iterdir():
        if item.is_dir() and item.name.startswith('chapter') and '-' in item.name:
            chapters.append(item)
    return sorted(chapters, key=lambda x: extract_chapter_number_from_dir(x.name))


def extract_chapter_number_from_dir(dirname: str) -> int:
    """Extract chapter number from directory name (e.g., 'chapter1-vector-space' -> 1)"""
    match = re.match(r'chapter(\d+)', dirname)
    if match:
        return int(match.group(1))
    return 999  # Put unknown chapters at the end


def find_chapter_file(chapter_dir: Path) -> Optional[Path]:
    """Find the chapter markdown file in a chapter directory"""
    # Look for chapter*.md files
    for md_file in chapter_dir.glob('chapter*.md'):
        return md_file
    return None


def get_pdf_page_count(pdf_file: Path) -> Optional[int]:
    """Get the number of pages in a PDF file using pdfinfo"""
    try:
        result = subprocess.run(
            ['pdfinfo', str(pdf_file)],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Look for "Pages: N" in the output
            match = re.search(r'Pages:\s*(\d+)', result.stdout)
            if match:
                return int(match.group(1))
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return None


def find_chapter_pdf(chapter_dir: Path, chapter_num: int) -> Optional[Path]:
    """Find the chapter PDF file in a chapter directory"""
    # Look for chapterN.pdf files
    pdf_file = chapter_dir / f"chapter{chapter_num}.pdf"
    if pdf_file.exists():
        return pdf_file
    # Also check for any PDF in the directory
    for pdf in chapter_dir.glob("*.pdf"):
        if f"chapter{chapter_num}" in pdf.name.lower():
            return pdf
    return None


def organize_into_parts(chapters: List[Tuple[int, str, str, Optional[int], Optional[int]]]) -> List[Tuple[str, List[Tuple[int, str, str, Optional[int], Optional[int]]]]]:
    """
    Organize chapters into parts based on chapter number ranges.
    This is a simple heuristic - you may want to customize this.
    """
    parts = []
    
    # Part I: Linear Algebra (Chapters 1-10)
    part1 = [ch for ch in chapters if 1 <= ch[0] <= 10]
    if part1:
        parts.append(("Part I: Linear Algebra Foundations", part1))
    
    # Part II: Calculus and Optimization (Chapters 11-12)
    part2 = [ch for ch in chapters if 11 <= ch[0] <= 12]
    if part2:
        parts.append(("Part II: Calculus and Optimization", part2))
    
    # Part III: Probability and Statistics (Chapters 13-15)
    part3 = [ch for ch in chapters if 13 <= ch[0] <= 15]
    if part3:
        parts.append(("Part III: Probability and Statistics", part3))
    
    # Part IV: Advanced Topics (Chapters 16+)
    part4 = [ch for ch in chapters if ch[0] >= 16]
    if part4:
        parts.append(("Part IV: Advanced Topics", part4))
    
    return parts


def generate_toc_markdown(parts: List[Tuple[str, List[Tuple[int, str, str, Optional[int], Optional[int]]]]], output_file: Path):
    """Generate the toc.md file content"""
    lines = [
        "---",
        "title: \"Table of Contents\"",
        "subtitle: \"Math for AI/ML - A comprehensive mathematics textbook for AI/ML\"",
        "date: \"\"",
        "toc: true",
        "toc-own-page: true",
        "toc-depth: 3",
        "numbersections: true",
        "lang: \"en\"",
        "...",
        ""
    ]
    
    for part_name, chapters in parts:
        lines.append(f"## {part_name}")
        lines.append("")
        
        for chapter_num, title, subtitle, start_page, end_page in chapters:
            # Format chapter entry with page numbers if available
            if start_page is not None and end_page is not None:
                if start_page == end_page:
                    page_info = f" ({start_page})"
                else:
                    page_info = f" ({start_page}--{end_page})"
            else:
                page_info = ""
            
            lines.append(f"### Chapter {chapter_num}: {title}{page_info}")
            if subtitle:
                lines.append(f"*{subtitle}*")
            lines.append("")
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Generated {output_file} with {sum(len(chapters) for _, chapters in parts)} chapters")


def main():
    """Main function"""
    import argparse
    import subprocess
    
    parser = argparse.ArgumentParser(description='Generate toc.md from chapter files')
    parser.add_argument('--output', '-o', default='toc.md', 
                       help='Output file path (default: toc.md)')
    parser.add_argument('--root', '-r', default='.', 
                       help='Root directory to search for chapters (default: current directory)')
    parser.add_argument('--pdf', '-p', action='store_true',
                       help='Also convert toc.md to PDF after generation')
    
    args = parser.parse_args()
    
    root_dir = Path(args.root).resolve()
    output_file = Path(args.output)
    
    if not root_dir.exists():
        print(f"Error: Root directory {root_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Find all chapter directories
    chapter_dirs = find_chapter_directories(root_dir)
    
    if not chapter_dirs:
        print(f"Warning: No chapter directories found in {root_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Extract chapter information and page counts
    chapters = []
    cumulative_pages = 1  # Start from page 1
    
    for chapter_dir in chapter_dirs:
        chapter_file = find_chapter_file(chapter_dir)
        if chapter_file:
            info = extract_chapter_info(chapter_file)
            if info:
                chapter_num, title, subtitle = info
                # Try to find PDF and get page count
                pdf_file = find_chapter_pdf(chapter_dir, chapter_num)
                page_count = None
                start_page = cumulative_pages
                
                if pdf_file and pdf_file.exists():
                    page_count = get_pdf_page_count(pdf_file)
                    if page_count:
                        end_page = start_page + page_count - 1
                        # Add page range to the chapter info
                        chapters.append((chapter_num, title, subtitle, start_page, end_page))
                        cumulative_pages = end_page + 1
                    else:
                        chapters.append((chapter_num, title, subtitle, None, None))
                else:
                    chapters.append((chapter_num, title, subtitle, None, None))
            else:
                print(f"Warning: Could not extract info from {chapter_file}", file=sys.stderr)
        else:
            print(f"Warning: No chapter*.md file found in {chapter_dir}", file=sys.stderr)
    
    if not chapters:
        print("Error: No chapter information extracted", file=sys.stderr)
        sys.exit(1)
    
    # Sort by chapter number (first element of tuple)
    chapters.sort(key=lambda x: x[0])
    
    # Organize into parts
    parts = organize_into_parts(chapters)
    
    # Generate toc.md
    generate_toc_markdown(parts, output_file)
    
    # Convert to PDF if requested
    if args.pdf:
        script_dir = Path(__file__).parent
        convert_script = script_dir / 'convert_toc_to_pdf.sh'
        if convert_script.exists():
            print(f"\n📄 Converting {output_file} to PDF...")
            try:
                subprocess.run(['bash', str(convert_script)], check=True, cwd=root_dir)
            except subprocess.CalledProcessError:
                print(f"⚠️  Warning: PDF conversion failed. You can run manually:")
                print(f"   ./scripts/convert_toc_to_pdf.sh")
        else:
            print(f"⚠️  Warning: Conversion script not found at {convert_script}")
            print(f"   You can convert manually using pandoc:")
            print(f"   pandoc {output_file} -o {output_file.with_suffix('.pdf')} --pdf-engine=xelatex")


if __name__ == '__main__':
    main()
