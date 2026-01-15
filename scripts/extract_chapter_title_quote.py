#!/usr/bin/env python3
"""
Script to extract chapter title, subtitle, and quote from Markdown and generate LaTeX code.

This script:
1. Reads a markdown file
2. Extracts chapter number, title, and subtitle from the first # heading
3. Extracts quote block (blockquote + author)
4. Generates LaTeX chaptertitlepage and chapterquote code
5. Can output LaTeX code only or process entire file for PDF conversion

Usage:
    python3 scripts/extract_chapter_title_quote.py --generate-only chapter1.md
    python3 scripts/extract_chapter_title_quote.py --process-file input.md output.md
"""

import re
import sys
import argparse
from pathlib import Path


def extract_chapter_info(md_content):
    """
    Extract chapter number, title, and subtitle from markdown.
    
    Expected format:
    # Chapter 1: Title Here
    
    *Subtitle here*
    
    Returns: (chapter_num, title, subtitle) or (None, None, None) if not found
    """
    lines = md_content.split('\n')
    
    # Find the first # heading that looks like a chapter
    chapter_pattern = r'^#\s+Chapter\s+(\d+):\s*(.+)$'
    subtitle_pattern = r'^\*\s*(.+?)\s*\*$'
    
    chapter_num = None
    title = None
    subtitle = None
    
    for i, line in enumerate(lines):
        # Match chapter heading
        match = re.match(chapter_pattern, line)
        if match:
            chapter_num = match.group(1)
            title = match.group(2).strip()
            # Remove Pandoc attributes like {-} or {.unnumbered} from title
            title = re.sub(r'\s*\{[^}]*\}\s*$', '', title).strip()
            
            # Look for subtitle in the next few lines
            for j in range(i + 1, min(i + 5, len(lines))):
                subtitle_match = re.match(subtitle_pattern, lines[j])
                if subtitle_match:
                    subtitle = subtitle_match.group(1).strip()
                    break
            
            return chapter_num, title, subtitle
    
    return None, None, None


def format_title_for_latex(title):
    """
    Format title for LaTeX, handling line breaks.
    If title is long, suggest a break point.
    """
    # Simple heuristic: if title is short, don't break
    if len(title) <= 35:
        return title
    
    # Try to break at common connecting words (keep the word in first part)
    break_points = [' to ', ' and ', ' with ', ' for ', ' of ', ' in ', ' on ']
    for bp in break_points:
        if bp in title:
            parts = title.split(bp, 1)
            # Keep the connecting word with the first part
            first_part = parts[0] + bp.rstrip()
            if len(first_part) < 40 and len(parts[0]) > 10:
                return f"{first_part}\\\\[0.3cm]{parts[1]}"
    
    # Fallback: break at space near middle, but prefer breaking after "to"
    words = title.split()
    if len(words) > 1:
        # Look for "to" and break after it
        for i, word in enumerate(words):
            if word.lower() == 'to' and i < len(words) - 1:
                first_half = ' '.join(words[:i+1])
                second_half = ' '.join(words[i+1:])
                return f"{first_half}\\\\[0.3cm]{second_half}"
        
        # No "to" found, break at middle
        mid = len(words) // 2
        first_half = ' '.join(words[:mid])
        second_half = ' '.join(words[mid:])
        return f"{first_half}\\\\[0.3cm]{second_half}"
    
    return title


def clean_title_for_metadata(title):
    """
    Remove LaTeX commands from title for use in PDF metadata (bookmarks, etc.).
    This removes line break commands like \\[0.3cm] and replaces them with spaces.
    """
    import re
    # Remove LaTeX line break commands: \\[0.3cm], \\[0.5cm], etc.
    cleaned = re.sub(r'\\\\\[[^\]]+\]', ' ', title)
    # Remove extra spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def generate_latex_title_page(chapter_num, title, subtitle):
    """Generate LaTeX chaptertitlepage code."""
    formatted_title = format_title_for_latex(title)
    
    latex_code = f"""```{{=latex}}
\\begin{{chaptertitlepage}}{{{chapter_num}}}{{{formatted_title}}}{{{subtitle or ''}}}
\\end{{chaptertitlepage}}
```"""
    
    return latex_code


def generate_latex_title_page_with_quote(chapter_num, title, subtitle, quote_text, quote_author, code_summary=""):
    """Generate LaTeX chaptertitlepage code with optional quote and code summary."""
    formatted_title = format_title_for_latex(title)
    
    # Build quote content if available
    quote_content = ""
    if quote_text and quote_author:
        # Remove quotes from text since LaTeX will add proper quote marks
        unquoted_text = quote_text.strip('"').strip("'")
        # Escape special LaTeX characters in quote content
        # Style: quote text left-aligned in italic, author right-aligned in smaller font
        # Add LaTeX quote marks only around the quote text, not the author
        quote_content = f"``{unquoted_text}''\\\\[0.5cm]\n\\hfill\\textnormal{{\\fontsize{{9}}{{11}}\\selectfont--- {quote_author}}}"
    
    # Build code summary content if available
    code_summary_content = ""
    if code_summary:
        # Convert markdown list items to LaTeX format
        # Each line starting with "- `code`: description" becomes a LaTeX item
        summary_lines = code_summary.split('\n')
        latex_items = []
        for line in summary_lines:
            if line.strip().startswith('-'):
                # Extract code and description: - `code`: description or - `code` (extra): description
                # Handle both formats: - `code`: desc and - `code` (extra): desc
                match = re.match(r'^-\s*`([^`]+)`(?:\s*\([^)]+\))?:\s*(.+)$', line.strip())
                if match:
                    code_part = match.group(1)
                    desc_part = match.group(2)
                    # Escape LaTeX special characters in description
                    # But preserve LaTeX commands and math mode expressions
                    # Strategy: protect them first, escape the rest, then restore
                    import re as re_module
                    
                    # Step 1: Protect math mode expressions $...$
                    math_pattern = r'\$[^$]+\$'
                    math_placeholders = {}
                    math_counter = [0]
                    def protect_math(match):
                        placeholder = f"XXXMATH{math_counter[0]}XXX"
                        math_placeholders[placeholder] = match.group(0)
                        math_counter[0] += 1
                        return placeholder
                    desc_protected = re_module.sub(math_pattern, protect_math, desc_part)
                    
                    # Step 2: Protect LaTeX commands \command or \command{arg}
                    # Match: \ followed by letters (command name), optionally followed by {arg}
                    latex_cmd_pattern = r'\\[a-zA-Z]+\*?(?:\{[^}]*\})*'
                    cmd_placeholders = {}
                    cmd_counter = [0]
                    def protect_cmd(match):
                        placeholder = f"XXXCMD{cmd_counter[0]}XXX"
                        cmd_placeholders[placeholder] = match.group(0)
                        cmd_counter[0] += 1
                        return placeholder
                    desc_protected = re_module.sub(latex_cmd_pattern, protect_cmd, desc_protected)
                    
                    # Step 3: Now escape special characters in the remaining text
                    desc_escaped = desc_protected.replace('\\', '\\textbackslash{}').replace('{', '\\{').replace('}', '\\}').replace('&', '\\&').replace('%', '\\%').replace('#', '\\#').replace('^', '\\textasciicircum{}').replace('_', '\\_')
                    
                    # Step 4: Restore LaTeX commands first (they may contain {})
                    for placeholder, cmd_expr in sorted(cmd_placeholders.items(), key=lambda x: -len(x[0])):
                        desc_escaped = desc_escaped.replace(placeholder, cmd_expr)
                    
                    # Step 5: Restore math mode expressions (they contain $ which is fine in LaTeX)
                    for placeholder, math_expr in sorted(math_placeholders.items(), key=lambda x: -len(x[0])):
                        desc_escaped = desc_escaped.replace(placeholder, math_expr)
                    # Escape LaTeX special characters in code part
                    code_escaped = code_part.replace('\\', '\\textbackslash{}').replace('{', '\\{').replace('}', '\\}').replace('_', '\\_')
                    # Add dark blue background with white text and rounded corners for code
                    #latex_items.append("\\item \\tcbox[colback=chapterblue,coltext=white,boxrule=0pt,arc=2pt,left=2pt,right=2pt,top=1pt,bottom=1pt]{\\texttt{" + code_escaped + "}} " + desc_escaped)
                    # Add dark blue border with dark blue text and white background for code
                    # Use \mbox to keep code box and description on the same line
                    # Adjust baseline to align box with text vertically
                    latex_items.append("\\item \\mbox{\\raisebox{-0.85ex}{\\tcbox[colback=white,coltext=chapterblue,colframe=chapterbluelight,boxrule=0.5pt,arc=2pt,left=2pt,right=2pt,top=1pt,bottom=0.5pt]{\\texttt{" + code_escaped + "}}}} " + desc_escaped)
        
        if latex_items:
            # Only generate the list items; decorative line and title are in LaTeX environment
            items_text = "\n".join(latex_items)
            code_summary_content = items_text
    
    # Build LaTeX code with separate parameters for quote (4th) and code summary (5th)
    # IMPORTANT: Even though chaptertitlepage environment definition has \newpage,
    # when the environment is inserted via ```{=latex} block, pandoc may not execute
    # the environment's end code properly. We need to explicitly add \newpage after
    # \end{chaptertitlepage} to ensure proper page separation for wrapfigure to work.
    if quote_content and code_summary_content:
        latex_code = f"""```{{=latex}}
\\begin{{chaptertitlepage}}{{{chapter_num}}}{{{formatted_title}}}{{{subtitle or ''}}}[{quote_content}][{code_summary_content}]
\\end{{chaptertitlepage}}
\\newpage
```"""
    elif quote_content:
        latex_code = f"""```{{=latex}}
\\begin{{chaptertitlepage}}{{{chapter_num}}}{{{formatted_title}}}{{{subtitle or ''}}}[{quote_content}]
\\end{{chaptertitlepage}}
\\newpage
```"""
    elif code_summary_content:
        latex_code = f"""```{{=latex}}
\\begin{{chaptertitlepage}}{{{chapter_num}}}{{{formatted_title}}}{{{subtitle or ''}}}[][{code_summary_content}]
\\end{{chaptertitlepage}}
\\newpage
```"""
    else:
        latex_code = f"""```{{=latex}}
\\begin{{chaptertitlepage}}{{{chapter_num}}}{{{formatted_title}}}{{{subtitle or ''}}}
\\end{{chaptertitlepage}}
\\newpage
```"""
    
    return latex_code


def process_markdown_file(md_file, remove_header=False):
    """Process markdown file and add LaTeX title page."""
    md_path = Path(md_file)
    
    if not md_path.exists():
        print(f"Error: File not found: {md_file}", file=sys.stderr)
        return False
    
    content = md_path.read_text(encoding='utf-8')
    
    # Check if LaTeX title page already exists
    if '\\begin{chaptertitlepage}' in content:
        print(f"Note: LaTeX chaptertitlepage already exists in {md_file}")
        return True
    
    # Extract chapter info
    chapter_num, title, subtitle = extract_chapter_info(content)
    
    if not chapter_num or not title:
        print(f"Warning: Could not extract chapter info from {md_file}", file=sys.stderr)
        print("Expected format: # Chapter 1: Title Here", file=sys.stderr)
        return False
    
    # Generate LaTeX code
    latex_code = generate_latex_title_page(chapter_num, title, subtitle)
    
    # Find where to insert (before the chapter heading)
    chapter_pattern = r'^(#\s+Chapter\s+\d+:.*)$'
    lines = content.split('\n')
    insert_pos = None
    
    for i, line in enumerate(lines):
        if re.match(chapter_pattern, line):
            insert_pos = i
            break
    
    if insert_pos is None:
        print(f"Error: Could not find chapter heading in {md_file}", file=sys.stderr)
        return False
    
    # Build new content
    new_lines = []
    
    # Insert LaTeX code before chapter heading
    new_lines.append(latex_code)
    new_lines.append('')
    
    # Add remaining content, optionally removing the markdown header
    if remove_header:
        # Skip the chapter heading and subtitle
        i = insert_pos
        # Skip chapter heading
        i += 1
        # Skip empty lines
        while i < len(lines) and not lines[i].strip():
            i += 1
        # Skip subtitle if it's italic
        if i < len(lines) and re.match(r'^\*\s*.+\s*\*$', lines[i]):
            i += 1
        # Skip empty lines after subtitle
        while i < len(lines) and not lines[i].strip():
            i += 1
        
        new_lines.extend(lines[i:])
    else:
        # Keep the markdown header
        new_lines.extend(lines[insert_pos:])
    
    # Write back
    new_content = '\n'.join(new_lines)
    md_path.write_text(new_content, encoding='utf-8')
    
    print(f"✓ Added LaTeX chaptertitlepage to {md_file}")
    print(f"  Chapter {chapter_num}: {title}")
    if subtitle:
        print(f"  Subtitle: {subtitle}")
    
    return True


def generate_latex_only(md_file):
    """Generate LaTeX code only, without modifying the file."""
    md_path = Path(md_file)
    
    if not md_path.exists():
        return None
    
    content = md_path.read_text(encoding='utf-8')
    
    # Extract chapter info
    chapter_num, title, subtitle = extract_chapter_info(content)
    
    if not chapter_num or not title:
        return None
    
    # Generate LaTeX code
    latex_code = generate_latex_title_page(chapter_num, title, subtitle)
    return latex_code


def process_file_for_pdf(input_file, output_file):
    """
    Process markdown file and generate LaTeX-enhanced version for PDF conversion.
    Does not modify the source file.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_file}", file=sys.stderr)
        return False
    
    content = input_path.read_text(encoding='utf-8')
    lines = content.split('\n')
    
    # Extract chapter info
    chapter_num, title, subtitle = extract_chapter_info(content)
    
    if not chapter_num or not title:
        # No chapter title found, just copy the file
        output_path.write_text(content, encoding='utf-8')
        return True
    
    # Extract quote first (before generating title page)
    quote_pattern = r'^\>\s*(?:"(.+)"|(.+))$'
    author_pattern = r'^-\s*(.+)$'
    quote_text = ""
    quote_author = ""
    
    # Find quote in content
    for i, line in enumerate(lines):
        quote_match = re.match(quote_pattern, line)
        if quote_match:
            quote_text = quote_match.group(1) or quote_match.group(2)
            if quote_text and quote_text.strip():
                quote_text = quote_text.strip()
                # Skip NOTES: and NOTEE blocks (these are note sections, not quotes)
                if quote_text.startswith('NOTES:') or quote_text.startswith('NOTEE'):
                    continue
                # Look for author on next line
                if i + 1 < len(lines):
                    # Skip empty lines
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j < len(lines):
                        author_match = re.match(author_pattern, lines[j])
                        if author_match:
                            quote_author = author_match.group(1).strip()
                            break
    
    # Extract code summary
    code_summary = ""
    code_summary_start = None
    for i, line in enumerate(lines):
        if re.match(r'^\*\*Code Summary\*\*', line, re.IGNORECASE):
            code_summary_start = i
            break
    
    if code_summary_start is not None:
        # Collect lines until we hit an empty line followed by a heading or end of file
        summary_lines = []
        i = code_summary_start + 1
        # Skip empty line after "**Code Summary**"
        while i < len(lines) and not lines[i].strip():
            i += 1
        # Collect all list items (lines starting with -)
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith('-'):
                summary_lines.append(line.strip())
            elif line.strip() and not line.strip().startswith('#'):
                # Stop if we hit non-list, non-heading content
                break
            elif not line.strip():
                # Check if next non-empty line is a heading
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and lines[j].startswith('#'):
                    break
            else:
                break
            i += 1
        if summary_lines:
            code_summary = '\n'.join(summary_lines)
    
    # Generate LaTeX title page with quote and code summary
    latex_title = generate_latex_title_page_with_quote(chapter_num, title, subtitle, quote_text, quote_author, code_summary)
    
    # Process content: find chapter title, subtitle, and quote
    chapter_pattern = r'^(#\s+Chapter\s+\d+:.*)$'
    subtitle_pattern = r'^\*\s*(.+?)\s*\*$'
    
    output_lines = []
    i = 0
    
    # Add LaTeX title page
    output_lines.append(latex_title)
    output_lines.append('')
    
    while i < len(lines):
        line = lines[i]
        
        # Skip chapter title (already in LaTeX)
        if re.match(chapter_pattern, line):
            i += 1
            # Skip empty lines after chapter title
            while i < len(lines) and not lines[i].strip():
                i += 1
            # Skip subtitle (already in LaTeX)
            if i < len(lines) and re.match(subtitle_pattern, lines[i]):
                i += 1
            # Skip empty lines after subtitle
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue
        
        # Skip code summary block (already included in title page)
        # Check for **Code Summary** (case insensitive, with optional whitespace)
        # This must come BEFORE subtitle check to avoid false matches
        if re.match(r'^\*\*Code Summary\*\*', line, re.IGNORECASE) or '**Code Summary**' in line:
            i += 1
            # Skip empty line if present
            while i < len(lines) and not lines[i].strip():
                i += 1
            # Skip all summary lines (starting with -)
            while i < len(lines) and lines[i].strip().startswith('-'):
                i += 1
            # Skip empty lines after summary
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue
        
        # Skip subtitle if standalone (already in LaTeX)
        if re.match(subtitle_pattern, line):
            i += 1
            # Skip empty lines after subtitle
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue
        
        # Skip quote block (already included in title page)
        # BUT don't skip NOTES: and NOTEE blocks (these are note sections, not quotes)
        quote_match = re.match(quote_pattern, line)
        if quote_match:
            quote_text = quote_match.group(1) or quote_match.group(2)
            # Check if this is a NOTES: or NOTEE block (note sections, not quotes)
            if quote_text and (quote_text.strip().startswith('NOTES:') or quote_text.strip().startswith('NOTEE')):
                # This is a note section, not a quote - keep it
                output_lines.append(line)
                i += 1
                continue
            # This is a regular quote block - skip it (already in title page)
            i += 1
            # Skip empty line if present
            while i < len(lines) and not lines[i].strip():
                i += 1
            # Skip author line (starting with -) - always skip it, Code Summary handler will handle Code Summary separately
            # The author line is the first line starting with - after the quote
            if i < len(lines) and lines[i].strip().startswith('-'):
                i += 1
            # Skip empty lines after author
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue
        
        # Add all other lines
        output_lines.append(line)
        i += 1
    
    # Write output - ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(output_lines), encoding='utf-8')
    
    # Verify file was created
    if not output_path.exists():
        print(f"Error: Failed to create output file: {output_file}", file=sys.stderr)
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Add LaTeX chapter title page from Markdown chapter header'
    )
    parser.add_argument('md_file', nargs='?', help='Markdown file to process')
    parser.add_argument(
        '--remove-header',
        action='store_true',
        help='Remove the Markdown chapter header after adding LaTeX title page'
    )
    parser.add_argument(
        '--generate-only',
        action='store_true',
        help='Only generate LaTeX code, do not modify file (output to stdout)'
    )
    parser.add_argument(
        '--process-file',
        nargs=2,
        metavar=('INPUT', 'OUTPUT'),
        help='Process input file and write LaTeX-enhanced version to output file'
    )
    
    args = parser.parse_args()
    
    if args.process_file:
        input_file, output_file = args.process_file
        success = process_file_for_pdf(input_file, output_file)
        sys.exit(0 if success else 1)
    
    if args.generate_only:
        if not args.md_file:
            print("Error: --generate-only requires md_file argument", file=sys.stderr)
            sys.exit(1)
        latex_code = generate_latex_only(args.md_file)
        if latex_code:
            print(latex_code)
            sys.exit(0)
        else:
            sys.exit(1)
    
    if not args.md_file:
        parser.print_help()
        sys.exit(1)
    
    success = process_markdown_file(args.md_file, args.remove_header)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
