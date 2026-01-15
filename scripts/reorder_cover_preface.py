#!/usr/bin/env python3
"""
Reorder cover page and preface in LaTeX file to appear before TOC.
This script is called from build_book_from_markdown.sh to fix the order
of cover and preface pages that Pandoc places incorrectly.
"""

import sys
import os
import re
import argparse


def main():
    parser = argparse.ArgumentParser(description='Reorder cover and preface in LaTeX file')
    parser.add_argument('tex_file', help='Path to LaTeX file to process')
    parser.add_argument('--cover-bg', default='', help='Path to cover background file')
    parser.add_argument('--cover-is-full-page', action='store_true', 
                       help='Whether cover is a full-page PDF (from Canva)')
    parser.add_argument('--preface-file', default='', help='Path to preface markdown file')
    
    args = parser.parse_args()
    
    tex_file = args.tex_file
    cover_bg_abs = args.cover_bg
    cover_is_full_page = args.cover_is_full_page
    preface_file_path = args.preface_file
    
    # Generate cover LaTeX code
    if cover_bg_abs:
        if cover_is_full_page:
            # Full-page cover PDF (from Canva) - use includepdf
            cover_latex = r"\phantomsection" + "\n" + r"\pdfbookmark[0]{Cover}{cover}" + "\n" + r"\includepdf[pages={1},pagecommand={\thispagestyle{empty}}]{" + cover_bg_abs + "}"
        else:
            # Background image - use coverpagewithbackground
            cover_latex = r"\phantomsection" + "\n" + r"\pdfbookmark[0]{Cover}{cover}" + "\n" + r"\coverpagewithbackground{" + cover_bg_abs + "}"
    else:
        cover_latex = None
    has_preface = os.path.exists(preface_file_path) if preface_file_path else False

    if not os.path.exists(tex_file):
        print(f"  Error: File not found: {tex_file}", file=sys.stderr)
        sys.exit(1)

    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.splitlines(True)

    # Find \begin{document}, TOC, cover, and preface
    doc_start = None
    toc_start = None
    cover_start = None
    preface_start = None
    preface_end = None

    # First pass: find document start and TOC
    for i, line in enumerate(lines):
        if r'\begin{document}' in line:
            doc_start = i
        if r'\tableofcontents' in line:
            toc_start = i
        # Look for cover in various formats
        if cover_latex:
            # Check for cover command directly (both includepdf and coverpagewithbackground)
            if r'\coverpagewithbackground{' in line or r'\includepdf[' in line:
                # Skip comment lines and command definitions
                if line.strip().startswith('%') or r'\newcommand' in line:
                    pass  # Skip
                # For includepdf, only match if it's for the cover file
                elif r'\includepdf[' in line and 'cover.pdf' not in line and 'cover-background' not in line:
                    # Check if nearby lines have Cover bookmark
                    if not any(r'\pdfbookmark[0]{Cover}' in lines[j] for j in range(max(0, i-5), min(i+2, len(lines)))):
                        pass  # Skip - not a cover includepdf
                    else:
                        cover_start = i
                else:
                    cover_start = i
            # Also check for cover in code block format (```{=latex} ... ```)
            elif i > 0 and '```{=latex}' in lines[i-1]:
                if r'\coverpagewithbackground{' in line or r'\includepdf[' in line:
                    cover_start = i - 1  # Start from the code block opening

    # Second pass: find preface (can be anywhere in the file)
    if has_preface:
        for i, line in enumerate(lines):
            # Look for \chapter{Preface} or \frontmatter followed by Preface
            if r'\chapter{Preface}' in line or (r'\frontmatter' in line and i+5 < len(lines) and any(r'Preface' in lines[j] for j in range(i, min(i+10, len(lines))))):
                # If we found \frontmatter, the preface starts there
                if r'\frontmatter' in line:
                    preface_start = i
                else:
                    # If we found \chapter{Preface}, look backwards for \frontmatter or \newpage
                    for j in range(max(0, i-20), i):
                        if r'\frontmatter' in lines[j]:
                            preface_start = j
                            break
                        elif r'\newpage' in lines[j] and preface_start is None:
                            preface_start = j
                
                # Find the end of preface - look for \mainmatter or chaptertitlepage after preface
                if preface_start is not None:
                    # Look for the first \mainmatter that comes after the preface content
                    # or the first chaptertitlepage
                    for j in range(i, len(lines)):
                        # Check for chaptertitlepage first (most reliable marker)
                        if r'\begin{chaptertitlepage}' in lines[j] and j > preface_start + 30:
                            # Find the \newpage before chaptertitlepage (but after preface content)
                            for k in range(j-1, max(j-20, preface_start + 30), -1):
                                if r'\newpage' in lines[k]:
                                    preface_end = k + 1
                                    break
                            if preface_end:
                                break
                        # Also check for \mainmatter that's after preface (not the one before TOC)
                        if r'\mainmatter' in lines[j] and j > preface_start + 30:
                            # Check if this mainmatter is after the preface's \setcounter{secnumdepth}{3}
                            # Look backwards to see if we've passed the preface content
                            found_secnumdepth_reset = False
                            for k in range(j-1, max(j-50, preface_start), -1):
                                if r'\setcounter{secnumdepth}{3}' in lines[k]:
                                    found_secnumdepth_reset = True
                                    break
                            if found_secnumdepth_reset:
                                # This is the mainmatter after preface, find \newpage before it
                                for k in range(j-1, max(j-20, preface_start + 30), -1):
                                    if r'\newpage' in lines[k]:
                                        preface_end = k + 1
                                        break
                                if preface_end:
                                    break
                    
                    # If we still don't have an end, use a fallback: look for \newpage before chaptertitlepage
                    if preface_end is None and preface_start is not None:
                        for j in range(preface_start + 100, min(preface_start + 200, len(lines))):
                            if r'\begin{chaptertitlepage}' in lines[j]:
                                # Go back to find the last \newpage before chapter
                                for k in range(j-1, max(j-10, preface_start + 50), -1):
                                    if r'\newpage' in lines[k]:
                                        preface_end = k + 1
                                        break
                                if preface_end:
                                    break
                    
                    # If we found the start, we're done
                    break

    if doc_start is not None and toc_start is not None:
        # Build insertion list in order: cover, preface
        to_insert_before_toc = []
        
        # Handle cover
        if cover_latex:
            # If cover not found, we need to insert it
            # If cover found but after TOC, we need to move it
            cover_before_toc = cover_start is not None and cover_start < toc_start
            if cover_start is None or not cover_before_toc:
                # Remove cover if it exists (to reinsert in correct position)
                if cover_start is not None:
                    # Find the end of the cover block (it might span multiple lines)
                    cover_end = cover_start
                    # Look for the closing brace and newline
                    while cover_end < len(lines):
                        if r'\coverpagewithbackground{' in lines[cover_end] or r'\includepdf[' in lines[cover_end]:
                            # Find the matching closing brace
                            brace_count = lines[cover_end].count('{') - lines[cover_end].count('}')
                            cover_end += 1
                            while cover_end < len(lines) and brace_count > 0:
                                brace_count += lines[cover_end].count('{') - lines[cover_end].count('}')
                                cover_end += 1
                            # Also check for the closing ``` if it's in a code block
                            if cover_end < len(lines) and '```' in lines[cover_end]:
                                cover_end += 1
                            break
                        cover_end += 1
                    # Remove the cover block
                    lines = lines[:cover_start] + lines[cover_end:]
                    # Update indices
                    if toc_start > cover_start:
                        toc_start -= (cover_end - cover_start)
                    if preface_start and preface_start > cover_start:
                        preface_start -= (cover_end - cover_start)
                # Add cover to insertion list (whether it existed or not)
                to_insert_before_toc.append(('cover', cover_latex))
        
        # Handle preface
        if has_preface:
            if preface_start is not None:
                print(f"  Debug: Found preface at line {preface_start+1}, end at {preface_end+1 if preface_end else 'unknown'}", file=sys.stderr)
                preface_before_toc = preface_start < toc_start
                if not preface_before_toc:
                    # Remove preface if it exists after TOC
                    if preface_end:
                        preface_content = ''.join(lines[preface_start:preface_end])
                        # Remove any duplicate \addcontentsline for Preface to avoid double TOC entry
                        # Pandoc will add it automatically when processing # Preface
                        preface_content = re.sub(r'\\addcontentsline\{toc\}\{chapter\}\{Preface\}.*?\n', '', preface_content)
                        # Remove the preface's \frontmatter since we'll be in the same frontmatter as TOC
                        # Split into lines and remove any line that's just \frontmatter
                        preface_lines = preface_content.splitlines(True)
                        filtered_lines = []
                        for line in preface_lines:
                            if r'\frontmatter' not in line.strip():
                                filtered_lines.append(line)
                        preface_content = ''.join(filtered_lines)
                        # Ensure preface has PDF bookmark if not already present
                        if r'\pdfbookmark' not in preface_content:
                            # Find \phantomsection and add bookmark after it
                            if r'\phantomsection' in preface_content:
                                preface_content = preface_content.replace(
                                    r'\phantomsection',
                                    r'\phantomsection' + '\n' + r'\pdfbookmark[0]{Preface}{preface}'
                                )
                            else:
                                # Add both phantomsection and bookmark at the beginning
                                preface_content = r'\phantomsection' + '\n' + r'\pdfbookmark[0]{Preface}{preface}' + '\n' + preface_content
                        lines = lines[:preface_start] + lines[preface_end:]
                        if toc_start > preface_start:
                            toc_start -= (preface_end - preface_start)
                        to_insert_before_toc.append(('preface', preface_content))
                        print(f"  Debug: Moving preface from line {preface_start+1} to before TOC", file=sys.stderr)
                    else:
                        print(f"  Warning: Found preface start at line {preface_start+1} but could not find end", file=sys.stderr)
                else:
                    # Preface is already before TOC, but ensure it has a bookmark
                    if preface_end:
                        preface_content = ''.join(lines[preface_start:preface_end])
                        # Check if bookmark exists, if not add it
                        if r'\pdfbookmark' not in preface_content:
                            new_lines = []
                            bookmark_added = False
                            for i, line in enumerate(lines[preface_start:preface_end]):
                                new_lines.append(line)
                                # Add bookmark after phantomsection
                                if r'\phantomsection' in line and not bookmark_added:
                                    new_lines.append(r'\pdfbookmark[0]{Preface}{preface}' + '\n')
                                    bookmark_added = True
                            # If no phantomsection found, add both at the start
                            if not bookmark_added:
                                new_lines.insert(0, r'\phantomsection' + '\n')
                                new_lines.insert(1, r'\pdfbookmark[0]{Preface}{preface}' + '\n')
                            lines = lines[:preface_start] + new_lines + lines[preface_end:]
                    print(f"  Debug: Preface already before TOC at line {preface_start+1}", file=sys.stderr)
            else:
                print(f"  Warning: Could not find preface in LaTeX file", file=sys.stderr)
        
        # Insert elements after \begin{document}, before TOC
        if to_insert_before_toc:
            insertion = ['\n']
            for name, content in to_insert_before_toc:
                insertion.append(content)
                if not content.endswith('\n'):
                    insertion.append('\n')
                insertion.append('\n')
            
            # Insert before TOC, not just after \begin{document}
            # We want: cover -> first \frontmatter -> preface -> TOC block
            # So insert preface AFTER the first \frontmatter but BEFORE the TOC block
            insert_pos = doc_start + 1
            first_frontmatter = None
            toc_block_start = None
            
            for i in range(doc_start + 1, min(doc_start + 50, len(lines))):
                # Find first \frontmatter (this is Pandoc's frontmatter for TOC)
                if first_frontmatter is None and r'\frontmatter' in lines[i]:
                    first_frontmatter = i
                # Find the opening brace { that contains TOC (comes after first frontmatter)
                if lines[i].strip() == '{' and first_frontmatter is not None:
                    # Check if this block contains \tableofcontents
                    for j in range(i+1, min(i+20, len(lines))):
                        if r'\tableofcontents' in lines[j]:
                            toc_block_start = i
                            break
                    if toc_block_start:
                        break
            
            # Insert AFTER the first \frontmatter but BEFORE the TOC block
            if toc_block_start is not None:
                insert_pos = toc_block_start
            elif first_frontmatter is not None:
                # Insert right after the first \frontmatter
                insert_pos = first_frontmatter + 1
            else:
                # Fallback: find \tableofcontents directly
                for i in range(doc_start + 1, min(doc_start + 50, len(lines))):
                    if r'\tableofcontents' in lines[i]:
                        insert_pos = i
                        break
            
            new_lines = lines[:insert_pos] + insertion + lines[insert_pos:]
            with open(tex_file, 'w', encoding='utf-8') as f:
                f.write(''.join(new_lines))
            elements = [name for name, _ in to_insert_before_toc]
            print(f"  Reordered: {' and '.join(elements)} now appear before TOC", file=sys.stderr)
        else:
            print("  Cover page and preface already before TOC", file=sys.stderr)
    else:
        print("  Warning: Could not find document start or TOC", file=sys.stderr)


if __name__ == '__main__':
    main()
