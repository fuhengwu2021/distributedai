#!/usr/bin/env python3
"""
Migrate Code Summary sections from inline content to summary/math.md or summary/numpy.md files.
Updates the Code Summary section to use include patterns like chapter11.
"""

import re
import os
from pathlib import Path


def extract_code_summary(content):
    """Extract Code Summary section from markdown content."""
    lines = content.split('\n')
    code_summary_start = None
    
    # Find Code Summary section
    for i, line in enumerate(lines):
        if re.match(r'^\*\*Code Summary\*\*', line, re.IGNORECASE):
            code_summary_start = i
            break
    
    if code_summary_start is None:
        return None, None
    
    # Extract summary lines (starting with -)
    summary_lines = []
    i = code_summary_start + 1
    
    # Skip empty lines after "**Code Summary**"
    while i < len(lines) and not lines[i].strip():
        i += 1
    
    # Collect all bullet points
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith('-'):
            summary_lines.append(line.rstrip())
        elif line.strip() and not line.strip().startswith('#'):
            # Non-empty line that's not a heading - might be part of summary
            if not line.strip():
                break
        elif line.strip().startswith('##'):
            # Hit a section heading, stop
            break
        elif not line.strip():
            # Empty line - check if next non-empty is a heading
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].startswith('#'):
                break
        i += 1
    
    if not summary_lines:
        return None, None
    
    summary_content = '\n'.join(summary_lines)
    
    # Determine if it's numpy or math
    is_numpy = bool(re.search(r'\bnp\.|numpy|torch\.', summary_content, re.IGNORECASE))
    
    return summary_content, 'numpy.md' if is_numpy else 'math.md'


def update_code_summary_section(content, summary_file):
    """Replace Code Summary section with include pattern."""
    lines = content.split('\n')
    new_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this is the Code Summary section
        if re.match(r'^\*\*Code Summary\*\*', line, re.IGNORECASE):
            # Add the new Code Summary header
            new_lines.append('**Code Summary**')
            new_lines.append('')
            
            # Add include pattern based on file type
            if summary_file == 'numpy.md':
                new_lines.append('<!-- include: summary/numpy.md if include_numpy -->')
            else:
                new_lines.append('<!-- include: summary/math.md if include_math -->')
            new_lines.append('')
            
            # Skip the old content
            i += 1
            # Skip empty lines
            while i < len(lines) and not lines[i].strip():
                i += 1
            # Skip all bullet points
            while i < len(lines) and lines[i].strip().startswith('-'):
                i += 1
            # Skip empty lines after summary
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue
        
        new_lines.append(line)
        i += 1
    
    return '\n'.join(new_lines)


def process_chapter(chapter_dir):
    """Process a single chapter directory."""
    chapter_dir = Path(chapter_dir)
    chapter_md = chapter_dir / f"{chapter_dir.name}.md"
    
    if not chapter_md.exists():
        # Try to find any .md file in the directory
        md_files = list(chapter_dir.glob("*.md"))
        if not md_files:
            return False, f"No markdown file found in {chapter_dir}"
        chapter_md = md_files[0]
    
    # Check if summary folder already exists
    summary_dir = chapter_dir / 'summary'
    if summary_dir.exists():
        return False, f"Summary folder already exists in {chapter_dir.name}"
    
    # Read chapter file
    try:
        content = chapter_md.read_text(encoding='utf-8')
    except Exception as e:
        return False, f"Error reading {chapter_md}: {e}"
    
    # Extract Code Summary
    summary_content, summary_file = extract_code_summary(content)
    
    if summary_content is None:
        return False, f"No Code Summary found in {chapter_dir.name}"
    
    # Create summary directory
    summary_dir.mkdir(exist_ok=True)
    
    # Write summary file
    summary_path = summary_dir / summary_file
    summary_path.write_text(summary_content + '\n', encoding='utf-8')
    
    # Update chapter file
    updated_content = update_code_summary_section(content, summary_file)
    chapter_md.write_text(updated_content, encoding='utf-8')
    
    return True, f"Processed {chapter_dir.name}: created summary/{summary_file}"


def main():
    """Process all chapter directories."""
    base_dir = Path(__file__).parent.parent
    
    # Find all chapter directories
    chapter_dirs = sorted([d for d in base_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('chapter') and d.name[7].isdigit()])
    
    print(f"Found {len(chapter_dirs)} chapter directories")
    print("=" * 60)
    
    processed = []
    skipped = []
    errors = []
    
    for chapter_dir in chapter_dirs:
        success, message = process_chapter(chapter_dir)
        if success:
            processed.append(chapter_dir.name)
            print(f"✓ {message}")
        else:
            if "already exists" in message or "No Code Summary" in message:
                skipped.append((chapter_dir.name, message))
                print(f"⊘ {message}")
            else:
                errors.append((chapter_dir.name, message))
                print(f"✗ {message}")
    
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Processed: {len(processed)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"  Errors: {len(errors)}")
    
    if processed:
        print(f"\nProcessed chapters:")
        for name in processed:
            print(f"  - {name}")
    
    if skipped:
        print(f"\nSkipped chapters:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")


if __name__ == '__main__':
    main()
