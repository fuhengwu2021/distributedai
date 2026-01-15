#!/usr/bin/env python3
"""
Create empty math.md files in summary folders and add math include before numpy include.
"""

import re
from pathlib import Path


def ensure_math_file(summary_dir):
    """Create math.md if it doesn't exist."""
    math_file = summary_dir / 'math.md'
    if not math_file.exists():
        math_file.write_text('', encoding='utf-8')
        return True
    return False


def update_chapter_file(chapter_md):
    """Add math include before numpy include if not already present."""
    try:
        content = chapter_md.read_text(encoding='utf-8')
    except Exception as e:
        return False, f"Error reading {chapter_md}: {e}"
    
    # Check if math include already exists
    if '<!-- include: summary/math.md if include_math -->' in content:
        return False, "Math include already exists"
    
    # Check if numpy include exists
    numpy_pattern = r'<!-- include: summary/numpy\.md if include_numpy -->'
    if not re.search(numpy_pattern, content):
        return False, "No numpy include found"
    
    # Add math include before numpy include
    new_content = re.sub(
        numpy_pattern,
        '<!-- include: summary/math.md if include_math -->\n<!-- include: summary/numpy.md if include_numpy -->',
        content
    )
    
    if new_content == content:
        return False, "Failed to update"
    
    chapter_md.write_text(new_content, encoding='utf-8')
    return True, "Updated"


def main():
    """Process all chapter directories."""
    base_dir = Path(__file__).parent.parent
    
    # Find all chapter directories
    chapter_dirs = sorted([d for d in base_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('chapter') and d.name[7].isdigit()])
    
    print(f"Found {len(chapter_dirs)} chapter directories")
    print("=" * 60)
    
    math_files_created = []
    math_files_existed = []
    chapters_updated = []
    chapters_skipped = []
    errors = []
    
    for chapter_dir in chapter_dirs:
        summary_dir = chapter_dir / 'summary'
        
        if not summary_dir.exists():
            continue
        
        # Create math.md if needed
        created = ensure_math_file(summary_dir)
        if created:
            math_files_created.append(chapter_dir.name)
            print(f"✓ Created math.md in {chapter_dir.name}/summary/")
        else:
            math_files_existed.append(chapter_dir.name)
        
        # Update chapter file
        chapter_md = chapter_dir / f"{chapter_dir.name}.md"
        if not chapter_md.exists():
            md_files = list(chapter_dir.glob("*.md"))
            if md_files:
                chapter_md = md_files[0]
            else:
                continue
        
        success, message = update_chapter_file(chapter_md)
        if success:
            chapters_updated.append(chapter_dir.name)
            print(f"✓ Updated {chapter_dir.name}: added math include")
        else:
            if "already exists" in message or "No numpy include" in message:
                chapters_skipped.append((chapter_dir.name, message))
            else:
                errors.append((chapter_dir.name, message))
    
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Math files created: {len(math_files_created)}")
    print(f"  Math files already existed: {len(math_files_existed)}")
    print(f"  Chapters updated: {len(chapters_updated)}")
    print(f"  Chapters skipped: {len(chapters_skipped)}")
    print(f"  Errors: {len(errors)}")
    
    if math_files_created:
        print(f"\nCreated math.md in:")
        for name in math_files_created:
            print(f"  - {name}/summary/")
    
    if chapters_updated:
        print(f"\nUpdated chapters:")
        for name in chapters_updated:
            print(f"  - {name}")


if __name__ == '__main__':
    main()
