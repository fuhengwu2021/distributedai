#!/usr/bin/env python3
"""
Remove alt attribute from \includegraphics commands in LaTeX files.
Pandoc 3.8+ automatically adds alt attribute from markdown alt text,
but LaTeX \includegraphics doesn't support the alt key.
"""
import sys
import os
import re

def remove_alt_from_includegraphics(content):
    """Remove alt={...} from all \includegraphics commands"""
    # More aggressive: find all includegraphics commands and process them
    # This handles all variations including nested braces
    
    # Find all \includegraphics[options]{file} patterns
    def process_includegraphics(match):
        full_match = match.group(0)
        # If alt= is in the options, remove it
        if 'alt=' in full_match:
            # Extract the parts
            # Match: \includegraphics[options]{file}
            parts = re.match(r'(\\includegraphics)(\[.*?\])(\{.*?\})', full_match)
            if parts:
                prefix = parts.group(1)
                options = parts.group(2)
                file_part = parts.group(3)
                
                # Remove alt={...} from options (handle all variations)
                # Remove with comma before
                options = re.sub(r',\s*alt\s*=\s*\{[^}]*\}', '', options)
                # Remove with comma after
                options = re.sub(r'alt\s*=\s*\{[^}]*\}\s*,', '', options)
                # Remove without comma (standalone)
                options = re.sub(r'alt\s*=\s*\{[^}]*\}', '', options)
                
                # Clean up
                options = options.strip()
                # Remove empty brackets
                if options == '[]' or options == '[' or options == ']':
                    return prefix + file_part
                # Clean up commas
                options = re.sub(r',\s*,+', ',', options)
                options = re.sub(r'\[\s*,+', '[', options)
                options = re.sub(r',+\s*\]', ']', options)
                
                return prefix + options + file_part
        return full_match
    
    # Process all includegraphics commands
    content = re.sub(r'\\includegraphics\[[^\]]*\]\{[^}]+\}', process_includegraphics, content)
    
    # Also handle cases where includegraphics might span lines
    # Simple approach: remove alt= anywhere it appears near includegraphics
    lines = content.split('\n')
    fixed_lines = []
    for line in lines:
        if 'includegraphics' in line and 'alt=' in line:
            # Remove alt={...} pattern
            line = re.sub(r',?\s*alt\s*=\s*\{[^}]*\}', '', line)
            line = re.sub(r'alt\s*=\s*\{[^}]*\},?', '', line)
            # Clean up
            line = re.sub(r',\s*,+', ',', line)
            line = re.sub(r'\[\s*,+', '[', line)
            line = re.sub(r',+\s*\]', ']', line)
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: remove_alt_attribute.py <latex_file>")
        sys.exit(1)
    
    tex_file = sys.argv[1]
    if not os.path.exists(tex_file):
        sys.exit(0)
    
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    fixed_content = remove_alt_from_includegraphics(content)
    
    # Debug: check if alt= still exists
    if 'alt=' in fixed_content and 'includegraphics' in fixed_content:
        # More aggressive: use sed-like approach - remove alt= anywhere near includegraphics
        # Split by includegraphics and process each occurrence
        parts = fixed_content.split('\\includegraphics')
        result_parts = [parts[0]]  # First part (before first includegraphics)
        for part in parts[1:]:
            # Find the options bracket and remove alt= from it
            # Match [options]{file} pattern
            if '[' in part and ']' in part:
                bracket_start = part.find('[')
                bracket_end = part.find(']', bracket_start)
                if bracket_end > bracket_start:
                    options = part[bracket_start+1:bracket_end]
                    file_part = part[bracket_end+1:]
                    # Remove alt= from options
                    options = re.sub(r',?\s*alt\s*=\s*\{[^}]*\}', '', options)
                    options = re.sub(r'alt\s*=\s*\{[^}]*\},?', '', options)
                    options = options.strip()
                    # Clean up
                    if options and not options.startswith(','):
                        options = '[' + options + ']'
                    else:
                        options = ''
                    result_parts.append('\\includegraphics' + options + file_part)
                else:
                    result_parts.append('\\includegraphics' + part)
            else:
                result_parts.append('\\includegraphics' + part)
        fixed_content = ''.join(result_parts)
    
    if fixed_content != original_content:
        with open(tex_file, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
